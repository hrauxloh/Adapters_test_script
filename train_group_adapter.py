"""Train a standalone adapter on the group-reference-detection task
(Group_bool: does this text reference a social/political group), completely
independent of the polarization task.

This is Step 1 of the group-identity ablation plan: the adapter trained here
gets saved (its temporary classification head is discarded) and later reused
frozen, alongside sst2/emotion, in fusion models that predict `polarization`
— exactly like sst2/emotion themselves are used, except this one has no
pretrained starting point, so it's trained from scratch on
group_reference_data.csv here first.

Uses the same architecture (Pfeiffer/SeqBn bottleneck) as sst2/emotion, so
it's dimensionally compatible for fusion, and the same
validation-split/early-stopping discipline as the rest of the pipeline —
this script never needs eng_train.csv/eng_test.csv, it only touches its own
group-reference data.

Usage:
    python train_group_adapter.py --train-file group_reference_data.csv
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from adapters import AutoAdapterModel, AdapterTrainer, SeqBnConfig

from data import load_train_val_split, tokenize_dataset

MODEL_NAME = "bert-base-uncased"
ADAPTER_NAME = "group"
LABEL_COLUMN = "Group_bool"


def build_model():
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    # Fresh, randomly-initialized adapter — same architecture family (Pfeiffer/
    # SeqBn) as sst2/emotion, so it's compatible for fusion later. Unlike
    # sst2/emotion, there's no pretrained starting point: it has to learn the
    # task from this dataset alone.
    model.add_adapter(ADAPTER_NAME, config=SeqBnConfig())
    model.add_classification_head(ADAPTER_NAME, num_labels=2)

    # Freeze the base model; only this adapter + its (temporary) head train.
    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)
    model.active_head = ADAPTER_NAME

    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="group_reference_data.csv")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", default="output_group")
    parser.add_argument("--epochs", type=int, default=10, help="Upper bound; early stopping usually stops sooner.")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use fp16 mixed precision training. Defaults to on when a GPU is available.",
    )
    args = parser.parse_args()

    use_fp16 = args.fp16 if args.fp16 is not None else torch.cuda.is_available()
    if args.fp16 and not torch.cuda.is_available():
        print("Warning: --fp16 requested but no GPU is available; running in full precision.")
        use_fp16 = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_split, val_split = load_train_val_split(
        args.train_file, args.val_fraction, seed=args.seed, label_column=LABEL_COLUMN
    )
    train_dataset = tokenize_dataset(train_split, tokenizer, args.max_length, label_column=LABEL_COLUMN)
    val_dataset = tokenize_dataset(val_split, tokenizer, args.max_length, label_column=LABEL_COLUMN)

    model = build_model()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        dataloader_num_workers=2,
        seed=args.seed,
        report_to="none",
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    metrics = trainer.evaluate()

    majority_class_baseline = max(np.bincount(val_split[LABEL_COLUMN])) / len(val_split)
    print("Validation metrics:", metrics)
    print(f"(majority-class baseline accuracy on this validation split: {majority_class_baseline:.3f})")
    if metrics["eval_accuracy"] <= majority_class_baseline:
        print(
            "WARNING: the adapter did not beat the majority-class baseline — "
            "it likely hasn't learned a useful signal yet. Consider more data, "
            "more epochs/patience, or a different learning rate before using "
            "this adapter downstream."
        )

    # Only the adapter itself is kept — the temporary classification head above
    # was scaffolding to train it and isn't used once it's fused with others.
    model.save_adapter(f"{args.output_dir}/{ADAPTER_NAME}", ADAPTER_NAME, with_head=False)
    print(f"\nSaved adapter to {args.output_dir}/{ADAPTER_NAME}")


if __name__ == "__main__":
    main()
