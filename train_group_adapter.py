"""
WHAT THIS SCRIPT DOES
----------------------
Trains a brand-new adapter — from nothing, no pretrained head start — to
detect whether a piece of text references a social/political group at all
(the Group_bool column: yes/no). This is completely separate from the
polarization task; it only ever touches group_reference_data.csv.

This is one of three "build an adapter from scratch" scripts in this
project (the others are train_valence_adapter.py and
train_emotion_adapter.py) — all three follow the same shape: attach a
fresh adapter, add a temporary head just to train it, then keep only the
adapter once it's done. This one is the simplest of the three (plain
yes/no classification), since valence needs a regression head and emotion
needs a multi-label head instead.

group_reference_data.csv is a single file, not already split into
train/validation — so unlike the valence/emotion scripts, this one carves
its own validation slice out of --train-file using the same
load_train_val_split() helper the polarization model uses.

Menu of what happens when you run this file, in order:
  1. Load group_reference_data.csv and split off a validation slice.
  2. Build the model: attach one fresh, empty adapter to frozen BERT, plus
     a temporary yes/no classification head on top.
  3. Train ONLY the adapter + temporary head, checking against the
     validation slice, until it stops improving.
  4. Compare the result to a "just guess the most common answer" baseline,
     and warn if the adapter didn't beat it.
  5. Save the adapter alone — the temporary head is thrown away, the same
     way sst2/emotion's original heads were thrown away when we loaded them.

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

MODEL_NAME = "bert-base-uncased"  # the frozen base model the adapter attaches to
ADAPTER_NAME = "group"  # what this adapter will be called when saved/reloaded
LABEL_COLUMN = "Group_bool"  # the column in group_reference_data.csv we're predicting


def build_model():
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    # Fresh, randomly-initialized adapter — same architecture family (Pfeiffer/
    # SeqBn) as sst2/emotion, so it's compatible for fusion later. Unlike
    # sst2/emotion, there's no pretrained starting point: it has to learn the
    # task from this dataset alone.
    model.add_adapter(ADAPTER_NAME, config=SeqBnConfig())
    # A plain 2-class head (yes/no) — this is the "standard" version of the
    # three adapter scripts; compare against train_valence_adapter.py
    # (1 continuous output) and train_emotion_adapter.py (11 independent
    # yes/no outputs) to see what changes for a different kind of label.
    model.add_classification_head(ADAPTER_NAME, num_labels=2)

    # Freeze the base model; only this adapter + its (temporary) head train.
    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)
    model.active_head = ADAPTER_NAME

    return model


def compute_metrics(eval_pred):
    # Turns raw predictions + correct answers into accuracy/precision/recall/F1.
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

    # group_reference_data.csv is one file — split off a validation slice
    # ourselves (stratified, so the yes/no balance matches in both halves).
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
