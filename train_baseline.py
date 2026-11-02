"""Baseline for comparison against train_fusion.py: trains a fresh binary
classification head directly on frozen bert-base-uncased, with NO adapters
and NO fusion layer. Everything else (data, frozen base, hyperparameters,
metrics) is kept identical to train_fusion.py, so the difference in test-set
metrics between the two scripts isolates the value the sst2/emotion adapter
fusion actually adds over a plain frozen-BERT baseline.

Usage:
    python train_baseline.py --train-file eng_train.csv --test-file eng_test.csv
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, Trainer, TrainingArguments
from adapters import AutoAdapterModel

from data import load_split, tokenize_dataset

MODEL_NAME = "bert-base-uncased"
HEAD_NAME = "polarization"


def build_model():
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)
    model.add_classification_head(HEAD_NAME, num_labels=2)
    model.active_head = HEAD_NAME

    # Freeze the base model so only the new head is trained — the same
    # amount of "new" capacity as train_fusion.py's fusion+head, minus the
    # two adapters and the fusion layer itself.
    for param in model.bert.parameters():
        param.requires_grad = False

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
    parser.add_argument("--train-file", default="eng_train.csv")
    parser.add_argument("--test-file", default="eng_test.csv")
    parser.add_argument("--output-dir", default="output_baseline")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=96)
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

    train_dataset = tokenize_dataset(load_split(args.train_file), tokenizer, args.max_length)
    test_dataset = tokenize_dataset(load_split(args.test_file), tokenizer, args.max_length)

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
        # The plain Trainer (unlike AdapterTrainer) doesn't auto-detect the
        # label column for an adapters-library head model, so it has to be
        # told explicitly, both to keep it during the forward pass and to
        # compute eval metrics against it.
        remove_unused_columns=False,
        label_names=["labels"],
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=use_fp16,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("Baseline (no adapters) test set metrics:", metrics)

    model.save_head(f"{args.output_dir}/head", HEAD_NAME)


if __name__ == "__main__":
    main()
