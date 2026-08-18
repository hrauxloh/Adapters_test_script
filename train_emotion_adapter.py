"""Train a standalone adapter on multi-label emotion detection, using
Ec_train.csv / Ec_val.csv / Ec_test.csv (SemEval-2018 Task 1, E-c: text plus
11 binary emotion columns — anger, anticipation, disgust, fear, joy, love,
optimism, pessimism, sadness, surprise, trust — where any number of them,
including zero, can be present at once).

This is a separate, parallel experiment to the original sst2/emotion fusion
model — it does not touch or overwrite that model's output. The adapter
trained here is meant to later be fused (alongside a matching
train_valence_adapter.py adapter) into its own new polarization model, for
comparison against the AdapterHub-based one.

Uses a genuine multi-label head (independent sigmoid output per emotion,
binary cross-entropy loss) rather than 11 separate adapters — one adapter
can jointly learn correlations between emotions, and is far cheaper to
train than 11 standalone ones.

Usage:
    python train_emotion_adapter.py
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from adapters import AutoAdapterModel, AdapterTrainer, SeqBnConfig

from data import load_multilabel_split, tokenize_multilabel_dataset

MODEL_NAME = "bert-base-uncased"
ADAPTER_NAME = "emotion_ec"
EMOTION_COLUMNS = [
    "anger", "anticipation", "disgust", "fear", "joy", "love",
    "optimism", "pessimism", "sadness", "surprise", "trust",
]


def build_model():
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    model.add_adapter(ADAPTER_NAME, config=SeqBnConfig())
    model.add_classification_head(ADAPTER_NAME, num_labels=len(EMOTION_COLUMNS), multilabel=True)

    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)
    model.active_head = ADAPTER_NAME

    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="Ec_train.csv")
    parser.add_argument("--val-file", default="Ec_val.csv")
    parser.add_argument("--output-dir", default="output_emotion_ec")
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

    train_split = load_multilabel_split(args.train_file, EMOTION_COLUMNS)
    val_split = load_multilabel_split(args.val_file, EMOTION_COLUMNS)
    train_dataset = tokenize_multilabel_dataset(train_split, tokenizer, args.max_length)
    val_dataset = tokenize_multilabel_dataset(val_split, tokenizer, args.max_length)

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
        metric_for_best_model="f1_micro",
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

    # A trivial "predict nothing" baseline: micro-F1 is 0 whenever every label
    # is predicted negative, so any positive f1_micro already beats it. What's
    # more informative is comparing against always predicting each label's
    # majority class (usually all-negative, given how sparse most emotions are).
    print("Validation metrics:", metrics)
    if metrics["eval_f1_micro"] <= 0.0:
        print(
            "WARNING: micro-F1 is zero or undefined — the adapter is not "
            "predicting any positive labels. It likely hasn't learned a "
            "useful signal yet."
        )

    model.save_adapter(f"{args.output_dir}/{ADAPTER_NAME}", ADAPTER_NAME, with_head=False)
    print(f"\nSaved adapter to {args.output_dir}/{ADAPTER_NAME}")


if __name__ == "__main__":
    main()
