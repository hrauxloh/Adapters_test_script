"""
WHAT THIS SCRIPT DOES
----------------------
Trains a brand-new adapter — from nothing, no pretrained head start — to
detect 11 different emotions in text at once (SemEval-2018 Task 1, E-c:
anger, anticipation, disgust, fear, joy, love, optimism, pessimism,
sadness, surprise, trust). Uses Ec_train.csv / Ec_val.csv / Ec_test.csv,
and never touches the polarization data at all.

This is one of three "build an adapter from scratch" scripts in this
project (the others are train_group_adapter.py and
train_valence_adapter.py) — all three follow the same shape: attach a
fresh adapter, add a temporary head just to train it, then keep only the
adapter once it's done.

What's different here: each row of Ec_*.csv can have ANY number of the 11
emotions present at once (including zero, including several together) —
it's not "pick the one right answer" the way train_group_adapter.py's
yes/no task is. So instead of one classification head, this uses a
multi-label head — effectively 11 independent yes/no decisions made
together — so the adapter can learn how emotions relate to each other,
rather than training 11 separate single-emotion adapters.

Ec_train.csv / Ec_val.csv already come as separate files (SemEval provided
them pre-split) — unlike train_group_adapter.py, this script doesn't need
to carve its own validation slice out of the training file.

Menu of what happens when you run this file, in order:
  1. Load Ec_train.csv and Ec_val.csv, combining the 11 emotion columns
     into one label per row (e.g. [0, 1, 0, 0, 1, ...]).
  2. Build the model: attach one fresh, empty adapter to frozen BERT, plus
     a temporary head with 11 independent yes/no outputs.
  3. Train ONLY the adapter + temporary head, checking against the
     validation slice, until it stops improving.
  4. Report how well it's doing across all 11 emotions.
  5. Save the adapter alone — the temporary head is thrown away, the same
     way sst2/emotion's original heads were thrown away when we loaded them.

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

MODEL_NAME = "bert-base-uncased"  # the frozen base model the adapter attaches to
ADAPTER_NAME = "emotion_ec"  # what this adapter will be called when saved/reloaded
EMOTION_COLUMNS = [
    "anger", "anticipation", "disgust", "fear", "joy", "love",
    "optimism", "pessimism", "sadness", "surprise", "trust",
]  # the 11 columns in Ec_*.csv we're predicting, together


def build_model():
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    model.add_adapter(ADAPTER_NAME, config=SeqBnConfig())
    # multilabel=True switches the head from "pick one class" (softmax) to
    # "independently decide yes/no for each of the 11" (sigmoid) — this one
    # flag is what makes it multi-label instead of a normal classifier.
    # Compare against train_group_adapter.py (num_labels=2, plain yes/no)
    # and train_valence_adapter.py (num_labels=1, a single number) to see
    # what changes for a different kind of label.
    model.add_classification_head(ADAPTER_NAME, num_labels=len(EMOTION_COLUMNS), multilabel=True)

    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)
    model.active_head = ADAPTER_NAME

    return model


def compute_metrics(eval_pred):
    # Turns each of the 11 raw scores into a yes/no (via sigmoid + a 0.5
    # cutoff), then reports two versions of F1:
    #   - f1_micro: treats every individual yes/no decision as equally
    #     important, across all examples and all 11 emotions together.
    #   - f1_macro: treats every EMOTION as equally important, so a rare
    #     emotion like "trust" counts as much as a common one like "joy",
    #     instead of being drowned out by it.
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid: turns raw scores into 0-1 probabilities
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
