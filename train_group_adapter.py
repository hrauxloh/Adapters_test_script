"""
WHAT THIS SCRIPT DOES
----------------------
Trains a brand-new adapter — from nothing, no pretrained head start — to
predict how strongly a piece of text expresses "us vs. them" / in-group vs.
out-group framing, as a continuous score from 0 (none) to 1 (strong). Uses
UsVsThem_train.csv / UsVsThem_valid.csv, and never touches the
polarization data at all.

This is one of three "build an adapter from scratch" scripts in this
project (the others are train_valence_adapter.py and
train_emotion_adapter.py) — all three follow the same shape: attach a
fresh adapter, add a temporary head just to train it, then keep only the
adapter once it's done.

Like train_valence_adapter.py, this predicts a continuous score rather
than a yes/no category, so it uses a regression head and a "how far off
was the guess" loss instead of ordinary classification.

UsVsThem_train.csv / UsVsThem_valid.csv already come as separate files (like
train_valence_adapter.py's vreg files) — so neither of those two scripts
needs to carve its own validation slice out of the training file, unlike
train_baseline.py/train_fusion.py do for the polarization data.

Menu of what happens when you run this file, in order:
  1. Load UsVsThem_train.csv and UsVsThem_valid.csv.
  2. Build the model: attach one fresh, empty adapter to frozen BERT, plus
     a temporary head that outputs a single number instead of class scores.
  3. Train ONLY the adapter + temporary head, using a "how far off was the
     guess" loss instead of the usual right/wrong classification loss.
  4. Compare the result to a "just guess the average" baseline, and warn
     if the adapter didn't beat it.
  5. Save the adapter alone — the temporary head is thrown away, the same
     way sst2/emotion's original heads were thrown away when we loaded them.

Usage:
    python train_group_adapter.py
"""

import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from adapters import AutoAdapterModel, AdapterTrainer, SeqBnConfig

from data import load_split, tokenize_dataset

MODEL_NAME = "bert-base-uncased"  # the frozen base model the adapter attaches to
ADAPTER_NAME = "group"  # what this adapter will be called when saved/reloaded
LABEL_COLUMN = "usVSthem_scale"  # the column in UsVsThem_*.csv we're predicting (0..1)


class RegressionAdapterTrainer(AdapterTrainer):
    """AdapterTrainer with MSE regression loss instead of classification loss."""

    # This builds on the library's normal AdapterTrainer, only changing how
    # the "how wrong was the model" score gets calculated — for a number
    # line target like this, we use mean-squared-error (MSE) instead of the
    # usual classification loss.
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # pull the correct us-vs-them scores out of the batch
        outputs = model(**inputs)  # ask the model for its prediction
        logits = outputs.get("logits")  # here this is one raw number per example, not class scores
        loss = torch.nn.functional.mse_loss(logits.squeeze(-1), labels.float())
        return (loss, outputs) if return_outputs else loss


def build_model():
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    # Fresh, randomly-initialized adapter — same architecture family (Pfeiffer/
    # SeqBn) as sst2/emotion, so it's compatible for fusion later. Unlike
    # sst2/emotion, there's no pretrained starting point: it has to learn the
    # task from this dataset alone.
    model.add_adapter(ADAPTER_NAME, config=SeqBnConfig())
    # num_labels=1 -> a single raw (unbounded) output, i.e. a regression score,
    # instead of a score per class. Compare against train_emotion_adapter.py
    # (num_labels=11, multilabel=True) to see what changes for a different
    # kind of label.
    model.add_classification_head(ADAPTER_NAME, num_labels=1)

    # Freeze the base model; only this adapter + its (temporary) head train.
    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)
    model.active_head = ADAPTER_NAME

    return model


def compute_metrics(eval_pred):
    # Regression doesn't have "precision/recall" the way yes/no classification
    # does, so this reports different, more appropriate numbers instead:
    #   - mae: mean absolute error — on average, how far off was each guess,
    #     in the same 0-1 units as the label? Lower is better, and this is
    #     the main number used to pick the best checkpoint.
    #   - correlation: how well the model's ranking of examples (low to high
    #     us-vs-them framing) matches the true ranking, regardless of the
    #     exact scale — 1.0 is a perfect match, 0 is no relationship.
    preds, labels = eval_pred
    preds = preds.squeeze(-1)
    mae = float(np.mean(np.abs(preds - labels)))
    correlation = float(np.corrcoef(preds, labels)[0, 1])
    return {
        "mae": mae,
        "correlation": correlation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="UsVsThem_train.csv")
    parser.add_argument("--val-file", default="UsVsThem_valid.csv")
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

    # label_dtype=float keeps the continuous 0..1 score intact — the default
    # int cast (used for yes/no labels elsewhere) would truncate it to 0.
    train_split = load_split(args.train_file, label_column=LABEL_COLUMN, label_dtype=float)
    val_split = load_split(args.val_file, label_column=LABEL_COLUMN, label_dtype=float)
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
        metric_for_best_model="mae",
        greater_is_better=False,
        fp16=use_fp16,
        dataloader_num_workers=2,
        seed=args.seed,
        report_to="none",
    )

    trainer = RegressionAdapterTrainer(
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

    naive_baseline_mae = float(np.mean(np.abs(np.array(val_split[LABEL_COLUMN]) - np.mean(train_split[LABEL_COLUMN]))))
    print("Validation metrics:", metrics)
    print(f"(naive baseline MAE, always predicting the train-set mean: {naive_baseline_mae:.3f} — lower is better, so the adapter should score below this)")
    if metrics["eval_mae"] >= naive_baseline_mae:
        print(
            "WARNING: the adapter did not beat the naive mean-prediction baseline — "
            "it likely hasn't learned a useful signal yet."
        )

    # Only the adapter itself is kept — the temporary regression head above
    # was scaffolding to train it and isn't used once it's fused with others.
    model.save_adapter(f"{args.output_dir}/{ADAPTER_NAME}", ADAPTER_NAME, with_head=False)
    print(f"\nSaved adapter to {args.output_dir}/{ADAPTER_NAME}")


if __name__ == "__main__":
    main()
