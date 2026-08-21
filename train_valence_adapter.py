"""
WHAT THIS SCRIPT DOES
----------------------
Trains a brand-new adapter — from nothing, no pretrained head start — to
predict valence: how negative or positive a piece of text is, on a -3 to +3
scale (SemEval-2018 Task 1, V-oc). Uses voc_train.csv / voc_val.csv /
voc_test.csv, and never touches the polarization data at all.

This is one of three "build an adapter from scratch" scripts in this
project (the others are train_group_adapter.py and
train_emotion_adapter.py) — all three follow the same shape: attach a
fresh adapter, add a temporary head just to train it, then keep only the
adapter once it's done.

What's different here: Valence_Code is ordinal (an ordered scale), not a
set of unrelated categories — mistaking -3 for +3 is a much bigger error
than mistaking it for +2. Plain classification (like train_group_adapter.py
uses) would treat every wrong guess as equally wrong, which throws that
ordering away. So instead of a normal classification head, this one
predicts a single number (regression), trained so that predictions further
from the true value are penalized more.

voc_train.csv / voc_val.csv already come as separate files (SemEval
provided them pre-split) — unlike train_group_adapter.py, this script
doesn't need to carve its own validation slice out of the training file.

Menu of what happens when you run this file, in order:
  1. Load voc_train.csv and voc_val.csv, and shift Valence_Code from
     -3..+3 to 0..6 (an implementation detail, explained below).
  2. Build the model: attach one fresh, empty adapter to frozen BERT, plus
     a temporary head that outputs a single number instead of class scores.
  3. Train ONLY the adapter + temporary head, using a "how far off was the
     guess" loss instead of the usual right/wrong classification loss.
  4. Compare the result to a "just guess the average" baseline, and warn
     if the adapter didn't beat it.
  5. Save the adapter alone — the temporary head is thrown away, the same
     way sst2/emotion's original heads were thrown away when we loaded them.

Usage:
    python train_valence_adapter.py
"""

import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from adapters import AutoAdapterModel, AdapterTrainer, SeqBnConfig

from data import load_split, tokenize_dataset

MODEL_NAME = "bert-base-uncased"  # the frozen base model the adapter attaches to
ADAPTER_NAME = "valence"  # what this adapter will be called when saved/reloaded
LABEL_COLUMN = "Valence_Code"  # the column in voc_*.csv we're predicting
MIN_CODE = -3
MAX_CODE = 3
NUM_CLASSES = MAX_CODE - MIN_CODE + 1  # 7, used only for clipping predictions back to a valid class


class RegressionAdapterTrainer(AdapterTrainer):
    """AdapterTrainer with MSE regression loss instead of classification loss."""

    # This builds on the library's normal AdapterTrainer, only changing how
    # the "how wrong was the model" score gets calculated — for a number
    # line target like this, we use mean-squared-error (MSE) instead of the
    # usual classification loss.
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # pull the correct valence scores out of the batch
        outputs = model(**inputs)  # ask the model for its prediction
        logits = outputs.get("logits")  # here this is one raw number per example, not class scores
        loss = torch.nn.functional.mse_loss(logits.squeeze(-1), labels.float())
        return (loss, outputs) if return_outputs else loss


def build_model():
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    model.add_adapter(ADAPTER_NAME, config=SeqBnConfig())
    # num_labels=1 -> a single raw (unbounded) output, i.e. a regression score,
    # instead of a score per class. Compare against train_group_adapter.py
    # (num_labels=2, plain yes/no) and train_emotion_adapter.py
    # (num_labels=11, multilabel=True) to see what changes for a different
    # kind of label.
    model.add_classification_head(ADAPTER_NAME, num_labels=1)

    model.train_adapter(ADAPTER_NAME)
    model.set_active_adapters(ADAPTER_NAME)
    model.active_head = ADAPTER_NAME

    return model


def remap_labels(dataset):
    """Shift Valence_Code from [-3, 3] to [0, 6] (0-indexed, for consistency
    with how classification labels are handled elsewhere), while keeping it
    a plain integer — the regression loss casts to float at compute time.
    """
    # e.g. -3 becomes 0, 0 becomes 3, +3 becomes 6 — just relabeling the same
    # 7 points, nothing about their order or spacing changes.
    return dataset.map(lambda ex: {LABEL_COLUMN: ex[LABEL_COLUMN] - MIN_CODE})


def compute_metrics(eval_pred):
    # Regression doesn't have "precision/recall" the way yes/no classification
    # does, so this reports different, more appropriate numbers instead:
    #   - mae: mean absolute error — on average, how many points off was
    #     each guess? Lower is better, and this is the main number to watch.
    #   - exact_accuracy: how often the rounded guess matched exactly.
    #   - within_one_accuracy: how often the rounded guess was at most 1
    #     point off — a more forgiving, still-useful measure for a 7-point
    #     ordinal scale, where being "close" genuinely means something.
    preds, labels = eval_pred
    preds = preds.squeeze(-1)
    mae = float(np.mean(np.abs(preds - labels)))
    rounded = np.clip(np.round(preds), 0, NUM_CLASSES - 1)
    exact_accuracy = float(np.mean(rounded == labels))
    within_one_accuracy = float(np.mean(np.abs(rounded - labels) <= 1))
    return {
        "mae": mae,
        "exact_accuracy": exact_accuracy,
        "within_one_accuracy": within_one_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="voc_train.csv")
    parser.add_argument("--val-file", default="voc_val.csv")
    parser.add_argument("--output-dir", default="output_valence")
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

    train_split = remap_labels(load_split(args.train_file, label_column=LABEL_COLUMN))
    val_split = remap_labels(load_split(args.val_file, label_column=LABEL_COLUMN))
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

    model.save_adapter(f"{args.output_dir}/{ADAPTER_NAME}", ADAPTER_NAME, with_head=False)
    print(f"\nSaved adapter to {args.output_dir}/{ADAPTER_NAME}")


if __name__ == "__main__":
    main()
