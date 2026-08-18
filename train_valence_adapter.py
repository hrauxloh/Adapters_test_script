"""Train a standalone adapter on valence (sentiment intensity) as an ordinal
regression task, using voc_train.csv / voc_val.csv / voc_test.csv
(SemEval-2018 Task 1, V-oc: text + Valence_Code from -3 to +3).

This is a separate, parallel experiment to the original sst2/emotion fusion
model — it does not touch or overwrite that model's output. The adapter
trained here is meant to later be fused (alongside a matching
train_emotion_adapter.py adapter) into its own new polarization model, for
comparison against the AdapterHub-based one.

Valence_Code is treated as a genuinely ordinal target via regression (a
single continuous output trained with MSE loss), not as 7 unordered classes
— plain multi-class classification would penalize a prediction of -3 for a
true value of +3 exactly as much as a prediction of +2, which throws away
the ordering. Regression keeps predictions further from the true value
penalized more, without requiring a custom ordinal-classification head.

Usage:
    python train_valence_adapter.py
"""

import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from adapters import AutoAdapterModel, AdapterTrainer, SeqBnConfig

from data import load_split, tokenize_dataset

MODEL_NAME = "bert-base-uncased"
ADAPTER_NAME = "valence"
LABEL_COLUMN = "Valence_Code"
MIN_CODE = -3
MAX_CODE = 3
NUM_CLASSES = MAX_CODE - MIN_CODE + 1  # 7, used only for clipping predictions


class RegressionAdapterTrainer(AdapterTrainer):
    """AdapterTrainer with MSE regression loss instead of classification loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.mse_loss(logits.squeeze(-1), labels.float())
        return (loss, outputs) if return_outputs else loss


def build_model():
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    model.add_adapter(ADAPTER_NAME, config=SeqBnConfig())
    # num_labels=1 -> a single raw (unbounded) output, i.e. a regression score.
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
    return dataset.map(lambda ex: {LABEL_COLUMN: ex[LABEL_COLUMN] - MIN_CODE})


def compute_metrics(eval_pred):
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
