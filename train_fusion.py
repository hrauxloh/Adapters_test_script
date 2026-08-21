"""
WHAT THIS SCRIPT DOES
----------------------
Fuses several pre-trained "adapter" modules (small bolt-on pieces attached
to bert-base-uncased) into one model, and trains a new head on top of that
fusion to predict the binary `polarization` label.

By default it fuses two adapters downloaded from AdapterHub (sentiment,
emotion) — but the --adapters setting can point it at any adapters at all,
including ones trained from scratch elsewhere in this project (see
train_group_adapter.py / train_valence_adapter.py / train_emotion_adapter.py).

Model selection (early stopping, "best epoch") is done against a validation
split held out from --train-file — NOT eng_test.csv. This keeps the test
set untouched for a single final evaluation (see evaluate_test.py), instead
of implicitly tuning to it every time a setting is changed.

Menu of what happens when you run this file, in order:
  1. Read the settings you passed on the command line (--adapters,
     --lr, --epochs, etc.)
  2. Load the training data and carve off a validation slice.
  3. Build the model: load each adapter frozen, add a fusion layer that
     combines them, add a brand-new head for polarization.
  4. Train ONLY the fusion layer + new head (everything else stays frozen)
     until it stops improving.
  5. Save the trained fusion layer, head, and a note of which adapters
     were used, so later scripts can reload the exact same model.

Usage:
    python train_fusion.py --train-file eng_train.csv
"""

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from adapters import AutoAdapterModel, AdapterTrainer, Fuse

from data import load_train_val_split, tokenize_dataset

MODEL_NAME = "bert-base-uncased"  # the frozen base model every adapter attaches to
HEAD_NAME = "polarization"  # name of the new prediction head we train
DEFAULT_ADAPTERS = [
    "sst2=AdapterHub/bert-base-uncased-pf-sst2",
    "emotion=AdapterHub/bert-base-uncased-pf-emotion",
]  # which adapters to fuse if --adapters isn't given


def parse_adapter_specs(specs):
    """Parse ["name=source", ...] into [(name, source), ...].

    `source` can be an AdapterHub/HF Hub id (e.g. AdapterHub/bert-base-uncased-pf-sst2)
    or a local path to a previously saved adapter (e.g. output_valence/valence) —
    model.load_adapter() already handles both transparently.
    """
    parsed = []
    for spec in specs:
        # Splits "name=source" at the first "=" into the two pieces.
        name, sep, source = spec.partition("=")
        if not sep or not name or not source:
            raise ValueError(f"Invalid --adapters entry {spec!r}, expected name=source")
        parsed.append((name, source))
    return parsed  # a list of (name, source) pairs


class ConfigurableLossTrainer(AdapterTrainer):
    """AdapterTrainer with optional class-weighted / label-smoothed cross-entropy."""

    # This builds on the library's normal AdapterTrainer, only changing how
    # the "how wrong was the model" score (the loss) gets calculated.

    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        # *args/**kwargs = accept anything a normal AdapterTrainer accepts,
        # and hand it straight through — we're only adding two extra settings.
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # This function measures how wrong the model's predictions were.
        labels = inputs.pop("labels")  # pull the correct answers out of the batch
        outputs = model(**inputs)  # ask the model for its predictions
        logits = outputs.get("logits")  # the model's raw, un-normalized scores per class
        # class_weights (if set) makes mistakes on the rarer class count for more,
        # so the model can't just get a good score by always guessing the common one.
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        # label_smoothing softens the target from e.g. [0, 1] to [0.05, 0.95], so the
        # model isn't pushed toward 100% certainty — this tends to help it generalize.
        # It defaults to 0.0 here, meaning it's off unless --label-smoothing is set.
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=self.label_smoothing)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_model(adapter_specs):
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)  # load the frozen base model

    # Load each adapter, frozen, without its original head (if it has one).
    for name, source in adapter_specs:
        model.load_adapter(source, load_as=name, with_head=False)

    fusion_setup = Fuse(*[name for name, _ in adapter_specs])

    # Add an AdapterFusion layer combining all of them, plus a fresh binary head for polarization.
    model.add_adapter_fusion(fusion_setup)
    model.add_classification_head(HEAD_NAME, num_labels=2)

    # Freeze the base model and the source adapters; only the fusion layer
    # and the new head remain trainable.
    model.train_adapter_fusion(fusion_setup)
    model.set_active_adapters(fusion_setup)
    model.active_head = HEAD_NAME

    return model, fusion_setup


def compute_metrics(eval_pred):
    # Turns the model's raw predictions + the correct answers into the
    # accuracy/precision/recall/F1 numbers printed during training.
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


def build_arg_parser():
    # This sets up every command-line option this script accepts.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="eng_train.csv")  # the CSV to train on
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of --train-file held out for validation/early stopping.")  # 20% held out by default
    parser.add_argument("--train-fraction", type=float, default=1.0, help="Subsample the remaining training portion (after the val split) to this fraction, for cheap sweep runs.")  # 1.0 = use ALL of the remaining training data (the default); lower it (e.g. 0.5) only for quick, cheap test runs
    parser.add_argument("--output-dir", default="output")  # where trained files get saved
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True, help="Save the trained fusion+head to --output-dir. Turn off for throwaway sweep trials.")
    parser.add_argument("--epochs", type=int, default=10, help="Upper bound on epochs; early stopping usually stops sooner.")  # ceiling on epochs, not a fixed target
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience, in evals (epochs) without improvement.")  # stop after this many epochs with no improvement
    parser.add_argument("--batch-size", type=int, default=32)  # how many examples are processed together at once
    parser.add_argument("--lr", type=float, default=5e-5)  # learning rate — a typical value for fine-tuning transformers
    parser.add_argument("--max-length", type=int, default=96)  # texts get cut off (or padded) to this many tokens
    parser.add_argument("--label-smoothing", type=float, default=0.0)  # off by default; see ConfigurableLossTrainer above
    parser.add_argument("--class-weighted", action=argparse.BooleanOptionalAction, default=False, help="Weight the loss inversely to class frequency, to help the minority (polarized) class.")  # off by default
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=DEFAULT_ADAPTERS,  # which adapters to fuse, if not overridden
        help="Adapters to fuse, as name=source pairs (space-separated). Source can be an "
        "AdapterHub/HF Hub id or a local path to a saved adapter, e.g. "
        "valence=output_valence/valence emotion_ec=output_emotion_ec/emotion_ec",
    )
    parser.add_argument("--seed", type=int, default=42)  # for reproducible shuffling/splitting
    parser.add_argument(
        "--fp16",  # "fp16" = do the math in 16-bit numbers instead of the usual 32-bit — faster on a GPU, same result for practical purposes
        action=argparse.BooleanOptionalAction,
        default=None,  # None = decide automatically based on whether a GPU is available
        help="Use fp16 mixed precision training. Defaults to on when a GPU is available.",
    )
    return parser


def train_and_evaluate(args):
    use_fp16 = args.fp16 if args.fp16 is not None else torch.cuda.is_available()
    if args.fp16 and not torch.cuda.is_available():
        print("Warning: --fp16 requested but no GPU is available; running in full precision.")
        use_fp16 = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_split, val_split = load_train_val_split(args.train_file, args.val_fraction, seed=args.seed)
    if args.train_fraction < 1.0:
        # Only reached if you deliberately asked for less than 100% of the
        # training data (e.g. sweep.py does this to keep its search cheap).
        keep = int(len(train_split) * args.train_fraction)
        train_split = train_split.shuffle(seed=args.seed).select(range(keep))

    class_weights = None
    if args.class_weighted:
        labels_np = np.array(train_split["polarization"])
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels_np)
        class_weights = torch.tensor(weights, dtype=torch.float)

    train_dataset = tokenize_dataset(train_split, tokenizer, args.max_length)
    val_dataset = tokenize_dataset(val_split, tokenizer, args.max_length)

    adapter_specs = parse_adapter_specs(args.adapters)
    model, fusion_setup = build_model(adapter_specs)

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

    trainer = ConfigurableLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    metrics = trainer.evaluate()

    if args.save_artifacts:
        model.save_adapter_fusion(f"{args.output_dir}/fusion", fusion_setup)
        model.save_head(f"{args.output_dir}/head", HEAD_NAME)
        # Records exactly which adapters (and where from) were fused, so
        # calibrate.py / evaluate_test.py know how to reload this same model
        # later without guessing.
        with open(f"{args.output_dir}/adapters.json", "w") as f:
            json.dump([{"name": name, "source": source} for name, source in adapter_specs], f, indent=2)

    return metrics


def main():
    args = build_arg_parser().parse_args()
    metrics = train_and_evaluate(args)
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()
