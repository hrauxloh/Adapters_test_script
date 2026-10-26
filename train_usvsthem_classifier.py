"""Full RoBERTa fine-tuning for the "us vs. them" populist-attitude task,
following the approach in Huguet Cabot et al. 2021 ("Us vs. Them: A Dataset
of Populist Attitudes, News Bias and Emotions", EACL 2021) -- modernized:
their original code (github.com/LittlePea13/UsVsThem) pins PyTorch Lightning
0.8.3, transformers 3.0.0, plus apex and horovod, has at least one broken
import in testing_reg.py, and needs undocumented manual fixes to run at all.
This rebuilds the same idea -- full RoBERTa fine-tuning, trained jointly on
the us-vs-them score and an auxiliary emotion-prediction task -- against the
current, maintained Hugging Face Trainer instead, with no apex/horovod/
Lightning dependency.

Unlike every other training script in this repo (train_group_adapter.py,
train_valence_adapter.py, etc.), this does NOT attach a frozen-base adapter
-- it fine-tunes all of roberta-base's weights directly, matching the
paper's own setup. It's a standalone experiment and isn't wired into this
repo's adapter-fusion pipeline.

Two heads sit on top of RoBERTa's [CLS] representation:
  - a regression head predicting usVSthem_scale (0..1)
  - a multi-label head predicting the 13 emotion flags (Anger, Contempt,
    Disgust, Fear, Gratitude, Guilt, Happiness, Hope, Pride, Relief,
    Sadness, Sympathy, Emotions_Neutral)
Total loss = regression MSE + --loss-aux-weight * emotion BCE. Setting
--loss-aux-weight 0 trains on the us-vs-them score alone (single-task
ablation, useful as a comparison point).

Uses the *_public.csv files (UsVsThem_{train,valid,test}_public.csv), which
carry the emotion columns the plain UsVsThem_{train,valid,test}.csv files
(used elsewhere in this repo) don't.

This file exposes build_arg_parser()/train_and_evaluate() the same way
train_fusion.py does, so grid_search_usvsthem.py can import and call it
directly for each grid point instead of shelling out.

Usage:
    python train_usvsthem_classifier.py --train-file UsVsThem_train_public.csv --val-file UsVsThem_valid_public.csv
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

MODEL_NAME = "roberta-base"
TEXT_COLUMN = "body"
SCORE_COLUMN = "usVSthem_scale"
EMOTION_COLUMNS = [
    "Anger", "Contempt", "Disgust", "Fear", "Gratitude", "Guilt", "Happiness",
    "Hope", "Pride", "Relief", "Sadness", "Sympathy", "Emotions_Neutral",
]


class UsVsThemDataset(torch.utils.data.Dataset):
    """Tokenizes once up front; __getitem__ just slices pre-computed tensors."""

    def __init__(self, df, tokenizer, max_length):
        texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        self.scores = df[SCORE_COLUMN].astype(float).to_numpy()
        self.emotions = df[EMOTION_COLUMNS].astype(float).to_numpy()

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        item["score"] = torch.tensor(self.scores[idx], dtype=torch.float)
        item["emotions"] = torch.tensor(self.emotions[idx], dtype=torch.float)
        return item


class MultiTaskRoberta(nn.Module):
    """RoBERTa (all weights train by default, or the top --freeze-layers-
    from-the-bottom left frozen) with two small heads on the [CLS] token:
    one number (the us-vs-them score) and 13 numbers (one per emotion,
    each treated as an independent yes/no).

    Note: this reads last_hidden_state[:, 0] directly, NOT
    outputs.pooler_output -- the pooler's randomly-initialized weights
    (see the "MISSING" load-report lines at startup) are loaded but never
    actually used here. The freshly-random parameters each run are
    score_head/emotion_head below, not the pooler."""

    def __init__(self, model_name, num_emotions, dropout=0.1, freeze_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, 1)
        self.emotion_head = nn.Linear(hidden_size, num_emotions)
        if freeze_layers > 0:
            freeze_lower_layers(self.encoder, freeze_layers)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0])
        return {
            "score": self.score_head(cls_repr).squeeze(-1),
            "emotions": self.emotion_head(cls_repr),
        }


def freeze_lower_layers(encoder, num_layers):
    """Freezes the embeddings and the bottom `num_layers` transformer
    layers (out of roberta-base's 12), leaving only the top layers + the
    two heads to actually train. Standard small-data fine-tuning move:
    fewer trainable parameters generally means less run-to-run variance
    from the heads' random initialization and less overfitting risk on a
    ~3.7k-row training set."""
    for param in encoder.embeddings.parameters():
        param.requires_grad = False
    total_layers = len(encoder.encoder.layer)
    if num_layers > total_layers:
        raise ValueError(f"--freeze-layers {num_layers} exceeds the model's {total_layers} layers.")
    for layer in encoder.encoder.layer[:num_layers]:
        for param in layer.parameters():
            param.requires_grad = False


class MultiTaskTrainer(Trainer):
    """Combines the regression loss (primary task) with the auxiliary
    emotion loss, weighted by --loss-aux-weight -- the modernized
    equivalent of the paper's --loss_aux flag."""

    def __init__(self, *args, loss_aux_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_aux_weight = loss_aux_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Reads score/emotions without mutating the shared `inputs` dict --
        # Trainer's own evaluation loop reads these same keys out of `inputs`
        # again *after* this call to build the labels it hands
        # compute_metrics, so popping them here would leave that step with
        # nothing (silently breaking eval metrics rather than erroring).
        score_labels = inputs["score"]
        emotion_labels = inputs["emotions"]
        # Allowlist rather than excluding ("everything but score/emotions")
        # -- newer transformers versions can inject extra batch metadata
        # (e.g. num_items_in_batch) into `inputs` for loss averaging, and
        # MultiTaskRoberta.forward has no **kwargs to absorb a surprise key.
        model_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        outputs = model(**model_inputs)

        regression_loss = nn.functional.mse_loss(outputs["score"], score_labels)
        loss = regression_loss
        if self.loss_aux_weight > 0:
            emotion_loss = nn.functional.binary_cross_entropy_with_logits(outputs["emotions"], emotion_labels)
            loss = regression_loss + self.loss_aux_weight * emotion_loss

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    # Read via the named fields, not positional unpacking of eval_pred
    # itself -- EvalPrediction has gained a 3rd field (inputs) across
    # transformers versions, so `a, b = eval_pred` is a version trap.
    # Trainer concatenates whatever the model returns -- score is (N,),
    # emotions is (N, 13), so predictions/label_ids are each a tuple of the two.
    score_preds, _emotion_preds = eval_pred.predictions
    score_labels, _emotion_labels = eval_pred.label_ids
    mae = float(np.mean(np.abs(score_preds - score_labels)))
    correlation = float(np.corrcoef(score_preds, score_labels)[0, 1])
    return {"mae": mae, "correlation": correlation}


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="UsVsThem_train_public.csv")
    parser.add_argument("--val-file", default="UsVsThem_valid_public.csv")
    parser.add_argument("--test-file", default="UsVsThem_test_public.csv", help="Only read if --evaluate-test is passed.")
    parser.add_argument("--evaluate-test", action="store_true", help="Also evaluate on --test-file. Only use this for the final, single run on the winning config -- never during a grid search.")
    parser.add_argument("--output-dir", default="output_usvsthem")
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=6, help="Upper bound; early stopping usually stops sooner.")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16, help="Smaller than the adapter scripts' default (32) -- full fine-tuning needs more memory per example.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Full fine-tuning needs a smaller LR than adapter training -- 1e-5 to 3e-5 is the typical RoBERTa range, vs. 5e-5 for the frozen-base adapters elsewhere in this repo.")
    parser.add_argument("--loss-aux-weight", type=float, default=0.0, help="Weight of the auxiliary emotion loss added to the primary regression loss. 0 = single-task (ablation).")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-layers", type=int, default=0, help="Freeze the embeddings + bottom N of roberta-base's 12 transformer layers (0 = fully trainable, matching the paper's own full fine-tuning setup).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (L2 regularization). 0 = off (matches earlier runs); ~0.01 is a common transformer fine-tuning default.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Fraction of total training steps spent linearly warming up the learning rate from 0. 0 = off (matches earlier runs); ~0.06 is a common default and can smooth out early-training grad-norm spikes.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use fp16 mixed precision training. Defaults to on when a GPU is available.",
    )
    return parser


def train_and_evaluate(args):
    use_fp16 = args.fp16 if args.fp16 is not None else torch.cuda.is_available()
    if args.fp16 and not torch.cuda.is_available():
        print("Warning: --fp16 requested but no GPU is available; running in full precision.")
        use_fp16 = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)
    train_dataset = UsVsThemDataset(train_df, tokenizer, args.max_length)
    val_dataset = UsVsThemDataset(val_df, tokenizer, args.max_length)

    model = MultiTaskRoberta(
        MODEL_NAME, num_emotions=len(EMOTION_COLUMNS), dropout=args.dropout, freeze_layers=args.freeze_layers
    )
    if args.freeze_layers > 0:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Froze embeddings + bottom {args.freeze_layers} layer(s): {n_trainable:,}/{n_total:,} parameters still trainable.")

    # The installed transformers version dropped TrainingArguments'
    # warmup_ratio convenience field (only the absolute warmup_steps
    # remains), so --warmup-ratio is converted to a step count by hand,
    # using the same steps-per-epoch math Trainer itself uses internally.
    steps_per_epoch = -(-len(train_dataset) // args.batch_size)  # ceiling division
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = round(args.warmup_ratio * total_steps)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        # Checkpointing + load_best_model_at_end stay on even for throwaway
        # grid-search trials (save_artifacts=False) -- they're what
        # EarlyStoppingCallback needs to function at all, and skipping them
        # would silently turn off early stopping for every trial, burning
        # through the full --epochs every time instead of cutting bad
        # combinations short. save_artifacts only gates the final permanent
        # copy (trainer.save_model below), not this internal checkpointing.
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
        # Our batch dict uses "score"/"emotions" instead of the usual
        # "labels" key -- without this, Trainer's default label_names=["labels"]
        # wouldn't find them, so it would skip compute_loss during evaluation
        # and instead call the model directly with score/emotions as stray
        # kwargs, which MultiTaskRoberta.forward doesn't accept.
        label_names=["score", "emotions"],
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        loss_aux_weight=args.loss_aux_weight,
    )

    trainer.train()
    val_metrics = trainer.evaluate()

    naive_baseline_mae = float(np.mean(np.abs(val_df[SCORE_COLUMN].to_numpy() - train_df[SCORE_COLUMN].mean())))
    print("Validation metrics:", val_metrics)
    print(f"(naive baseline MAE, always predicting the train-set mean: {naive_baseline_mae:.3f} -- lower is better)")
    if val_metrics["eval_mae"] >= naive_baseline_mae:
        print("WARNING: the model did not beat the naive mean-prediction baseline.")

    result = {"val": val_metrics}

    if args.evaluate_test:
        test_df = pd.read_csv(args.test_file)
        test_dataset = UsVsThemDataset(test_df, tokenizer, args.max_length)
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        print("Test metrics:", test_metrics)
        result["test"] = test_metrics

    if args.save_artifacts:
        trainer.save_model(f"{args.output_dir}/final")
        tokenizer.save_pretrained(f"{args.output_dir}/final")
        print(f"\nSaved model to {args.output_dir}/final")

    return result


def main():
    args = build_arg_parser().parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
