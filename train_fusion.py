"""Fuse two pre-trained AdapterHub task adapters (SST-2 sentiment, emotion)
on top of bert-base-uncased and train the fusion + a new classification head
to predict the binary `polarization` label.

Usage:
    python train_fusion.py --train-file eng_train.csv --test-file eng_test.csv
"""

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, TrainingArguments
from adapters import AutoAdapterModel, AdapterTrainer, Fuse

from data import load_split, tokenize_dataset

MODEL_NAME = "bert-base-uncased"
SST2_ADAPTER = "AdapterHub/bert-base-uncased-pf-sst2"
EMOTION_ADAPTER = "AdapterHub/bert-base-uncased-pf-emotion"
SST2_NAME = "sst2"
EMOTION_NAME = "emotion"
HEAD_NAME = "polarization"
FUSION_SETUP = Fuse(SST2_NAME, EMOTION_NAME)


def build_model():
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)

    # Load the two pre-trained task adapters, frozen, without their original heads.
    model.load_adapter(SST2_ADAPTER, load_as=SST2_NAME, with_head=False)
    model.load_adapter(EMOTION_ADAPTER, load_as=EMOTION_NAME, with_head=False)

    # Add an AdapterFusion layer combining both, plus a fresh binary head for polarization.
    model.add_adapter_fusion(FUSION_SETUP)
    model.add_classification_head(HEAD_NAME, num_labels=2)

    # Freeze the base model and the two source adapters; only the fusion layer
    # and the new head remain trainable.
    model.train_adapter_fusion(FUSION_SETUP)
    model.set_active_adapters(FUSION_SETUP)
    model.active_head = HEAD_NAME

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
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

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
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("Test set metrics:", metrics)

    model.save_adapter_fusion(f"{args.output_dir}/fusion", FUSION_SETUP)
    model.save_head(f"{args.output_dir}/head", HEAD_NAME)


if __name__ == "__main__":
    main()
