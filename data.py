"""
WHAT THIS FILE IS FOR
----------------------
This is not a script you run by itself — it's a shared toolbox of small
functions that every training script in this project reuses, so the same
"read a CSV and get it ready for the model" logic only has to exist once.

Menu of what's in this file, in the order it's used:
  1. read_csv (private helper) - reads a CSV file and keeps only the text
     and label columns you ask for.
  2. load_split - reads a CSV where every row already has one label
     (e.g. the polarization data).
  3. load_train_val_split - same as above, but also carves off a slice of
     the rows to use for checking progress during training (this is how
     the model avoids "cheating" by peeking at the final test data).
  4. tokenize_dataset - turns human-readable text into the numbers a BERT
     model actually reads.
  5. load_multilabel_split / tokenize_multilabel_dataset - the same two
     jobs as above, but for data where several label columns apply to one
     row at once (used for the multi-label emotion data).
"""

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

TEXT_COLUMN = "text"
LABEL_COLUMN = "polarization"  # default label column, for the polarization CSVs


def _read_csv(csv_path: str, label_column: str = LABEL_COLUMN) -> pd.DataFrame:
    # Reads any CSV that has a "text" column plus one label column, and drops
    # any row that's missing either. Reused by every single-label dataset in
    # this project (polarization, group-identity, valence) — not specific to
    # any one of them, which is why label_column is a setting, not fixed.
    df = pd.read_csv(csv_path)
    df = df[[TEXT_COLUMN, label_column]].dropna()
    df[label_column] = df[label_column].astype(int)
    return df.reset_index(drop=True)


def load_split(csv_path: str, label_column: str = LABEL_COLUMN) -> Dataset:
    """Read a CSV and keep only the text/label columns."""
    return Dataset.from_pandas(_read_csv(csv_path, label_column))


def load_train_val_split(
    csv_path: str, val_fraction: float = 0.2, seed: int = 42, label_column: str = LABEL_COLUMN
) -> tuple[Dataset, Dataset]:
    """Read a training CSV and split off a stratified validation set.

    Used to keep hyperparameter selection (learning rate, class weighting,
    label smoothing, early stopping) separate from the held-out test set, so
    the final test-set number isn't tuned to.
    """
    # "Stratified" means the split keeps the same balance of labels in both
    # halves — e.g. if 36% of rows are polarized, both the training slice and
    # the validation slice will still be about 36% polarized, not skewed.
    df = _read_csv(csv_path, label_column)
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, random_state=seed, stratify=df[label_column]
    )
    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
    )


def tokenize_dataset(
    dataset: Dataset, tokenizer, max_length: int = 256, label_column: str = LABEL_COLUMN
) -> Dataset:
    """Tokenize the text column and format the dataset for a torch Trainer."""

    def _tokenize(batch):
        # "Tokenize" = chop the text into sub-word pieces BERT knows, then
        # convert each piece to a number (its ID in BERT's vocabulary).
        # padding="max_length" pads every example to the same length so they
        # can be grouped into batches; truncation cuts off anything longer.
        return tokenizer(
            batch[TEXT_COLUMN],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(_tokenize, batched=True, remove_columns=[TEXT_COLUMN])
    # Every model/Trainer in this project expects the label column to be
    # called "labels" specifically, so rename it here rather than in every
    # training script separately.
    dataset = dataset.rename_column(label_column, "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def load_multilabel_split(csv_path: str, label_columns: list) -> Dataset:
    """Read a CSV with several binary label columns (e.g. multiple emotions)
    and combine them into a single multi-hot "labels" column.
    """
    df = pd.read_csv(csv_path)
    df = df[[TEXT_COLUMN] + label_columns].dropna()  # keep only the columns we need
    df[label_columns] = df[label_columns].astype("float32")  # 0/1 as decimals, for the loss function later
    # Combine e.g. anger=0, joy=1, fear=0, ... into one list per row: [0, 1, 0, ...]
    df["labels"] = df[label_columns].values.tolist()
    df = df[[TEXT_COLUMN, "labels"]].reset_index(drop=True)
    return Dataset.from_pandas(df)


def tokenize_multilabel_dataset(dataset: Dataset, tokenizer, max_length: int = 256) -> Dataset:
    """Tokenize the text column of a multi-label dataset built by load_multilabel_split."""

    def _tokenize(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(_tokenize, batched=True, remove_columns=[TEXT_COLUMN])
    # No rename needed here — load_multilabel_split already called the
    # combined column "labels".
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset
