"""Loading and tokenization helpers for the polarization train/test CSVs."""

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

TEXT_COLUMN = "text"
LABEL_COLUMN = "polarization"


def _read_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df.reset_index(drop=True)


def load_split(csv_path: str) -> Dataset:
    """Read a polarization CSV and keep only the text/label columns."""
    return Dataset.from_pandas(_read_csv(csv_path))


def load_train_val_split(csv_path: str, val_fraction: float = 0.2, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Read a training CSV and split off a stratified validation set.

    Used to keep hyperparameter selection (learning rate, class weighting,
    label smoothing, early stopping) separate from the held-out test set, so
    the final test-set number isn't tuned to.
    """
    df = _read_csv(csv_path)
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, random_state=seed, stratify=df[LABEL_COLUMN]
    )
    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
    )


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 256) -> Dataset:
    """Tokenize the text column and format the dataset for a torch Trainer."""

    def _tokenize(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(_tokenize, batched=True, remove_columns=[TEXT_COLUMN])
    dataset = dataset.rename_column(LABEL_COLUMN, "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset
