"""Loading and tokenization helpers for the polarization train/test CSVs."""

import pandas as pd
from datasets import Dataset

TEXT_COLUMN = "text"
LABEL_COLUMN = "polarization"


def load_split(csv_path: str) -> Dataset:
    """Read a polarization CSV and keep only the text/label columns."""
    df = pd.read_csv(csv_path)
    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return Dataset.from_pandas(df.reset_index(drop=True))


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
