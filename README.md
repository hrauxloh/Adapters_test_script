# Adapters_test_script

Fuses two pre-trained [AdapterHub](https://adapterhub.ml/) task adapters on
top of `bert-base-uncased` to predict a binary `polarization` label:

- [`AdapterHub/bert-base-uncased-pf-sst2`](https://huggingface.co/AdapterHub/bert-base-uncased-pf-sst2) (sentiment)
- [`AdapterHub/bert-base-uncased-pf-emotion`](https://huggingface.co/AdapterHub/bert-base-uncased-pf-emotion) (emotion)

Both adapters are loaded frozen, combined with an
[AdapterFusion](https://docs.adapterhub.ml/adapter_fusion.html) layer, and a
new binary classification head is trained on top for the `polarization`
task. Only the fusion layer and the new head are trained — the base model
and the two source adapters stay frozen.

## Data

`eng_train.csv` / `eng_test.csv` each contain a `text` column and a binary
`polarization` column (0/1), plus additional sub-category columns that are
not used by this script.

## Setup

```bash
pip install -r requirements.txt
```

## Train + evaluate

```bash
python train_fusion.py --train-file eng_train.csv --test-file eng_test.csv
```

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--max-length`,
`--output-dir`. Test-set accuracy/precision/recall/F1 are printed after
training, and the trained fusion layer + head are saved under
`<output-dir>/fusion` and `<output-dir>/head`.

Note: this requires downloading `bert-base-uncased` and the two adapters
from the Hugging Face Hub on first run.

## Inspect which adapter drove each decision

After training (so `<output-dir>/fusion` and `<output-dir>/head` exist), run:

```bash
python inspect_fusion.py --output-dir output --data-file eng_test.csv
```

This reloads the trained fusion + head, runs it over test examples with
`output_adapter_fusion_attentions=True`, and prints how much weight the
fusion layer assigned to the `sst2` adapter vs. the `emotion` adapter —
both averaged per transformer layer, and per individual example. Weights
are a softmax over the two adapters, so they always sum to ~1 per token.

It also reports the model's own prediction confidence (softmax
probability of its predicted class), split into average confidence on
correct vs. incorrect predictions, and lists the misclassified examples
the model was *most* confident about — useful for spotting cases where
it's confidently wrong rather than just uncertain on hard examples.

Useful flags: `--num-examples`, `--show-examples`, `--show-misclassified`,
`--batch-size`.
