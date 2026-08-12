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
python train_fusion.py --train-file eng_train.csv
```

`train_fusion.py` holds out a validation split from `--train-file` itself
(`--val-fraction`, default 0.2, stratified so the label balance matches) and
uses it for early stopping / picking the best epoch. It does **not** touch
`eng_test.csv` — that file is reserved for a single, final evaluation once
all tuning is done (see [Tuning pipeline](#tuning-pipeline) below). It
prints `Validation metrics: {...}` after training, and saves the trained
fusion layer + head under `<output-dir>/fusion` and `<output-dir>/head`.

Useful flags: `--epochs` (an upper bound; early stopping usually stops
sooner), `--patience`, `--batch-size`, `--lr`, `--max-length`,
`--label-smoothing`, `--class-weighted`/`--no-class-weighted`,
`--val-fraction`, `--output-dir`.

Note: this requires downloading `bert-base-uncased` and the two adapters
from the Hugging Face Hub on first run.

## Tuning pipeline

Four scripts (`sweep.py`, `train_fusion.py`, `calibrate.py`,
`evaluate_test.py`) implement a small, deliberately staged tuning process,
aimed at three goals at once: raising F1, reducing how confidently the
model gets things wrong, and doing both without burning much compute. Each
step below explains what it varies and why that specific thing was chosen.

### 1. Validation split, not the test set, for tuning

`train_fusion.py` (and `sweep.py`, which calls it) hold out a stratified
validation split from `--train-file` for early stopping and comparing
settings. `eng_test.csv` is deliberately never touched until step 7. The
reason: if different settings are repeatedly judged against the test set,
whichever setting happens to look best on it gets chosen *because* it fits
that particular 1,452 rows — the final reported number would then be
optimistic, not a fair estimate of real performance on new data.

### 2. Learning rate sweep

```bash
python sweep.py --train-file eng_train.csv
```

The fusion layer and head are the only trainable parameters (everything
else is frozen), and for a module that small, learning rate is usually the
single highest-leverage setting — too low and it barely learns before
early stopping kicks in, too high and it overshoots. `sweep.py`'s first
stage tries `--lr-candidates` (default `1e-5,5e-5,1e-4,5e-4`) and keeps
whichever gives the best validation F1.

### 3. Class-weighted loss

Second stage: with the winning learning rate fixed, tries the loss with
and without inverse-frequency class weighting. `polarization` is
imbalanced (~37% positive in the training data), which can bias a model
toward the majority class and hurt recall on the class that actually
matters. Weighting is only kept if it measurably helps this specific
model/data combination — it isn't assumed to help by default, since it can
also trade away precision.

### 4. Label smoothing

Third stage: with the winning LR and class-weighting decision fixed, tries
`--label-smoothing-candidates` (default `0.0,0.1`). Label smoothing
discourages the model from pushing predicted probabilities to extremes
during training, which is a training-time lever for the same "reduce
confidently-wrong errors" goal that calibration (step 6) addresses
post-hoc — trying both is cheap, and whichever value wins on validation F1
is kept.

Each of these three stages runs on a subsample of the training data
(`--train-fraction`, default 0.5) with early stopping, specifically to
keep the search itself cheap — it's meant to rule settings in or out
quickly, not to produce the final model. Results print as a table at each
stage, and the winning combination is written to `best_config.json`.

### 5. Final full-data training run

Using the config `sweep.py` wrote out, retrain once on the *full* training
data (no `--train-fraction` subsampling this time) — this is the actual
model you keep:

```bash
python train_fusion.py --train-file eng_train.csv \
  --lr <best_lr> --class-weighted-or-not --label-smoothing <best_value>
```

(`sweep.py` prints the exact command with your winning values filled in.)

### 6. Calibration: temperature scaling + threshold tuning

```bash
python calibrate.py --output-dir output --train-file eng_train.csv
```

This is the dedicated fix for "confidence of errors," and it's
post-training — no retraining, no extra GPU time:

- **Temperature scaling** fits a single scalar that rescales the model's
  output probabilities so they better reflect how often it's actually
  right, using the same validation split as training. A model that's
  consistently overconfident gets a temperature > 1, which pulls its
  probabilities back toward less extreme values without changing which
  class it predicts or touching accuracy/F1.
- **Threshold tuning** sweeps the decision cutoff (instead of assuming the
  default 0.5) to find the value that maximizes F1 on the calibrated
  validation probabilities.

Both values are saved to `<output-dir>/calibration.json`.

### 7. Final test evaluation — run once

```bash
python evaluate_test.py --output-dir output --test-file eng_test.csv
```

Loads the final model plus `calibration.json` and evaluates on
`eng_test.csv` for the first and only time, applying the calibrated
probabilities and tuned threshold. This is the number to report. Going
back to steps 2-6 after looking at this output and re-running it defeats
the reason the test set was held out in the first place — if changes are
needed, they belong before this step, using the validation split, not
after it.

It also writes a `summary.md` to `<output-dir>` (so if `--output-dir`
points into a mounted Google Drive folder, the summary is saved there
automatically) with the winning `best_config.json` settings, the
calibration values, and the final metrics — a single record of the whole
run. Re-running `evaluate_test.py` just regenerates this file from the
same already-trained model and already-fit calibration; it doesn't re-tune
anything, so it's safe to run again purely to (re)produce the summary.

### Running the whole pipeline

```bash
python sweep.py --train-file eng_train.csv
# copy the printed command, e.g.:
python train_fusion.py --train-file eng_train.csv --lr 5e-5 --class-weighted --label-smoothing 0.1
python calibrate.py --output-dir output --train-file eng_train.csv
python evaluate_test.py --output-dir output --test-file eng_test.csv
```

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

## Is the adapter fusion actually helping?

`train_baseline.py` trains a fresh classification head directly on frozen
`bert-base-uncased` — no adapters, no fusion — with everything else (data,
hyperparameters, frozen-base setup) identical to `train_fusion.py`. Run it
the same way:

```bash
python train_baseline.py --train-file eng_train.csv --test-file eng_test.csv
```

Compare its printed `Baseline (no adapters) test set metrics: {...}` line
against the fusion model's numbers from `evaluate_test.py` (same metrics:
accuracy/precision/recall/F1, on the same test set — `train_baseline.py`
isn't part of the validation/calibration pipeline above, it evaluates
directly against `eng_test.csv` since it's a single one-off comparison
rather than something being iteratively tuned). If the fusion run scores
meaningfully higher, the sst2/emotion adapters are adding real value; if
the two are close, the adapters aren't contributing much beyond what
frozen BERT's own representation already captures.
