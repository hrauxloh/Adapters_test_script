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

By default it fuses `sst2`+`emotion` from AdapterHub, but `--adapters` can
point it at any set of adapters, local or remote — see
[Custom adapters: valence + emotion (E-c)](#custom-adapters-valence--emotion-e-c)
below for a worked example that fuses two adapters trained from scratch in
this repo instead. Whichever adapters are used get recorded in
`<output-dir>/adapters.json`, so `calibrate.py`/`evaluate_test.py` know how
to reload the right ones later — models trained before this existed (i.e.
the original `sst2`+`emotion` output) have no such file and fall back to
that original pair automatically.

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

## This branch: alternate valence + group-identity training data

This branch (`claude/alt-valence-group-data`) swaps out the data behind two
of the three from-scratch adapters, keeping everything else in this repo —
the emotion adapter, the fusion/calibration/evaluation pipeline, the
ablation runner, the bootstrap confidence intervals, the plots — identical
to the main line of this project, since none of those treat an adapter as
anything more than a frozen black box:

- **Valence** now trains on `vreg_{train,dev,test}.csv` (`text`, `valence`
  — a genuinely continuous score from 0 to 1), instead of the old SemEval
  V-oc data (`voc_*.csv`, `Valence_Code`, an ordinal -3..+3 scale).
- **Group identity** now trains on `UsVsThem_{train,valid,test}.csv`
  (`text`, `usVSthem_scale` — also continuous, 0 to 1, how strongly a text
  expresses in-group/out-group framing), instead of the old
  `group_reference_data.csv` (`text`, `Group_bool`, a plain yes/no label).

The group adapter's target used to be binary, so it trained as ordinary
2-class classification. Now that it's continuous like valence, both
adapters use the same regression approach — see below — matching this
project's standing rule of keeping the training approach as similar as
possible across adapters, differing only where the data genuinely forces
it. `data.py`'s CSV loader gained a `label_dtype` setting for this reason:
the default (`int`) is fine for yes/no labels, but would silently truncate
a continuous score like `0.6` down to `0` — these two scripts pass
`label_dtype=float` to keep the real value.

## Custom adapters: valence + emotion (E-c)

This branch trains two adapters from scratch — valence, and emotion
(`Ec_{train,val,test}.csv`: `text` plus 11 binary emotion columns) — and
fuses them into a **separate** polarization model, entirely parallel to the
original `sst2`+`emotion` one (different output directory, nothing about
the original model's saved files is touched), so the two can be compared
later.

**Valence, as regression, not classification.** Valence here is a genuinely
continuous quantity (0 to 1), not a set of categories — mistaking 0.9 for
0.8 is a much smaller error than mistaking it for 0.1, which plain
classification would throw away. `train_valence_adapter.py` instead trains
a single continuous output with MSE loss, which naturally penalizes
predictions in proportion to how far off they are:

```bash
python train_valence_adapter.py
```

Reports MAE (mean absolute error, the primary metric — lower is better) and
a correlation coefficient (how well the model's ranking of examples matches
the true ranking) instead of precision/recall/F1, which don't fit a
regression target. Also prints a naive baseline (always predicting the
training-set mean) to compare against. Saves only the adapter to
`output_valence/valence`.

**Emotion, as multi-label, not 6-way single-label.** Unlike the pretrained
`emotion` adapter (single-label), `Ec_*.csv` allows any number of the 11
emotions to be present in the same text — `train_emotion_adapter.py` uses a
genuine multi-label head (`multilabel=True`: independent sigmoid output per
emotion, binary cross-entropy loss) rather than training 11 separate
adapters, so it can learn correlations between emotions rather than
treating them as unrelated:

```bash
python train_emotion_adapter.py
```

Reports micro/macro F1 across all 11 labels. Saves only the adapter to
`output_emotion_ec/emotion_ec`.

**Fusing them.** `train_fusion.py` now accepts `--adapters` as a list of
`name=source` pairs (source can be a local path or a Hub id), so the same
script builds this new model instead of a separate one:

```bash
python train_fusion.py --train-file eng_train.csv \
  --adapters valence=output_valence/valence emotion_ec=output_emotion_ec/emotion_ec \
  --output-dir output_custom
```

From here, `calibrate.py` and `evaluate_test.py` work exactly as before,
just pointed at `output_custom` — they read `output_custom/adapters.json`
to know to reload `valence`+`emotion_ec` rather than `sst2`+`emotion`:

```bash
python calibrate.py --output-dir output_custom --train-file eng_train.csv
python evaluate_test.py --output-dir output_custom --test-file eng_test.csv
```

Comparing `output_custom`'s final test metrics against `output`'s (the
original model) tells you whether your own trained valence/emotion
adapters do better, worse, or about the same as the pretrained AdapterHub
ones — same comparison logic as the baseline check above, just between two
fusion models instead of fusion-vs-none. Hyperparameters here are all
defaults (standard learning rate, batch size, etc.) — tuning this model the
way `sweep.py` tunes the original is future work, not done yet.

## Training a new adapter from scratch: group identity

`sst2` and `emotion` are pretrained adapters pulled from AdapterHub. There's
no equivalent pretrained adapter for in-group/out-group ("us vs. them")
framing, so `train_group_adapter.py` trains one from scratch, entirely
independent of the polarization task:

```bash
python train_group_adapter.py
```

`UsVsThem_{train,valid,test}.csv` (`text`, `usVSthem_scale`) already comes
pre-split, unrelated to `eng_train.csv`/`eng_test.csv` — so like
`train_valence_adapter.py`, this script doesn't carve its own validation
split, it just reads `--train-file`/`--val-file` directly, and never
touches the polarization test set. Since the label here is a continuous 0-1
score rather than yes/no, this trains the same way `train_valence_adapter.py`
does now: a single continuous output with MSE loss (see the section above
for why), reporting MAE and a correlation coefficient instead of
precision/recall/F1. The new adapter uses the same Pfeiffer/`SeqBn`
bottleneck architecture as `sst2`/`emotion`, so it's dimensionally
compatible for fusion. Only the adapter itself is saved (`<output-dir>/group`)
— its temporary regression head, used only to train it, is discarded, the
same way `sst2`/`emotion` are loaded with `with_head=False` elsewhere in
this repo.

The script prints a naive-mean-baseline comparison alongside its validation
metrics, and warns explicitly if the adapter doesn't beat it — worth
checking before using this adapter in any downstream fusion model.

This adapter is a building block, not a finished result on its own: testing
whether group identity actually matters for polarization detection (as
opposed to sentiment or emotion) requires the full ablation set described
in the next section, not just this one training run.

## Testing which features actually matter: the ablation plan

The claim "valence, emotion, and group identity are important features for
detecting polarization" is stronger than what fusing all three together
and beating a no-adapter baseline can show — that only proves the
combination helps, not that each one individually does, or that these
specific indicators matter rather than any diverse set of adapters.
Properly testing the individual claim needs a full factorial ablation:

| Adapters included | What it isolates | Status |
|---|---|---|
| none (baseline) | Reference point | `train_baseline.py` |
| valence only | Valence's individual contribution | `run_ablation_models.py` |
| emotion_ec only | Emotion's individual contribution | `run_ablation_models.py` |
| group only | Group identity's individual contribution | `run_ablation_models.py` |
| valence + emotion_ec | Pairwise combination | already trained (`output_custom`) |
| valence + group | Pairwise combination | `run_ablation_models.py` |
| emotion_ec + group | Pairwise combination | `run_ablation_models.py` |
| valence + emotion_ec + group | All three combined | already trained (`output_full`) |
| 3 unrelated-task adapters | Confound control — does *any* 3-adapter fusion help, or specifically these | not built yet |

`train_fusion.py`'s `--adapters` setting already supports any of these
combinations, including a single adapter — when given exactly one, it
skips the fusion layer entirely (an `AdapterFusion` layer has its own
extra trainable weights even with only one adapter inside it, so using it
for a "one adapter alone" row would give that row more capacity than it's
supposed to have) and just trains a fresh head directly on that one frozen
adapter, matching how `train_group_adapter.py` etc. train adapters
standalone.

**`run_ablation_models.py`** runs the five combinations above that aren't
already trained — for each one, it runs `train_fusion.py` → `calibrate.py`
→ `evaluate_test.py` back to back, with standard/default hyperparameters
(no tuning), saving each model's results in its own output directory:

```bash
python run_ablation_models.py --drive-root /content/drive/MyDrive
```

**`summarize_ablation_results.py`** then reads every model's `summary.md`
(the three already-trained ones plus the five just produced), prints a
combined results table, saves it as a CSV, and draws a bar chart of F1
across all eight models — color-coded by baseline/single/pair/full — saved
as both a PNG and a vector PDF:

```bash
python summarize_ablation_results.py --drive-root /content/drive/MyDrive
```

The confound control (fusing three adapters trained on unrelated tasks)
still needs sourcing suitable adapters first — it isn't part of either
script above.
