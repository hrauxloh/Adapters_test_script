"""Inspect a trained AdapterFusion model: for each transformer layer, how much
weight the fusion layer assigned to the sst2 adapter vs. the emotion adapter
when making its polarization predictions, and how confident the model was on
correct vs. incorrect predictions.

Usage:
    python inspect_fusion.py --output-dir output --data-file eng_test.csv
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel, Fuse

from data import load_split

MODEL_NAME = "bert-base-uncased"
SST2_ADAPTER = "AdapterHub/bert-base-uncased-pf-sst2"
EMOTION_ADAPTER = "AdapterHub/bert-base-uncased-pf-emotion"
SST2_NAME = "sst2"
EMOTION_NAME = "emotion"
HEAD_NAME = "polarization"
FUSION_SETUP = Fuse(SST2_NAME, EMOTION_NAME)
FUSION_KEY = ",".join(FUSION_SETUP)
# Fallback for models trained before adapters.json existed (e.g. the original
# sst2+emotion model) — such an output-dir has no adapters.json, so we assume
# the original pair.
DEFAULT_ADAPTER_SPECS = [(SST2_NAME, SST2_ADAPTER), (EMOTION_NAME, EMOTION_ADAPTER)]


def load_trained_model(output_dir):
    adapters_meta_path = f"{output_dir}/adapters.json"
    if os.path.exists(adapters_meta_path):
        with open(adapters_meta_path) as f:
            adapter_specs = [(a["name"], a["source"]) for a in json.load(f)]
    else:
        adapter_specs = DEFAULT_ADAPTER_SPECS

    model = AutoAdapterModel.from_pretrained(MODEL_NAME)
    for name, source in adapter_specs:
        model.load_adapter(source, load_as=name, with_head=False)
    model.load_adapter_fusion(f"{output_dir}/fusion", set_active=True)
    model.load_head(f"{output_dir}/head")
    model.active_head = HEAD_NAME
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--data-file", default="eng_test.csv")
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--show-examples", type=int, default=15)
    parser.add_argument("--show-misclassified", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_trained_model(args.output_dir)

    dataset = load_split(args.data_file)
    n = min(args.num_examples, len(dataset))
    dataset = dataset.select(range(n))
    texts = dataset["text"]
    labels = dataset["polarization"]

    layer_sums = {}
    layer_token_counts = {}
    per_example_weights = []
    preds = []
    confidences = []

    for start in range(0, n, args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encodings, output_adapter_fusion_attentions=True)

        batch_probs = torch.softmax(outputs.logits, dim=-1).numpy()  # [batch, num_labels]
        batch_preds = batch_probs.argmax(axis=-1)
        preds.extend(batch_preds.tolist())
        confidences.extend(batch_probs[np.arange(len(batch_preds)), batch_preds].tolist())

        mask = encodings["attention_mask"].numpy().astype(bool)  # [batch, seq]
        fusion_attn = outputs.adapter_fusion_attentions[FUSION_KEY]

        layer_ids = sorted(fusion_attn.keys())
        stacked = np.stack(
            [fusion_attn[layer_id]["output_adapter"] for layer_id in layer_ids]
        )  # [num_layers, batch, seq, num_adapters]

        for layer_id, weights in zip(layer_ids, stacked):
            masked = weights[mask]  # [num_real_tokens_in_batch, num_adapters]
            layer_sums[layer_id] = layer_sums.get(layer_id, 0.0) + masked.sum(axis=0)
            layer_token_counts[layer_id] = layer_token_counts.get(layer_id, 0) + masked.shape[0]

        # Per-example average across all layers and its own real tokens.
        for i in range(len(batch_texts)):
            example_mask = mask[i]
            example_weights = stacked[:, i, example_mask, :].mean(axis=(0, 1))
            per_example_weights.append(example_weights)

    print(f"Averaged fusion weight per layer across {n} examples from {args.data_file}:")
    print(f"{'layer':>6} {SST2_NAME:>10} {EMOTION_NAME:>10}")
    for layer_id in sorted(layer_sums.keys()):
        avg = layer_sums[layer_id] / layer_token_counts[layer_id]
        print(f"{layer_id:>6} {avg[0]:>10.3f} {avg[1]:>10.3f}")

    print()
    print(f"Per-example fusion weight, averaged across all layers (first {args.show_examples} shown):")
    print(f"{'label':>6} {'pred':>6} {'conf':>6} {SST2_NAME:>10} {EMOTION_NAME:>10}  text")
    for i in range(min(args.show_examples, n)):
        w = per_example_weights[i]
        text_preview = texts[i][:60].replace("\n", " ")
        print(f"{labels[i]:>6} {preds[i]:>6} {confidences[i]:>6.3f} {w[0]:>10.3f} {w[1]:>10.3f}  {text_preview}")

    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    confidences_arr = np.array(confidences)
    correct = labels_arr == preds_arr

    print()
    print(f"Prediction confidence (the model's own probability for whichever class it picked):")
    print(f"  overall accuracy on these {n} examples: {correct.mean():.3f}")
    if correct.any():
        print(f"  avg confidence when CORRECT:   {confidences_arr[correct].mean():.3f}")
    if (~correct).any():
        print(f"  avg confidence when INCORRECT: {confidences_arr[~correct].mean():.3f}")
    print(
        "  (if these two numbers are close, the model is often 'confidently wrong' rather than\n"
        "   just uncertain on hard cases)"
    )

    print()
    misclassified = np.where(~correct)[0]
    order = misclassified[np.argsort(-confidences_arr[misclassified])]
    show_n = min(args.show_misclassified, len(order))
    print(f"Most confidently WRONG examples ({show_n} of {len(misclassified)} misclassified shown):")
    print(f"{'label':>6} {'pred':>6} {'conf':>6} {SST2_NAME:>10} {EMOTION_NAME:>10}  text")
    for i in order[:show_n]:
        w = per_example_weights[i]
        text_preview = texts[i][:60].replace("\n", " ")
        print(f"{labels[i]:>6} {preds[i]:>6} {confidences[i]:>6.3f} {w[0]:>10.3f} {w[1]:>10.3f}  {text_preview}")


if __name__ == "__main__":
    main()
