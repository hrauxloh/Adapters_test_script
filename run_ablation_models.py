"""
WHAT THIS SCRIPT DOES
----------------------
Runs the 5 ablation models that haven't been trained yet, completing the
factorial ablation table from the README (everything except the
unrelated-adapter confound control): each of the three adapters (group,
valence, emotion_ec) alone, and the two remaining pairs
(valence+group, emotion_ec+group).

For each of the 5, it runs the exact same three steps you've been running
by hand — train_fusion.py, calibrate.py, evaluate_test.py — as subprocesses,
one after another, with standard/default hyperparameters throughout (no
tuning). This is meant to be run once, straight through; each model's
result lands in its own output directory with a calibration.json and a
summary.md, exactly like your other runs.

Menu of what happens when you run this file, in order:
  1. Define the 5 missing model configurations (name, adapters, output dir).
  2. For each one: train it, calibrate it, and run the final test evaluation.
  3. Print a short recap of all 5 F1 scores at the end.

Usage:
    python run_ablation_models.py --drive-root /content/drive/MyDrive
"""

import argparse
import subprocess
import sys


def build_model_configs(drive_root):
    group_adapter = f"{drive_root}/output_group/group"
    valence_adapter = f"{drive_root}/output_valence/valence"
    emotion_adapter = f"{drive_root}/output_emotion_ec/emotion_ec"

    return [
        {
            "name": "group_only",
            "adapters": [f"group={group_adapter}"],
            "output_dir": f"{drive_root}/output_group_only",
        },
        {
            "name": "valence_only",
            "adapters": [f"valence={valence_adapter}"],
            "output_dir": f"{drive_root}/output_valence_only",
        },
        {
            "name": "emotion_only",
            "adapters": [f"emotion_ec={emotion_adapter}"],
            "output_dir": f"{drive_root}/output_emotion_only",
        },
        {
            "name": "valence_group",
            "adapters": [f"valence={valence_adapter}", f"group={group_adapter}"],
            "output_dir": f"{drive_root}/output_valence_group",
        },
        {
            "name": "emotion_group",
            "adapters": [f"emotion_ec={emotion_adapter}", f"group={group_adapter}"],
            "output_dir": f"{drive_root}/output_emotion_group",
        },
    ]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive",
        help="Folder containing your already-trained adapters (output_group, "
        "output_valence, output_emotion_ec) — the new fusion models are also "
        "saved here, in their own new subfolders.",
    )
    parser.add_argument("--train-file", default="eng_train.csv")
    parser.add_argument("--test-file", default="eng_test.csv")
    args = parser.parse_args()

    configs = build_model_configs(args.drive_root)
    python = sys.executable

    for config in configs:
        print(f"\n{'=' * 60}\nModel: {config['name']}\n{'=' * 60}")

        run([
            python, "train_fusion.py",
            "--train-file", args.train_file,
            "--adapters", *config["adapters"],
            "--output-dir", config["output_dir"],
        ])
        run([
            python, "calibrate.py",
            "--output-dir", config["output_dir"],
            "--train-file", args.train_file,
        ])
        run([
            python, "evaluate_test.py",
            "--output-dir", config["output_dir"],
            "--test-file", args.test_file,
        ])

    print(f"\n{'=' * 60}\nAll 5 ablation models finished.\n{'=' * 60}")
    for config in configs:
        print(f"  {config['name']}: see {config['output_dir']}/summary.md")


if __name__ == "__main__":
    main()
