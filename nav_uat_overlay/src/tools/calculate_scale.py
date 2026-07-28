#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_trials(csv_file):
    trials = defaultdict(dict)

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["section"] != "summary":
                continue

            trial = int(row["trial"])
            key = row["key"]
            value = row["value"]

            try:
                value = float(value)
            except Exception:
                pass

            trials[trial][key] = value

    return trials


def recommended_scale(summary):
    ratio = summary.get("command_response_ratio", None)

    if ratio is None:
        return None

    if isinstance(ratio, str):
        return None

    if abs(ratio) < 1e-8:
        return None

    return 1.0 / abs(ratio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    trials = load_trials(args.csv)

    if len(trials) == 0:
        raise RuntimeError("No summary section found.")

    scales = []

    print("=" * 70)

    for trial in sorted(trials):

        summary = trials[trial]

        scale = recommended_scale(summary)

        if scale is None:
            print(f"Trial {trial}: skipped")
            continue

        scales.append(scale)

        if "motion" in summary:
            name = summary["motion"]
        else:
            target = summary.get("target_yaw_rate", 0.0)
            name = "angular_positive" if target > 0 else "angular_negative"

        print(
            f"Trial {trial:2d} "
            f"{name:18s} "
            f"command_ratio={summary['command_response_ratio']:.4f} "
            f"recommended_scale={scale:.4f}"
        )

    print("=" * 70)

    scales = np.asarray(scales)

    print(f"Trials           : {len(scales)}")
    print(f"Mean scale       : {scales.mean():.4f}")
    print(f"Std              : {scales.std():.4f}")
    print(f"Min              : {scales.min():.4f}")
    print(f"Max              : {scales.max():.4f}")

    print("\nRecommended config value:\n")
    print(f"{scales.mean():.4f}")


if __name__ == "__main__":
    main()