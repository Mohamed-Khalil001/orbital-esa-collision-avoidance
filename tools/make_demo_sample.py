"""Create the small demo sample that ships with the repository.

The full ESA files are too large for git (see data/README.md), so the app falls back to a
small, reproducible subset of test_data.csv when the full files are not present.

Usage (from the project root):
    python tools/make_demo_sample.py --input data/test_data.csv
    python tools/make_demo_sample.py --input data/test_data.csv --events 150 --seed 42

The sample keeps every CDM of each selected event. About a quarter of the events are chosen
among those that contain at least one high-risk CDM, so the demo shows both quiet and alarming cases.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import config  # noqa: E402


def make_sample(df, n_events=150, high_share=0.27, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    max_risk = df.groupby(config.ID_COL)[config.RISK_COL].max()
    high_ids = max_risk.index[max_risk >= config.HIGH_RISK_THRESHOLD].to_numpy()
    other_ids = max_risk.index[max_risk < config.HIGH_RISK_THRESHOLD].to_numpy()

    n_high = min(int(round(n_events * high_share)), len(high_ids))
    n_other = min(n_events - n_high, len(other_ids))
    chosen = np.concatenate([
        rng.choice(high_ids, size=n_high, replace=False),
        rng.choice(other_ids, size=n_other, replace=False),
    ])
    out = df[df[config.ID_COL].isin(chosen)]
    return out.sort_values([config.ID_COL, config.TIME_COL], ascending=[True, False]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build data/sample/demo_events.csv")
    parser.add_argument("--input", default=str(config.DATA_DIR / "test_data.csv"))
    parser.add_argument("--output", default=str(config.SAMPLE_DIR / "demo_events.csv"))
    parser.add_argument("--events", type=int, default=150)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sample = make_sample(df, n_events=args.events, seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {sample[config.ID_COL].nunique()} events, {len(sample)} CDM rows")


if __name__ == "__main__":
    main()
