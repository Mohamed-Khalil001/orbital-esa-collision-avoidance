"""Generate SYNTHETIC data with the same shape as the competition data (columns, events, CDMs).

WARNING: this data is fake. It has no real physical or statistical meaning.
Use it only to check that your environment, libraries, notebooks and app run on your machine.
Never put any result computed from it in your book or report.

Usage (from the project folder):
    python tools/make_synthetic_data.py
    # then, before starting Jupyter or Streamlit, point the project to the synthetic data:
    #   Linux/Mac:  export COLLISION_DATA_DIR="$(pwd)/data_synthetic"
    #   Windows  :  set COLLISION_DATA_DIR=%cd%/data_synthetic
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHARED = [
    "sigma_rdot", "sigma_n", "cn_r", "cn_t", "cndot_n", "sigma_ndot", "cndot_r", "cndot_rdot",
    "cndot_t", "cndot_tdot", "sigma_r", "ct_r", "sigma_t", "ctdot_n", "crdot_n", "crdot_t",
    "crdot_r", "ctdot_r", "ctdot_rdot", "ctdot_t", "sigma_tdot", "position_covariance_det",
    "cd_area_over_mass", "cr_area_over_mass", "h_apo", "h_per", "ecc", "j2k_inc", "j2k_sma",
    "sedr", "span", "rcs_estimate", "actual_od_span", "obs_available", "obs_used",
    "recommended_od_span", "residuals_accepted", "time_lastob_end", "time_lastob_start",
    "weighted_rms",
]
OBJECT_TYPES = ["PAYLOAD", "DEBRIS", "ROCKET BODY", "UNKNOWN"]


def make_events(n_events, rng, first_event_id=0, high_risk_share=0.15):
    rows = []
    for k in range(n_events):
        event_id = first_event_id + k
        n_cdm = int(rng.integers(3, 16))
        ends_late = rng.random() < 0.25                       # event whose last CDM is far from TCA
        last_t = rng.uniform(1.0, 3.0) if ends_late else rng.uniform(0.05, 0.95)
        t = np.sort(rng.uniform(last_t, last_t + rng.uniform(3, 6), n_cdm))[::-1]
        t[-1] = last_t

        is_high = rng.random() < high_risk_share
        final_risk = rng.uniform(-6, -3) if is_high else rng.uniform(-14, -6.2)
        miss0 = 10 ** rng.uniform(1.5, 3.5) if is_high else 10 ** rng.uniform(2.5, 4.5)
        obj = rng.choice(OBJECT_TYPES)
        mission = int(rng.integers(1, 8))

        # relative velocity direction; relative position is perpendicular to it (at closest approach)
        v_dir = rng.normal(size=3)
        v_dir /= np.linalg.norm(v_dir)
        speed = rng.uniform(500, 15000)
        r_dir = rng.normal(size=3)
        r_dir -= r_dir.dot(v_dir) * v_dir
        r_dir /= np.linalg.norm(r_dir)

        base_sigma = {p: 10 ** rng.uniform(0.5, 2.5) for p in "rtn"}
        for i in range(n_cdm):
            shrink = 0.6 + 0.4 * (t[i] - last_t + 0.1) / (t[0] - last_t + 0.1)
            risk = final_risk + (t[i] - last_t) * rng.normal(0, 0.6)
            miss = miss0 * (1 + 0.1 * rng.normal())
            r_vec = r_dir * miss
            v_vec = v_dir * speed
            row = {
                "event_id": event_id, "time_to_tca": t[i], "risk": float(np.clip(risk, -30, 0)),
                "mission_id": mission, "max_risk_estimate": risk + rng.normal(0, 0.3),
                "max_risk_scaling": rng.uniform(0.5, 5), "miss_distance": miss,
                "relative_speed": speed, "mahalanobis_distance": float(np.linalg.norm(r_vec) / (50 + 100 * rng.random())),
                "relative_position_r": r_vec[0], "relative_position_t": r_vec[1], "relative_position_n": r_vec[2],
                "relative_velocity_r": v_vec[0], "relative_velocity_t": v_vec[1], "relative_velocity_n": v_vec[2],
                "c_object_type": obj, "geocentric_latitude": rng.uniform(-90, 90),
                "azimuth": rng.uniform(-180, 180), "elevation": rng.uniform(-90, 90),
                "F10": rng.uniform(60, 200), "AP": rng.uniform(0, 40),
                "F3M": rng.uniform(60, 200), "SSN": rng.uniform(0, 150),
            }
            for prefix in ("t", "c"):
                for name in SHARED:
                    row[f"{prefix}_{name}"] = rng.normal()
                for axis in "rtn":
                    row[f"{prefix}_sigma_{axis}"] = base_sigma[axis] * shrink * (2 if prefix == "c" else 1)
                row[f"{prefix}_ct_r"] = rng.uniform(-0.5, 0.5)
                row[f"{prefix}_cn_r"] = rng.uniform(-0.5, 0.5)
                row[f"{prefix}_cn_t"] = rng.uniform(-0.5, 0.5)
                row[f"{prefix}_span"] = rng.uniform(1, 10)
                row[f"{prefix}_position_covariance_det"] = (
                    row[f"{prefix}_sigma_r"] * row[f"{prefix}_sigma_t"] * row[f"{prefix}_sigma_n"])
                row[f"{prefix}_actual_od_span"] = rng.uniform(2, 10)
                row[f"{prefix}_obs_used"] = int(rng.integers(50, 500))
            rows.append(row)
    df = pd.DataFrame(rows)

    # random missing values in a few columns
    for col in rng.choice([c for c in df.columns if c.startswith(("t_", "c_")) and c != "c_object_type"],
                          size=10, replace=False):
        df.loc[rng.random(len(df)) < 0.05, col] = np.nan
    return df


def main(out_dir="data_synthetic", n_train=2500, n_test=500, seed=0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = make_events(n_train, rng, first_event_id=0)
    train.to_csv(out / "train_data.csv", index=False)

    # test: last CDM within 1 day, and only CDMs with time_to_tca >= 2 (as in the competition)
    test_full = make_events(n_test, rng, first_event_id=100000)
    last_t = test_full.groupby("event_id")["time_to_tca"].transform("min")
    test = test_full[(last_t < 1.0) & (test_full["time_to_tca"] >= 2.0)]
    test.to_csv(out / "test_data.csv", index=False)

    print(f"train: {len(train):,} rows, {train['event_id'].nunique():,} events, {train.shape[1]} columns")
    print(f"test : {len(test):,} rows, {test['event_id'].nunique():,} events")
    print(f"written to: {out.resolve()}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["data_synthetic"]))
