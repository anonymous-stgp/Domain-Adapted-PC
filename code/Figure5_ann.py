"""
Figure5_ann.py  —  ANN (8-16-8, ReLU, Adam) over K = 2..10
=============================================================
Two aggregation strategies:
  ensemble : train one ANN per donor on that donor's 2017 data,
             average predictions over K models
  concat   : pool all K donors' 2017 data into one training set,
             train one ANN, predict

Outputs (in results/intermediate/):
  Figure5_ann[_tXtoY]_detail.csv   — one row per (aggregation, K, target, year)
  Figure5_ann[_tXtoY]_summary.csv  — avg RMSE/MAE per (aggregation, K, year)

CLI:
  python Figure5_ann.py [turbine_range] [k_range]
  python Figure5_ann.py 1:11 2:10
  python Figure5_ann.py          # runs all turbines, K=2:10
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────

ROOT = Path.cwd().resolve()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed_data"
RESULTS_DIR   = ROOT / "results" / "intermediate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DONOR_FILE = PROCESSED_DIR / "matching_weighted_ks.csv"

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

TARGET        = "power"
BASE_FEATURES = ["wind_speed", "temperature", "turbulence_intensity", "std_wind_direction"]
ANGLE_FEATURE = "wind_direction"

SEED         = 42
TRAIN_YEAR   = 2017
TEST_YEARS   = [2017, 2018]
K_VALUES     = list(range(2, 11))   # 2..10

HIDDEN_LAYERS = (8, 16, 8)
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64
EPOCHS        = 100

DETAIL_COLS = [
    "model", "aggregation", "K", "target", "year",
    "donors_used", "n_models",
    "rmse", "mae", "runtime_sec",
]

# ─────────────────────────────────────────
# CLI ARGS
# arg1: turbine range "start:end"
# arg2: K range       "kmin:kmax"
# ─────────────────────────────────────────

TURBINE_RANGE = None
if len(sys.argv) >= 2:
    parts = sys.argv[1].split(":")
    if len(parts) == 2:
        TURBINE_RANGE = list(range(int(parts[0]), int(parts[1]) + 1))

if len(sys.argv) >= 3:
    kparts = sys.argv[2].split(":")
    if len(kparts) == 2:
        K_VALUES = list(range(int(kparts[0]), int(kparts[1]) + 1))

RANGE_TAG = (
    f"_t{min(TURBINE_RANGE)}to{max(TURBINE_RANGE)}" if TURBINE_RANGE else ""
)

DETAIL_FILE  = RESULTS_DIR / f"Figure5_ann{RANGE_TAG}_detail.csv"
SUMMARY_FILE = RESULTS_DIR / f"Figure5_ann{RANGE_TAG}_summary.csv"

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def rmse_fn(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae_fn(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def feature_names():
    return BASE_FEATURES + ["wind_direction_sin", "wind_direction_cos"]

def load_turbine_year(turbine_id: int, year: int) -> pd.DataFrame:
    path = DATA_DIR / f"Turbine{int(turbine_id)}_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = BASE_FEATURES + [ANGLE_FEATURE, TARGET]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().copy()
    if df.empty:
        raise ValueError(f"No usable rows in {path}")
    rad = np.deg2rad(df[ANGLE_FEATURE].to_numpy())
    df["wind_direction_sin"] = np.sin(rad)
    df["wind_direction_cos"] = np.cos(rad)
    df = df.drop(columns=[ANGLE_FEATURE])
    return df

def read_donor_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing donor file: {path}")
    df = pd.read_csv(path)
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[0]: "target"})
    donor_cols = sorted(
        [c for c in df.columns if c.startswith("donor")],
        key=lambda x: int(x.replace("donor_", "").replace("donor", ""))
    )
    if not donor_cols:
        raise ValueError(f"No donor columns in {path}")
    df = df[["target"] + donor_cols].copy()
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    for c in donor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, donor_cols

def build_ann():
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("ann", MLPRegressor(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            learning_rate_init=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            max_iter=EPOCHS,
            random_state=SEED,
            shuffle=True,
            early_stopping=False,
        )),
    ])

def load_done_keys() -> set:
    if not DETAIL_FILE.exists() or DETAIL_FILE.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(DETAIL_FILE)
        req = {"model", "aggregation", "K", "target", "year"}
        if not req.issubset(df.columns):
            return set()
        return set(
            f"ann|{r.aggregation}|{int(r.K)}|{int(r.target)}|{int(r.year)}"
            for _, r in df.iterrows()
        )
    except Exception:
        return set()

def make_key(agg, k, target, year):
    return f"ann|{agg}|{int(k)}|{int(target)}|{int(year)}"

def append_row(row: dict):
    row_df = pd.DataFrame([row])
    write_header = not DETAIL_FILE.exists() or DETAIL_FILE.stat().st_size == 0
    row_df.to_csv(DETAIL_FILE, mode="a", header=write_header, index=False)

# ─────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────

def run_ensemble(donors, target_id, test_year, feats):
    test_df = load_turbine_year(target_id, test_year)
    x_test  = test_df[feats].to_numpy()
    y_test  = test_df[TARGET].to_numpy()

    preds    = []
    runtimes = []

    for donor_id in donors:
        try:
            train_df = load_turbine_year(donor_id, TRAIN_YEAR)
            x_train  = train_df[feats].to_numpy()
            y_train  = train_df[TARGET].to_numpy()

            np.random.seed(SEED)
            model = build_ann()
            t0    = time.time()
            model.fit(x_train, y_train)
            pred  = model.predict(x_test)
            rt    = time.time() - t0

            preds.append(pred)
            runtimes.append(rt)
        except Exception as e:
            print(f"    [ERROR] ensemble donor {donor_id}: {e}", flush=True)
            continue

    if not preds:
        return None

    ensemble_pred = np.mean(np.vstack(preds), axis=0)
    return {
        "pred":        ensemble_pred,
        "actual":      y_test,
        "rmse":        rmse_fn(y_test, ensemble_pred),
        "mae":         mae_fn(y_test, ensemble_pred),
        "runtime_sec": float(np.sum(runtimes)),
        "n_models":    len(preds),
    }

# ─────────────────────────────────────────
# CONCAT
# ─────────────────────────────────────────

def run_concat(donors, target_id, test_year, feats):
    test_df = load_turbine_year(target_id, test_year)
    x_test  = test_df[feats].to_numpy()
    y_test  = test_df[TARGET].to_numpy()

    x_parts, y_parts = [], []
    for donor_id in donors:
        try:
            train_df = load_turbine_year(donor_id, TRAIN_YEAR)
            x_parts.append(train_df[feats].to_numpy())
            y_parts.append(train_df[TARGET].to_numpy())
        except Exception as e:
            print(f"    [ERROR] concat load donor {donor_id}: {e}", flush=True)
            continue

    if not x_parts:
        return None

    x_train = np.vstack(x_parts)
    y_train = np.concatenate(y_parts)

    np.random.seed(SEED)
    model = build_ann()
    t0    = time.time()
    model.fit(x_train, y_train)
    pred  = model.predict(x_test)
    rt    = time.time() - t0

    return {
        "pred":        pred,
        "actual":      y_test,
        "rmse":        rmse_fn(y_test, pred),
        "mae":         mae_fn(y_test, pred),
        "runtime_sec": float(rt),
        "n_models":    1,
    }

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    donor_df, donor_cols = read_donor_table(DONOR_FILE)
    feats     = feature_names()
    done_keys = load_done_keys()

    targets = sorted(donor_df["target"].unique())
    if TURBINE_RANGE is not None:
        targets = [t for t in targets if t in TURBINE_RANGE]

    print(f"Model         : ANN {HIDDEN_LAYERS}")
    print(f"Turbines      : {len(targets)}")
    print(f"K values      : {K_VALUES}")
    print(f"Years         : {TEST_YEARS}")
    print(f"Seed          : {SEED}")
    print(f"Already done  : {len(done_keys)} keys\n", flush=True)

    for target_id in targets:
        row = donor_df.loc[donor_df["target"] == target_id]
        if row.empty:
            continue
        all_donors = (
            row[donor_cols].iloc[0]
            .dropna().astype(int).tolist()
        )
        all_donors = [d for d in all_donors if d != target_id]

        for k in K_VALUES:
            donors = all_donors[:k]
            if len(donors) < k:
                print(f"  Skip K={k} target={target_id} — not enough donors", flush=True)
                continue

            for agg in ["ensemble", "concat"]:
                for test_year in TEST_YEARS:
                    key = make_key(agg, k, target_id, test_year)
                    if key in done_keys:
                        print(f"  Skip: {agg} K={k} target={target_id} year={test_year}", flush=True)
                        continue

                    print(f"  [ann|{agg}] K={k}  target={target_id}  year={test_year}", flush=True)

                    try:
                        res = (
                            run_ensemble(donors, target_id, test_year, feats)
                            if agg == "ensemble"
                            else run_concat(donors, target_id, test_year, feats)
                        )
                    except Exception as e:
                        print(f"  [ERROR] {e}", flush=True)
                        res = None

                    if res is None:
                        continue

                    append_row({
                        "model":       "ann",
                        "aggregation": agg,
                        "K":           k,
                        "target":      target_id,
                        "year":        test_year,
                        "donors_used": ",".join(map(str, donors)),
                        "n_models":    res["n_models"],
                        "rmse":        res["rmse"],
                        "mae":         res["mae"],
                        "runtime_sec": res["runtime_sec"],
                    })
                    done_keys.add(key)

                    print(
                        f"    -> RMSE: {res['rmse']:.4f}  MAE: {res['mae']:.4f}"
                        f"  time: {res['runtime_sec']:.1f}s",
                        flush=True
                    )

    # summary
    if DETAIL_FILE.exists() and DETAIL_FILE.stat().st_size > 0:
        detail_df  = pd.read_csv(DETAIL_FILE)
        summary_df = (
            detail_df
            .groupby(["model", "aggregation", "K", "year"], as_index=False)
            .agg(
                avg_rmse          =("rmse",        "mean"),
                avg_mae           =("mae",          "mean"),
                total_runtime_sec =("runtime_sec",  "sum"),
                n_targets         =("target",        "count"),
            )
        )
        summary_df.to_csv(SUMMARY_FILE, index=False)
        print(f"\nSummary saved to: {SUMMARY_FILE}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
