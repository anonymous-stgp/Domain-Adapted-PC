from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------- paths ----------
ROOT = Path.cwd().resolve()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "intermediate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- CLI arg: turbine range "start:end" (e.g. "1:22"). Default = all. ----------
TURBINE_RANGE = None
if len(sys.argv) >= 2:
    parts = sys.argv[1].split(":")
    if len(parts) == 2:
        TURBINE_RANGE = list(range(int(parts[0]), int(parts[1]) + 1))

RANGE_TAG = f"_t{min(TURBINE_RANGE)}to{max(TURBINE_RANGE)}" if TURBINE_RANGE else ""

DETAIL_FILE  = RESULTS_DIR / f"Table2_P_ANN{RANGE_TAG}_detail.csv"
SUMMARY_FILE = RESULTS_DIR / f"Table2_P_ANN{RANGE_TAG}_summary.csv"


# ---------- config ----------
TARGET = "power"
BASE_FEATURES = [
    "wind_speed",
    "temperature",
    "turbulence_intensity",
    "std_wind_direction",
]
ANGLE_FEATURE = "wind_direction"

SEED        = 42
TRAIN_YEAR  = 2017
TEST_YEARS  = [2017, 2018]

ALL_IDS = list(range(1, 67))

# ANN: 8-16-8 (matches uploaded Table2_TF_ANN.py exactly)
HIDDEN_LAYERS = (8, 16, 8)
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64
EPOCHS        = 100


# ---------- utils ----------
def mae_fn(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def rmse_fn(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def load_turbine_year(turbine_id: int, year: int) -> pd.DataFrame:
    path = DATA_DIR / f"Turbine{int(turbine_id)}_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = BASE_FEATURES + [ANGLE_FEATURE, TARGET]
    missing = [c for c in required if c not in df.columns]
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

def feature_names():
    return BASE_FEATURES + ["wind_direction_sin", "wind_direction_cos"]

def build_ann():
    return Pipeline(steps=[
        ("x_scaler", StandardScaler()),
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


# ---------- fit once on pooled (all-but-target) 2017, return fitted model ----------
def fit_pooled_model(target_id: int, feats: list):
    train_ids = [t for t in ALL_IDS if t != target_id]

    x_list, y_list = [], []
    for tid in train_ids:
        try:
            df = load_turbine_year(tid, TRAIN_YEAR)
        except Exception as e:
            print(f"    [WARN] could not load train turbine {tid}: {e}", flush=True)
            continue
        x_list.append(df[feats].to_numpy())
        y_list.append(df[TARGET].to_numpy())

    if not x_list:
        raise RuntimeError(f"No training data available for target {target_id}")

    x_train = np.vstack(x_list)
    y_train = np.concatenate(y_list)

    model = build_ann()
    t0 = time.time()
    model.fit(x_train, y_train)
    fit_time = time.time() - t0
    print(f"    Fit pooled model for target {target_id} done — "
          f"{fit_time:.1f} sec ({len(train_ids)} train turbines, {len(y_train)} rows)", flush=True)
    return model, fit_time


# ---------- predict for one (target, year) using fitted model ----------
def predict_pooled_model(model, target_id: int, test_year: int, feats: list):
    test_df = load_turbine_year(target_id, test_year)
    x_test  = test_df[feats].to_numpy()
    y_test  = test_df[TARGET].to_numpy()
    t0      = time.time()
    pred    = model.predict(x_test)
    pred_time = time.time() - t0
    return {
        "pred":      pred,
        "actual":    y_test,
        "rmse":      rmse_fn(y_test, pred),
        "mae":       mae_fn(y_test, pred),
        "pred_time": pred_time,
    }


def run():
    feats = feature_names()
    targets = ALL_IDS
    if TURBINE_RANGE is not None:
        targets = [t for t in targets if t in TURBINE_RANGE]
    n_targets = len(targets)

    # --- resume: load already-done (target, year) pairs ---
    done_keys = set()
    if DETAIL_FILE.exists() and DETAIL_FILE.stat().st_size > 0:
        try:
            prev = pd.read_csv(DETAIL_FILE)
            if {"target", "year"}.issubset(prev.columns):
                done_keys = set(
                    f"{int(r.target)}|{int(r.year)}"
                    for _, r in prev.iterrows()
                )
                print(f"[INFO] Resuming — {len(done_keys)} pairs already done.", flush=True)
        except Exception:
            pass

    print(f"[INFO] Starting P_ANN — {n_targets} targets x {len(TEST_YEARS)} years", flush=True)

    for t_idx, target_id in enumerate(targets, 1):
        years_needed = [y for y in TEST_YEARS if f"{target_id}|{y}" not in done_keys]
        if not years_needed:
            print(f"  [SKIP] Target {target_id} — all years done.", flush=True)
            continue

        print(f"[{t_idx}/{n_targets}] Target {target_id}", flush=True)

        try:
            model, fit_time = fit_pooled_model(target_id, feats)
        except Exception as e:
            print(f"    [ERROR] fit target {target_id}: {e}", flush=True)
            continue

        for test_year in years_needed:
            print(f"      Predicting year {test_year} ...", flush=True)
            try:
                res = predict_pooled_model(model, target_id, test_year, feats)
            except Exception as e:
                print(f"      [ERROR] predict year {test_year}: {e}", flush=True)
                continue

            row_dict = {
                "method":      "P_ANN",
                "target":      target_id,
                "year":        test_year,
                "n_train_turbines": len(ALL_IDS) - 1,
                "rmse":        res["rmse"],
                "mae":         res["mae"],
                "runtime_sec": float(fit_time + res["pred_time"]),
            }

            row_df = pd.DataFrame([row_dict])
            write_header = not DETAIL_FILE.exists() or DETAIL_FILE.stat().st_size == 0
            row_df.to_csv(DETAIL_FILE, mode="a", header=write_header, index=False)
            done_keys.add(f"{target_id}|{test_year}")

            print(f"      -> Saved target {target_id} year {test_year}. RMSE: {row_dict['rmse']:.4f}",
                  flush=True)

    # final summary
    if DETAIL_FILE.exists() and DETAIL_FILE.stat().st_size > 0:
        detail_df  = pd.read_csv(DETAIL_FILE)
        summary_df = (
            detail_df.groupby(["method", "year"], as_index=False)
            .agg(
                avg_rmse=("rmse", "mean"),
                avg_mae=("mae", "mean"),
                total_runtime_sec=("runtime_sec", "sum"),
            )
        )
        summary_df.to_csv(SUMMARY_FILE, index=False)
        print("[DONE] Summary saved.", flush=True)


if __name__ == "__main__":
    run()
