from pathlib import Path
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
PROCESSED_DIR = DATA_DIR / "processed_data"
RESULTS_DIR = ROOT / "results" / "intermediate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DONOR_FILE  = PROCESSED_DIR / "matching_weighted_ks.csv"
DETAIL_FILE = RESULTS_DIR / "Table2_TF_ANN_detail.csv"
SUMMARY_FILE = RESULTS_DIR / "Table2_TF_ANN_summary.csv"


# ---------- config ----------
TARGET = "power"
BASE_FEATURES = [
    "wind_speed",
    "temperature",
    "turbulence_intensity",
    "std_wind_direction",
]
ANGLE_FEATURE = "wind_direction"

K           = 7
SEED        = 2026
TRAIN_YEAR  = 2017
TEST_YEARS  = [2017, 2018]

# ANN: 8-16-8
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

def read_donor_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing donor file: {path}")
    df = pd.read_csv(path)
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[0]: "target"})
    donor_cols = [c for c in df.columns if c.startswith("donor")]
    if not donor_cols:
        raise ValueError(f"No donor columns found in {path}")
    donor_cols = sorted(donor_cols, key=lambda x: int(x.replace("donor_", "").replace("donor", "")))
    keep = ["target"] + donor_cols
    df = df[keep].copy()
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    for c in donor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

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


# ---------- fit once on donor 2017, return fitted model ----------
def fit_donor_model(donor_id: int):
    feats    = feature_names()
    train_df = load_turbine_year(donor_id, TRAIN_YEAR)
    x_train  = train_df[feats].to_numpy()
    y_train  = train_df[TARGET].to_numpy()
    model    = build_ann()
    t0       = time.time()
    model.fit(x_train, y_train)
    fit_time = time.time() - t0
    print(f"    Fit donor {donor_id} done — {fit_time:.1f} sec", flush=True)
    return model, fit_time


# ---------- predict for one (target, year) using fitted model ----------
def predict_donor_model(model, target_id: int, test_year: int):
    feats   = feature_names()
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
    donor_df   = read_donor_table(DONOR_FILE)
    donor_cols = [c for c in donor_df.columns if c.startswith("donor")]
    targets    = sorted(donor_df["target"].unique())
    n_targets  = len(targets)

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

    print(f"[INFO] Starting TF_ANN — {n_targets} targets x {len(TEST_YEARS)} years", flush=True)

    for t_idx, target_id in enumerate(targets, 1):
        row = donor_df.loc[donor_df["target"] == target_id]
        if row.empty:
            continue
        donors = row[donor_cols].iloc[0].dropna().astype(int).tolist()
        donors = [d for d in donors if d != target_id][:K]
        if not donors:
            continue

        # skip entire target if all years done
        if all(f"{target_id}|{y}" in done_keys for y in TEST_YEARS):
            print(f"  [SKIP] Target {target_id} — all years done.", flush=True)
            continue

        print(f"[{t_idx}/{n_targets}] Target {target_id} | Donors: {donors}", flush=True)

        # accumulators per year
        pred_lists    = {y: [] for y in TEST_YEARS}
        actual_dict   = {y: None for y in TEST_YEARS}
        runtime_lists = {y: [] for y in TEST_YEARS}
        rmse_lists    = {y: [] for y in TEST_YEARS}
        mae_lists     = {y: [] for y in TEST_YEARS}
        donors_used   = {y: [] for y in TEST_YEARS}

        for donor_id in donors:
            print(f"    Fitting donor {donor_id} ...", flush=True)

            try:
                model, fit_time = fit_donor_model(donor_id)
            except Exception as e:
                print(f"    [ERROR] fit donor {donor_id}: {e}", flush=True)
                continue

            # predict for each year with the same fitted model
            for test_year in TEST_YEARS:
                pair_key = f"{target_id}|{test_year}"
                if pair_key in done_keys:
                    continue

                print(f"      Predicting year {test_year} ...", flush=True)
                try:
                    res = predict_donor_model(model, target_id, test_year)
                except Exception as e:
                    print(f"      [ERROR] predict year {test_year}: {e}", flush=True)
                    continue

                pred_lists[test_year].append(res["pred"])
                actual_dict[test_year]   = res["actual"]
                runtime_lists[test_year].append(fit_time + res["pred_time"])
                rmse_lists[test_year].append(res["rmse"])
                mae_lists[test_year].append(res["mae"])
                donors_used[test_year].append(donor_id)

                print(f"      Year {test_year} done — RMSE: {res['rmse']:.4f} | {res['pred_time']:.1f} sec",
                      flush=True)

        # save one row per year
        for test_year in TEST_YEARS:
            pair_key = f"{target_id}|{test_year}"
            if pair_key in done_keys:
                continue
            if not pred_lists[test_year]:
                continue

            ensemble_pred = np.mean(np.vstack(pred_lists[test_year]), axis=0)
            actual        = actual_dict[test_year]

            row_dict = {
                "method":           "TF_ANN",
                "target":           target_id,
                "year":             test_year,
                "donors_used":      ",".join(map(str, donors_used[test_year])),
                "n_models":         len(pred_lists[test_year]),
                "rmse":             rmse_fn(actual, ensemble_pred),
                "mae":              mae_fn(actual, ensemble_pred),
                "runtime_sec":      float(np.sum(runtime_lists[test_year])),
                "mean_single_rmse": float(np.mean(rmse_lists[test_year])),
                "mean_single_mae":  float(np.mean(mae_lists[test_year])),
            }

            # incremental append to CSV
            row_df = pd.DataFrame([row_dict])
            write_header = not DETAIL_FILE.exists() or DETAIL_FILE.stat().st_size == 0
            row_df.to_csv(DETAIL_FILE, mode="a", header=write_header, index=False)
            done_keys.add(pair_key)

            print(f"  -> Saved target {target_id} year {test_year}. Ensemble RMSE: {row_dict['rmse']:.4f}",
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
