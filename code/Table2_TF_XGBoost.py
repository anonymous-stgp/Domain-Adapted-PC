from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# ---------- paths ----------
ROOT = Path.cwd().resolve()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed_data"
RESULTS_DIR   = ROOT / "results" / "intermediate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- year argument: python code/Table2_G_XGBoost.py 2017
args       = sys.argv[1:]
TEST_YEARS = [int(args[0])] if args else [2017, 2018]
year_suffix = f"_{args[0]}" if args else ""

DONOR_FILE   = PROCESSED_DIR / "matching_weighted_ks.csv"
DETAIL_FILE  = RESULTS_DIR / f"Table2_TF_XGBoost_detail{year_suffix}.csv"
SUMMARY_FILE = RESULTS_DIR / f"Table2_TF_XGBoost_summary{year_suffix}.csv"

# ---------- config ----------
TARGET        = "power"
BASE_FEATURES = ["wind_speed", "temperature", "turbulence_intensity", "std_wind_direction"]
ANGLE_FEATURE = "wind_direction"
K             = 7
TRAIN_YEAR    = 2017


# ---------- utils ----------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2))) if np.any(m) else np.nan

def load_turbine_year(turbine_id, year):
    path = DATA_DIR / f"Turbine{int(turbine_id)}_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    required = BASE_FEATURES + [ANGLE_FEATURE, TARGET]
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

def read_geo_donor_table(path):
    df = pd.read_csv(path)
    # long format
    if {"donor", "geo_distance"}.issubset(df.columns):
        df["target"] = df["target"].astype(int)
        df["donor"]  = df["donor"].astype(int)
        df["geo_distance"] = pd.to_numeric(df["geo_distance"], errors="coerce")
        df = df.dropna(subset=["target", "donor", "geo_distance"])
        return df.sort_values(["target", "geo_distance", "donor"]).reset_index(drop=True)
    # wide format: target, donor1/donor_1, score1/score_1 ...
    donor_cols = [c for c in df.columns if c.startswith("donor") and c != "donor"]
    score_cols = [c for c in df.columns if c.startswith("score")]
    rows = []
    for dc, sc in zip(donor_cols, score_cols):
        tmp = pd.DataFrame({
            "target":       df["target"].astype(int),
            "donor":        pd.to_numeric(df[dc], errors="coerce").astype("Int64"),
            "geo_distance": pd.to_numeric(df[sc], errors="coerce"),
        })
        rows.append(tmp)
    long = pd.concat(rows, ignore_index=True).dropna()
    long["target"] = long["target"].astype(int)
    long["donor"]  = long["donor"].astype(int)
    return long.sort_values(["target", "geo_distance", "donor"]).reset_index(drop=True)

def top_k_geo_donors(geo_df, target_id, k):
    sub = geo_df[(geo_df["target"] == target_id) & (geo_df["donor"] != target_id)]
    return sub.sort_values(["geo_distance", "donor"])["donor"].head(k).astype(int).tolist()

def build_model(seed):
    return LGBMRegressor(
        objective="regression", n_estimators=200, learning_rate=0.1,
        max_depth=8, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1,
    )

def build_concat_training_data(donor_ids):
    feats, frames = feature_names(), []
    for d in donor_ids:
        try:
            frames.append(load_turbine_year(d, TRAIN_YEAR))
        except Exception:
            continue
    if not frames:
        return None, None
    train_df = pd.concat(frames, ignore_index=True)
    return train_df[feats].to_numpy(), train_df[TARGET].to_numpy()


def run():
    geo_df    = read_geo_donor_table(DONOR_FILE)
    targets   = sorted(geo_df["target"].unique())
    n_targets = len(targets)

    # --- resume ---
    done_keys = set()
    if DETAIL_FILE.exists() and DETAIL_FILE.stat().st_size > 0:
        try:
            prev = pd.read_csv(DETAIL_FILE)
            if {"target", "year"}.issubset(prev.columns):
                done_keys = set(f"{int(r.target)}|{int(r.year)}" for _, r in prev.iterrows())
                print(f"[INFO] Resuming — {len(done_keys)} pairs already done.", flush=True)
        except Exception:
            pass

    print(f"[INFO] Starting TF_XGBoost — {n_targets} targets x {len(TEST_YEARS)} years", flush=True)

    feats = feature_names()

    for t_idx, target_id in enumerate(targets, 1):
        donors = top_k_geo_donors(geo_df, target_id, K)
        if not donors:
            continue

        if all(f"{target_id}|{y}" in done_keys for y in TEST_YEARS):
            print(f"  [SKIP] Target {target_id} — all years done.", flush=True)
            continue

        print(f"[{t_idx}/{n_targets}] Target {target_id} | Donors: {donors}", flush=True)

        # build training data once
        x_train, y_train = build_concat_training_data(donors)
        if x_train is None:
            continue

        # fit model once
        model = build_model(seed=target_id)
        t0 = time.time()
        model.fit(x_train, y_train)
        fit_time = time.time() - t0
        print(f"  Fit done — {fit_time:.1f} sec", flush=True)

        # predict for each year
        for test_year in TEST_YEARS:
            pair_key = f"{target_id}|{test_year}"
            if pair_key in done_keys:
                print(f"  [SKIP] Year {test_year}", flush=True)
                continue

            try:
                test_df = load_turbine_year(target_id, test_year)
            except Exception as e:
                print(f"  [ERROR] load target {test_year}: {e}", flush=True)
                continue

            x_test = test_df[feats].to_numpy()
            y_test = test_df[TARGET].to_numpy()

            t0   = time.time()
            pred = model.predict(x_test)
            pred_time = time.time() - t0

            row = {
                "method":      "TF_XGBoost",
                "target":      target_id,
                "year":        test_year,
                "donors_used": ",".join(map(str, donors)),
                "n_models":    len(donors),
                "rmse":        rmse(y_test, pred),
                "runtime_sec": fit_time + pred_time,
            }

            row_df = pd.DataFrame([row])
            write_header = not DETAIL_FILE.exists() or DETAIL_FILE.stat().st_size == 0
            row_df.to_csv(DETAIL_FILE, mode="a", header=write_header, index=False)
            done_keys.add(pair_key)
            print(f"  -> Saved year {test_year}. RMSE: {row['rmse']:.4f}", flush=True)

    # final summary
    if DETAIL_FILE.exists() and DETAIL_FILE.stat().st_size > 0:
        detail_df  = pd.read_csv(DETAIL_FILE)
        summary_df = detail_df.groupby(["method", "year"], as_index=False).agg(
            avg_rmse=("rmse", "mean"),
            total_runtime_sec=("runtime_sec", "sum"),
        )
        summary_df.to_csv(SUMMARY_FILE, index=False)
        print("[DONE] Summary saved.", flush=True)


if __name__ == "__main__":
    run()
