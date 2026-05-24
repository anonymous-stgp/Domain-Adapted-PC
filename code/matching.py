"""
Turbine Matching — FIR-weighted version
========================================
Exact replication of matching.ipynb logic (structure, distance functions,
ranking, output files) with FIR / Joseph weights substituted in place of
the old manual weights.

Outputs (written to OUT_DIR):
  pairwise_matching_<YEAR>.csv       — all pairwise distances
  pairwise_geographic_distance.csv   — Euclidean lat/lon distances
  matching_weighted_ks.csv           — donor rankings by WKS
  matching_mean_ks.csv               — donor rankings by mean KS
  matching_marginal_energy.csv       — donor rankings by energy distance
  matching_sinkhorn_wasserstein.csv  — donor rankings by Sinkhorn
  matching_geographic_distance.csv   — donor rankings by geo distance
  matching_files_created.csv         — summary of files written
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, energy_distance

# ─────────────────────────────────────────
# CONFIG  (edit these as needed)
# ─────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR = ROOT / "data"
OUT_DIR  = DATA_DIR / "processed_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR          = 2017
TOP_K         = 65
MAX_SAMPLES   = 800
SINKHORN_EPS  = 0.05
SINKHORN_ITERS = 200

CANDIDATE_FEATURES = [
    "wind_speed",
    "wind_direction",
    "temperature",
    "turbulence_intensity",
    "std_wind_direction",
]

# FIR / Joseph weights  (replaces the old manual 80/5/10/3/2 weights)
# Features with weight 0.0 still contribute to mean_KS / energy / sinkhorn,
# but are excluded from the weighted_ks sum (their ws entry is 0 → renorm drops them).
FEATURE_WEIGHTS = {
    "wind_speed":           0.494200186,
    "wind_direction":       0.003509981,
    "temperature":          0.009810552,
    "turbulence_intensity": 0.0,
    "std_wind_direction":   0.0,
}


# ─────────────────────────────────────────
# IO helpers  (identical to matching.ipynb)
# ─────────────────────────────────────────

def find_turbine_files(data_dir: Path, year: int) -> dict:
    rx = re.compile(rf"^Turbine(\d+)_{year}\.csv$", re.IGNORECASE)
    files = {}
    for p in data_dir.iterdir():
        m = rx.match(p.name)
        if m:
            files[int(m.group(1))] = p
    return dict(sorted(files.items()))


def clean_array(x) -> np.ndarray:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def subsample_sorted(x, max_samples: int = MAX_SAMPLES) -> np.ndarray:
    x = np.sort(clean_array(x))
    if len(x) <= max_samples:
        return x
    idx = np.linspace(0, len(x) - 1, max_samples).round().astype(int)
    return x[idx]


# ─────────────────────────────────────────
# Distance functions  (identical to matching.ipynb)
# ─────────────────────────────────────────

def sinkhorn_distance_1d(x, y, eps: float = SINKHORN_EPS,
                         n_iter: int = SINKHORN_ITERS) -> float:
    x = subsample_sorted(x)
    y = subsample_sorted(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    a = np.full(len(x), 1.0 / len(x))
    b = np.full(len(y), 1.0 / len(y))
    C = np.abs(x[:, None] - y[None, :])
    K = np.exp(-C / max(eps, 1e-8))
    u = np.ones(len(x))
    v = np.ones(len(y))
    for _ in range(n_iter):
        Kv = K @ v
        Kv[Kv == 0] = 1e-300
        u = a / Kv
        KTu = K.T @ u
        KTu[KTu == 0] = 1e-300
        v = b / KTu
    P = (u[:, None] * K) * v[None, :]
    return float(np.sum(P * C))


def average_ks(df_a: pd.DataFrame, df_b: pd.DataFrame,
               features: list) -> float:
    vals = []
    for col in features:
        a = clean_array(df_a[col])
        b = clean_array(df_b[col])
        if len(a) == 0 or len(b) == 0:
            continue
        vals.append(ks_2samp(a, b).statistic)
    return np.nan if len(vals) == 0 else float(np.mean(vals))


def weighted_ks(df_a: pd.DataFrame, df_b: pd.DataFrame,
                features: list, weights: dict) -> float:
    """
    Weighted mean KS.  Features with weight 0 are included in the KS
    computation but their weight is 0, so after renormalisation they
    contribute nothing — identical behaviour to matching.ipynb.
    """
    vals, ws = [], []
    for col in features:
        a = clean_array(df_a[col])
        b = clean_array(df_b[col])
        if len(a) == 0 or len(b) == 0:
            continue
        vals.append(ks_2samp(a, b).statistic)
        ws.append(weights[col])
    if len(vals) == 0:
        return np.nan
    ws = np.array(ws, dtype=float)
    total = ws.sum()
    if total <= 0:                      # all weights zero → fall back to mean
        return float(np.mean(vals))
    ws = ws / total
    return float(np.dot(ws, vals))


def average_energy(df_a: pd.DataFrame, df_b: pd.DataFrame,
                   features: list) -> float:
    vals = []
    for col in features:
        a = clean_array(df_a[col])
        b = clean_array(df_b[col])
        if len(a) == 0 or len(b) == 0:
            continue
        vals.append(energy_distance(a, b))
    return np.nan if len(vals) == 0 else float(np.mean(vals))


def average_sinkhorn(df_a: pd.DataFrame, df_b: pd.DataFrame,
                     features: list) -> float:
    vals = []
    for col in features:
        a = clean_array(df_a[col])
        b = clean_array(df_b[col])
        if len(a) == 0 or len(b) == 0:
            continue
        vals.append(sinkhorn_distance_1d(a, b))
    return np.nan if len(vals) == 0 else float(np.mean(vals))


# ─────────────────────────────────────────
# Ranking helper  (identical to matching.ipynb)
# ─────────────────────────────────────────

def ranking_table(df: pd.DataFrame, score_col: str,
                  top_k: int = TOP_K) -> pd.DataFrame:
    out = []
    for target, g in df.groupby("target"):
        g = g.sort_values([score_col, "donor"]).head(top_k).reset_index(drop=True)
        row = {"target": target}
        for r, (_, rr) in enumerate(g.iterrows(), start=1):
            row[f"donor_{r}"] = int(rr["donor"])
            row[f"score_{r}"] = float(rr[score_col])
        out.append(row)
    return pd.DataFrame(out).sort_values("target").reset_index(drop=True)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    # ── load turbine files ──────────────────────────────────────────────
    files = find_turbine_files(DATA_DIR, YEAR)
    if not files:
        raise FileNotFoundError(f"No turbine files found in {DATA_DIR} for year {YEAR}.")

    sample_df = pd.read_csv(next(iter(files.values())))
    features = [c for c in CANDIDATE_FEATURES if c in sample_df.columns]
    if not features:
        raise ValueError("None of the candidate features were found in the turbine files.")

    weights = {k: FEATURE_WEIGHTS[k] for k in features}
    turbines = {tid: pd.read_csv(path) for tid, path in files.items()}

    print(f"Loaded turbines : {len(turbines)}")
    print(f"Features        : {features}")
    print(f"Weights (FIR)   : { {k: weights[k] for k in features} }")

    # ── pairwise distances ──────────────────────────────────────────────
    rows = []
    ids  = sorted(turbines.keys())
    total_pairs = len(ids) * (len(ids) - 1)
    done = 0

    for target in ids:
        df_t = turbines[target]
        for donor in ids:
            if donor == target:
                continue
            df_d = turbines[donor]
            rows.append({
                "target":               target,
                "donor":                donor,
                "weighted_ks":          weighted_ks(df_t, df_d, features, weights),
                "mean_ks":              average_ks(df_t, df_d, features),
                "marginal_energy":      average_energy(df_t, df_d, features),
                "sinkhorn_wasserstein": average_sinkhorn(df_t, df_d, features),
            })
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{total_pairs} pairs done …")

    pairwise = (
        pd.DataFrame(rows)
        .sort_values(["target", "donor"])
        .reset_index(drop=True)
    )
    pairwise.to_csv(OUT_DIR / f"pairwise_matching_{YEAR}.csv", index=False)
    print(f"\nSaved pairwise_matching_{YEAR}.csv  ({len(pairwise)} rows)")

    # ── geographic distances ────────────────────────────────────────────
    loc_path = DATA_DIR / "location.csv"
    if not loc_path.exists():
        raise FileNotFoundError(f"Missing {loc_path}")

    loc = pd.read_csv(loc_path, encoding="utf-8-sig")
    loc.columns = [str(c).strip().replace("\ufeff", "") for c in loc.columns]

    id_col  = next((c for c in loc.columns if c.lower() in ["wt", "turbine", "id"]), None)
    lon_col = next((c for c in loc.columns if c.lower() == "longitude"), None)
    lat_col = next((c for c in loc.columns if c.lower() == "latitude"),  None)

    if id_col is None or lon_col is None or lat_col is None:
        raise ValueError(
            f"location.csv columns {list(loc.columns)} — expected WT/turbine/id, Longitude, Latitude"
        )

    loc = loc[[id_col, lon_col, lat_col]].copy()
    loc.columns = ["target", "Longitude", "Latitude"]
    for col in ["target", "Longitude", "Latitude"]:
        loc[col] = pd.to_numeric(loc[col], errors="coerce")
    loc = loc.dropna().copy()
    loc["target"] = loc["target"].astype(int)

    geo_rows = []
    for _, ri in loc.iterrows():
        for _, rj in loc.iterrows():
            if ri["target"] == rj["target"]:
                continue
            d = np.sqrt(
                (ri["Longitude"] - rj["Longitude"]) ** 2 +
                (ri["Latitude"]  - rj["Latitude"])  ** 2
            )
            geo_rows.append({
                "target":       int(ri["target"]),
                "donor":        int(rj["target"]),
                "geo_distance": float(d),
            })

    geo_pairwise = (
        pd.DataFrame(geo_rows)
        .sort_values(["target", "geo_distance", "donor"])
        .reset_index(drop=True)
    )
    geo_pairwise.to_csv(OUT_DIR / "pairwise_geographic_distance.csv", index=False)
    print("Saved pairwise_geographic_distance.csv")

    # ── ranking tables ──────────────────────────────────────────────────
    metric_map = {
        "weighted_ks":          "matching_weighted_ks.csv",
        "mean_ks":              "matching_mean_ks.csv",
        "marginal_energy":      "matching_marginal_energy.csv",
        "sinkhorn_wasserstein": "matching_sinkhorn_wasserstein.csv",
    }

    for metric, fname in metric_map.items():
        ranking_table(
            pairwise[["target", "donor", metric]].copy(), metric
        ).to_csv(OUT_DIR / fname, index=False)
        print(f"Saved {fname}")

    ranking_table(
        geo_pairwise[["target", "donor", "geo_distance"]].copy(), "geo_distance"
    ).to_csv(OUT_DIR / "matching_geographic_distance.csv", index=False)
    print("Saved matching_geographic_distance.csv")

    # ── summary ─────────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "file": [
            f"pairwise_matching_{YEAR}.csv",
            "pairwise_geographic_distance.csv",
            "matching_weighted_ks.csv",
            "matching_mean_ks.csv",
            "matching_marginal_energy.csv",
            "matching_sinkhorn_wasserstein.csv",
            "matching_geographic_distance.csv",
        ]
    })
    summary.to_csv(OUT_DIR / "matching_files_created.csv", index=False)
    print("\nDone. Files written to:", OUT_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
