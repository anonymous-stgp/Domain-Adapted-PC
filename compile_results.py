"""
compile_results.py
==================
Reads all intermediate summary CSVs and produces:
    results/final_results.csv

Columns:
  table, group, method, rmse_2017, gap_2017, rmse_2018, gap_2018,
  runtime_2017_sec, runtime_2018_sec

Table mapping:
  Table1_Metrics_LOTO  : LOTO metric comparison (weighted_ks, mean_ks, sinkhorn, marginal_energy)
  Table2_LOTO          : LOTO method comparison
  Table2_DFP           : DFP method comparison

TF method mapping (from Table1 summary files):
  weighted_ks          -> thinnedSV
  mean_ks              -> ANN
  sinkhorn_wasserstein -> thinned twinGP
  (marginal_energy excluded from TF rows — it maps to no named TF method)
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT      = Path(__file__).resolve().parent
INTER_DIR = ROOT / "results" / "intermediate"
OUT_FILE  = ROOT / "results" / "final_results.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# helpers
# ─────────────────────────────────────────

def gap(rmse, best):
    if pd.isna(rmse) or pd.isna(best) or best == 0:
        return np.nan
    return round((rmse - best) * 100 / best, 2)


def read_summary(path, rmse_col="avg_rmse", year_col="year", runtime_col="total_runtime_sec"):
    """Return dict {2017: {rmse, runtime}, 2018: {rmse, runtime}} from a summary CSV."""
    empty = {2017: {"rmse": np.nan, "runtime": np.nan},
             2018: {"rmse": np.nan, "runtime": np.nan}}
    if not path.exists():
        return empty
    df = pd.read_csv(path)
    if year_col not in df.columns:
        return empty
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    # detect rmse col
    rc = rmse_col if rmse_col in df.columns else next(
        (c for c in df.columns if "rmse" in c.lower()), None)
    # detect runtime col
    rtc = runtime_col if runtime_col in df.columns else next(
        (c for c in df.columns if "runtime" in c.lower()), None)
    out = {}
    for yr in [2017, 2018]:
        row = df[df[year_col] == yr]
        out[yr] = {
            "rmse":    float(row[rc].values[0])   if (not row.empty and rc)   else np.nan,
            "runtime": float(row[rtc].values[0])  if (not row.empty and rtc)  else np.nan,
        }
    return out


def read_summary_by_metric(path, metric_col, rmse_col="avg_rmse",
                            year_col="year", runtime_col="total_runtime_sec"):
    """Return {metric: {2017: {rmse, runtime}, 2018: {rmse, runtime}}}"""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if year_col not in df.columns:
        return {}
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    rc  = rmse_col    if rmse_col    in df.columns else next((c for c in df.columns if "rmse"    in c.lower()), None)
    rtc = runtime_col if runtime_col in df.columns else next((c for c in df.columns if "runtime" in c.lower()), None)
    out = {}
    for metric in df[metric_col].unique():
        sub = df[df[metric_col] == metric]
        out[metric] = {}
        for yr in [2017, 2018]:
            row = sub[sub[year_col] == yr]
            out[metric][yr] = {
                "rmse":    float(row[rc].values[0])  if (not row.empty and rc)  else np.nan,
                "runtime": float(row[rtc].values[0]) if (not row.empty and rtc) else np.nan,
            }
    return out


def merge_year_files(path_2017, path_2018):
    """Merge separate _2017/_2018 summary files into one dict."""
    d17 = read_summary(path_2017)
    d18 = read_summary(path_2018)
    return {
        2017: d17[2017],
        2018: d18[2018],
    }


def get_rmse(fname, fname_17=None, fname_18=None):
    """Try single file, then split year files."""
    p = INTER_DIR / fname
    if p.exists():
        v = read_summary(p)
        if not (pd.isna(v[2017]["rmse"]) and pd.isna(v[2018]["rmse"])):
            return v
    if fname_17 and fname_18:
        p17 = INTER_DIR / fname_17
        p18 = INTER_DIR / fname_18
        if p17.exists() or p18.exists():
            return merge_year_files(p17, p18)
    return {2017: {"rmse": np.nan, "runtime": np.nan},
            2018: {"rmse": np.nan, "runtime": np.nan}}


def build_rows(table_name, entries):
    """
    entries: list of (group, method, {2017: {rmse, runtime}, 2018: {rmse, runtime}})
    """
    valid_17 = [v[2017]["rmse"] for _, _, v in entries if not pd.isna(v[2017]["rmse"])]
    valid_18 = [v[2018]["rmse"] for _, _, v in entries if not pd.isna(v[2018]["rmse"])]
    best_17 = min(valid_17) if valid_17 else np.nan
    best_18 = min(valid_18) if valid_18 else np.nan

    rows = []
    for group, method, vals in entries:
        r17 = vals[2017]["rmse"];    rt17 = vals[2017]["runtime"]
        r18 = vals[2018]["rmse"];    rt18 = vals[2018]["runtime"]
        rows.append({
            "table":            table_name,
            "group":            group,
            "method":           method,
            "rmse_2017":        round(r17, 4) if not pd.isna(r17) else np.nan,
            "gap_2017":         gap(r17, best_17),
            "rmse_2018":        round(r18, 4) if not pd.isna(r18) else np.nan,
            "gap_2018":         gap(r18, best_18),
            "runtime_2017_sec": round(rt17, 1) if not pd.isna(rt17) else np.nan,
            "runtime_2018_sec": round(rt18, 1) if not pd.isna(rt18) else np.nan,
        })
    return rows


# ─────────────────────────────────────────
# TABLE 1 — metric comparison (LOTO)
# ─────────────────────────────────────────

def build_table1():
    """
    Table1 shows LOTO performance under different matching metrics.
    Source: Table1_summary_all_metrics.csv (non-dfp)
    """
    all_path = INTER_DIR / "Table1_summary_all_metrics.csv"

    by_metric = {}
    if all_path.exists():
        df = pd.read_csv(all_path)
        metric_col = next((c for c in df.columns if "metric" in c.lower()), df.columns[0])
        by_metric  = read_summary_by_metric(all_path, metric_col)

    # fallback to individual files
    fallback = {
        "weighted_ks":          "Table1_weighted_ks_summary.csv",
        "mean_ks":              "Table1_mean_ks_summary.csv",
        "marginal_energy":      "Table1_marginal_energy_summary.csv",
        "sinkhorn_wasserstein": "Table1_sinkhorn_wasserstein_summary.csv",
    }
    for metric, fname in fallback.items():
        if metric not in by_metric:
            by_metric[metric] = read_summary(INTER_DIR / fname)

    label_map = {
        "weighted_ks":          "Weighted KS",
        "mean_ks":              "Mean KS Distance",
        "marginal_energy":      "Marginal Energy Distance",
        "sinkhorn_wasserstein": "Sinkhorn Wasserstein Distance",
        "d_WKS":                "Weighted KS",
        "d_KS":                 "Mean KS Distance",
        "d_KME":                "Marginal Energy Distance",
        "d_WKME":               "Sinkhorn Wasserstein Distance",
    }

    entries = []
    for metric, vals in by_metric.items():
        label = label_map.get(metric, metric)
        entries.append(("Similarity Metric", label, vals))

    if not entries:
        print("[WARN] No Table1 data found.")
        return []

    return build_rows("Table1_Metrics_LOTO", entries)


# ─────────────────────────────────────────
# TABLE 2 LOTO
# ─────────────────────────────────────────

def build_table2_loto():
    TF   = "Transfer learning (multi-source ensemble, K=7)"
    GEO  = "Geographic-neighbor transfer (K=7)"
    POOL = "Pooled training (all turbines)"

    # TF methods come from Table1 individual summary files (LOTO, non-dfp)
    # weighted_ks -> thinnedSV
    # mean_ks -> ANN
    # sinkhorn_wasserstein -> thinned twinGP
    tf_entries = [
        (TF, "thinnedSV",
         read_summary(INTER_DIR / "Table2_TF_thinnedSV_summary.csv")),
        (TF, "ANN",
         get_rmse("Table2_TF_ANN_summary.csv",
                  "Table2_TF_ANN_summary_2017.csv",
                  "Table2_TF_ANN_summary_2018.csv")),
        (TF, "LGBM",
         get_rmse("Table2_TF_LGBM_summary.csv")),
        (TF, "thinned twinGP",
         read_summary(INTER_DIR / "Table1_weighted_ks_summary.csv")),
    ]

    # Also try pulling from Table1 all-metrics summary as additional source
    all_loto = INTER_DIR / "Table1_summary_all_metrics.csv"
    if all_loto.exists():
        df = pd.read_csv(all_loto)
        metric_col = next((c for c in df.columns if "metric" in c.lower()), df.columns[0])
        tf_from_t1 = read_summary_by_metric(all_loto, metric_col)
        t1_map = {
            "weighted_ks":          "thinnedSV",
            "d_WKS":                "thinnedSV",
            "mean_ks":              "ANN",
            "d_KS":                 "ANN",
            "sinkhorn_wasserstein": "thinned twinGP",
            "d_WKME":               "thinned twinGP",
        }
        for metric, label in t1_map.items():
            if metric in tf_from_t1:
                # only use if not already populated from dedicated summary files
                existing = next((v for g, m, v in tf_entries if m == label), None)
                if existing is None or (pd.isna(existing[2017]["rmse"]) and pd.isna(existing[2018]["rmse"])):
                    # replace
                    tf_entries = [(g, m, v) for g, m, v in tf_entries if m != label]
                    tf_entries.append((TF, label, tf_from_t1[metric]))

    entries = tf_entries + [
        # Geographic
        (GEO, "XGBoost",
         get_rmse("Table2_G_XGBoost_summary.csv",
                  "Table2_G_XGBoost_summary_2017.csv",
                  "Table2_G_XGBoost_summary_2018.csv")),
        (GEO, "Random Forest",
         get_rmse("Table2_G_random_forest_summary.csv",
                  "Table2_G_random_forest_summary_2017.csv",
                  "Table2_G_random_forest_summary_2018.csv")),
        (GEO, "SVR",
         get_rmse("Table2_G_SVR_summary.csv",
                  "Table2_G_SVR_summary_2017.csv",
                  "Table2_G_SVR_summary_2018.csv")),
        # Pooled
        (POOL, "XGBoost",
         get_rmse("Table2_P_XGBoost_summary.csv",
                  "Table2_P_XGBoost_summary_2017.csv",
                  "Table2_P_XGBoost_summary_2018.csv")),
        (POOL, "GNN",
         get_rmse("Table2_P_GNN_summary.csv",
                  "Table2_P_GNN_loo_summary.csv")),
        (POOL, "BHM",     {2017: {"rmse": np.nan, "runtime": np.nan},
                           2018: {"rmse": np.nan, "runtime": np.nan}}),
        (POOL, "twinGP",
         get_rmse("Table2_P_twinGP_summary.csv",
                  "Table2_P_twinGP_summary_2017.csv",
                  "Table2_P_twinGP_summary_2018.csv")),
        (POOL, "Binning",
         get_rmse("Table2_P_Binning_summary.csv",
                  "Table2_P_Binning_summary_2017.csv",
                  "Table2_P_Binning_summary_2018.csv")),
    ]

    entries = [(g, m, v) for g, m, v in entries
               if not (pd.isna(v[2017]["rmse"]) and pd.isna(v[2018]["rmse"]))]

    if not entries:
        print("[WARN] No Table2 LOTO data found.")
        return []

    return build_rows("Table2_LOTO", entries)


# ─────────────────────────────────────────
# TABLE 2 DFP
# ─────────────────────────────────────────

def build_table2_dfp():
    TF   = "Transfer learning (multi-source ensemble, K=7)"
    GEO  = "Geographic-neighbor transfer (K=7)"
    POOL = "Pooled training (all turbines)"

    entries = [
        # TF — from dedicated summary files
        (TF, "thinnedSV",
         get_rmse("Table2_TF_thinnedSV_summary_dfp.csv")),
        (TF, "ANN",
         get_rmse("Table2_TF_ANN_summary_dfp.csv")),
        (TF, "LGBM",
         get_rmse("Table2_TF_LGBM_summary_dfp.csv")),
        (TF, "thinned twinGP",
         read_summary(INTER_DIR / "Table1_weighted_ks_summary_dfp.csv")),
        # Geographic
        (GEO, "XGBoost",
         get_rmse("Table2_G_XGBoost_summary_dfp.csv")),
        (GEO, "Random Forest",
         get_rmse("Table2_G_random_forest_summary_dfp.csv")),
        (GEO, "SVR",
         get_rmse("Table2_G_SVR_summary_dfp.csv")),
        # Pooled
        (POOL, "XGBoost",
         get_rmse("Table2_P_XGBoost_summary_dfp.csv")),
        (POOL, "GNN",
         get_rmse("Table2_P_GNN_dfp_summary.csv")),
        (POOL, "BHM",    {2017: {"rmse": np.nan, "runtime": np.nan},
                          2018: {"rmse": np.nan, "runtime": np.nan}}),
        (POOL, "twinGP",
         get_rmse("Table2_P_twinGP_summary_dfp.csv")),
        (POOL, "Binning",
         get_rmse("Table2_P_Binning_summary_dfp.csv")),
    ]

    entries = [(g, m, v) for g, m, v in entries
               if not (pd.isna(v[2017]["rmse"]) and pd.isna(v[2018]["rmse"]))]

    if not entries:
        print("[WARN] No Table2 DFP data found.")
        return []

    return build_rows("Table2_DFP", entries)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    all_rows = []
    all_rows += build_table1()
    all_rows += build_table2_loto()
    all_rows += build_table2_dfp()

    if not all_rows:
        print("[ERROR] No data found. Check intermediate CSVs in:", INTER_DIR)
        return

    df = pd.DataFrame(all_rows, columns=[
        "table", "group", "method",
        "rmse_2017", "gap_2017",
        "rmse_2018", "gap_2018",
        "runtime_2017_sec", "runtime_2018_sec",
    ])

    df.to_csv(OUT_FILE, index=False)
    print(f"[DONE] Saved {len(df)} rows to {OUT_FILE}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()