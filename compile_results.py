"""
compile_results.py
====================
Reads intermediate summary CSVs and produces results/final_results.csv,
formatted to match the three tables in the paper:

  Table 2  (LOTO metric comparison: WD, Sinkhorn, Mean, Marginal Energy)
  Table 3  (LOTO method comparison: WD Transfer / Pooled / Geographic groups)
  Table 4  (DFP: WD Transfer / Geographic groups, 2018 RMSE only, no Pooled group)

Methods without code in this repository (BHM, STGP) are filled in with
hardcoded values taken from the paper / external runs -- see HARDCODED_VALUES
below. Edit those constants if the underlying numbers change.

Row order within each table/group matches the paper exactly.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT      = Path(__file__).resolve().parent
INTER_DIR = ROOT / "results" / "intermediate"
OUT_FILE  = ROOT / "results" / "final_results.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# Hardcoded values for methods with no code in this repo
# (BHM: external repo, too slow to reproduce here -- see README.
#  STGP: results from a separate paper/repo, included for comparison only.)
# Edit these if the source numbers are updated.
# ─────────────────────────────────────────

HARDCODED_VALUES = {
    # (table, group, method): {2017: {"rmse":..., "runtime": np.nan}, 2018: {...}}
    ("Table3_LOTO", "Pooled", "STGP"): {
        2017: {"rmse": 3.73, "runtime": np.nan},
        2018: {"rmse": 3.97, "runtime": np.nan},
    },
    ("Table3_LOTO", "Pooled", "BHM"): {
        2017: {"rmse": 4.03, "runtime": np.nan},
        2018: {"rmse": 4.15, "runtime": np.nan},
    },
}


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
    rc = rmse_col if rmse_col in df.columns else next(
        (c for c in df.columns if "rmse" in c.lower()), None)
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


def get_rmse(fname):
    """Read a single summary CSV from INTER_DIR."""
    return read_summary(INTER_DIR / fname)


def get_hardcoded(table, group, method):
    return HARDCODED_VALUES.get((table, group, method), {
        2017: {"rmse": np.nan, "runtime": np.nan},
        2018: {"rmse": np.nan, "runtime": np.nan},
    })


def build_rows(table_name, entries, best_basis=None, years=(2017, 2018)):
    """
    entries: ordered list of (group, method, {2017: {rmse, runtime}, 2018: {rmse, runtime}})
    best_basis: optional dict {2017: best_rmse, 2018: best_rmse} to compute gaps against.
                If None, gaps are computed against the min RMSE among `entries` themselves.
    years: which years to emit columns for (e.g. (2018,) for DFP table)
    """
    if best_basis is None:
        valid = {yr: [v[yr]["rmse"] for _, _, v in entries if not pd.isna(v[yr]["rmse"])]
                 for yr in years}
        best_basis = {yr: (min(valid[yr]) if valid[yr] else np.nan) for yr in years}

    rows = []
    for group, method, vals in entries:
        row = {"table": table_name, "group": group, "method": method}
        for yr in years:
            r   = vals[yr]["rmse"]
            rt  = vals[yr]["runtime"]
            row[f"rmse_{yr}"]        = round(r, 4) if not pd.isna(r) else np.nan
            row[f"gap_{yr}"]         = gap(r, best_basis[yr])
            row[f"runtime_{yr}_sec"] = round(rt, 1) if not pd.isna(rt) else np.nan
        rows.append(row)
    return rows, best_basis


# ─────────────────────────────────────────
# TABLE 2 — metric comparison (LOTO)
# Order: Weighted Dissimilarity (WD), Sinkhorn Wasserstein, Mean Dissimilarity, Marginal Energy
# ─────────────────────────────────────────

def build_table2_metrics():
    label_map_order = [
        ("weighted_ks",          "Weighted Dissimilarity(WD)"),
        ("sinkhorn_wasserstein", "Sinkhorn Wasserstein Distance"),
        ("mean_ks",              "Mean Dissimilarity(unweighted)"),
        ("marginal_energy",      "Marginal Energy Distance"),
    ]

    all_path = INTER_DIR / "Table1_summary_all_metrics.csv"
    by_metric = {}
    if all_path.exists():
        df = pd.read_csv(all_path)
        metric_col = next((c for c in df.columns if "metric" in c.lower()), df.columns[0])
        by_metric  = read_summary_by_metric(all_path, metric_col)

    fallback = {
        "weighted_ks":          "Table1_weighted_ks_summary.csv",
        "mean_ks":              "Table1_mean_ks_summary.csv",
        "marginal_energy":      "Table1_marginal_energy_summary.csv",
        "sinkhorn_wasserstein": "Table1_sinkhorn_wasserstein_summary.csv",
    }
    for metric, fname in fallback.items():
        if metric not in by_metric or (
            pd.isna(by_metric[metric][2017]["rmse"]) and pd.isna(by_metric[metric][2018]["rmse"])
        ):
            by_metric[metric] = get_rmse(fname)

    entries = []
    for metric_key, label in label_map_order:
        vals = by_metric.get(metric_key, {
            2017: {"rmse": np.nan, "runtime": np.nan},
            2018: {"rmse": np.nan, "runtime": np.nan},
        })
        entries.append(("Similarity Metric", label, vals))

    rows, _ = build_rows("Table2_Metrics_LOTO", entries)
    return rows


# ─────────────────────────────────────────
# TABLE 3 — LOTO method comparison
# Order within each group matches the paper exactly.
# ─────────────────────────────────────────

def build_table3_loto():
    WD   = "The proposed WD Transfer (K=7)"
    POOL = "Pooled training (all turbines)"
    GEO  = "Geographic-neighbor transfer (K=7)"

    # --- WD Transfer group: thinnedSV, ANN, XGBoost, thinned twinGP ---
    wd_entries = [
        (WD, "thinnedSV",
         get_rmse("Table2_TF_thinnedSV_summary.csv")),
        (WD, "ANN",
         get_rmse("Table2_TF_ANN_summary.csv")),
        (WD, "XGBoost",
         get_rmse("Table2_TF_XGBoost_summary.csv")),
        (WD, "thinned twinGP",
         get_rmse("Table1_weighted_ks_summary.csv")),
    ]

    # --- Pooled group: STGP, ANN, thinned twinGP, XGBoost, GNN, BHM, Binning ---
    pool_entries = [
        (POOL, "STGP",
         get_hardcoded("Table3_LOTO", "Pooled", "STGP")),
        (POOL, "ANN",
         get_rmse("Table2_P_ANN_summary.csv")),
        (POOL, "thinned twinGP",
         get_rmse("Table2_P_thinned_twinGP_summary.csv")),
        (POOL, "XGBoost",
         get_rmse("Table2_P_XGBoost_summary.csv")),
        (POOL, "GNN",
         get_rmse("Table2_P_GNN_loo_summary.csv")),
        (POOL, "BHM",
         get_hardcoded("Table3_LOTO", "Pooled", "BHM")),
        (POOL, "Binning",
         get_rmse("Table2_P_Binning_summary.csv")),
    ]

    # --- Geographic group: ANN, thinned twinGP, Random Forest, XGBoost, SVR ---
    geo_entries = [
        (GEO, "ANN",
         get_rmse("Table2_G_ANN_summary.csv")),
        (GEO, "thinned twinGP",
         get_rmse("Table2_G_thinned_twinGP_summary.csv")),
        (GEO, "Random Forest",
         get_rmse("Table2_G_random_forest_summary.csv")),
        (GEO, "XGBoost",
         get_rmse("Table2_G_XGBoost_summary.csv")),
        (GEO, "SVR",
         get_rmse("Table2_G_SVR_summary.csv")),
    ]

    all_entries = wd_entries + pool_entries + geo_entries

    # Gap is computed relative to the best method in the WD Transfer group only
    # (per the paper's table caption), not the global best across all groups.
    wd_valid_17 = [v[2017]["rmse"] for _, _, v in wd_entries if not pd.isna(v[2017]["rmse"])]
    wd_valid_18 = [v[2018]["rmse"] for _, _, v in wd_entries if not pd.isna(v[2018]["rmse"])]
    best_basis = {
        2017: min(wd_valid_17) if wd_valid_17 else np.nan,
        2018: min(wd_valid_18) if wd_valid_18 else np.nan,
    }

    rows, _ = build_rows("Table3_LOTO", all_entries, best_basis=best_basis)
    return rows


# ─────────────────────────────────────────
# TABLE 4 — DFP (2018 RMSE only; WD Transfer and Geographic groups only)
# Order within each group matches the paper exactly.
# ─────────────────────────────────────────

def build_table4_dfp():
    WD  = "The proposed WD transfer (K=7)"
    GEO = "Geographic-neighbor transfer (K=7)"

    wd_entries = [
        (WD, "thinnedSV",
         get_rmse("Table2_TF_thinnedSV_summary_dfp.csv")),
        (WD, "ANN",
         get_rmse("Table2_TF_ANN_summary_dfp.csv")),
        (WD, "thinned twinGP",
         get_rmse("Table1_weighted_ks_summary_dfp.csv")),
        (WD, "XGBoost",
         get_rmse("Table2_TF_XGBoost_summary_dfp.csv")),
    ]

    geo_entries = [
        (GEO, "thinned twinGP",
         get_rmse("Table2_G_thinned_twinGP_dfp_summary.csv")),
        (GEO, "Random Forest",
         get_rmse("Table2_G_random_forest_summary_dfp.csv")),
        (GEO, "ANN",
         get_rmse("Table2_G_ANN_summary_dfp.csv")),
        (GEO, "XGBoost",
         get_rmse("Table2_G_XGBoost_summary_dfp.csv")),
        (GEO, "SVR",
         get_rmse("Table2_G_SVR_summary_dfp.csv")),
    ]

    all_entries = wd_entries + geo_entries

    # Gap relative to best method in WD transfer group, 2018 only.
    wd_valid_18 = [v[2018]["rmse"] for _, _, v in wd_entries if not pd.isna(v[2018]["rmse"])]
    best_basis = {2018: min(wd_valid_18) if wd_valid_18 else np.nan}

    rows, _ = build_rows("Table4_DFP", all_entries, best_basis=best_basis, years=(2018,))
    return rows


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    all_rows = []
    all_rows += build_table2_metrics()
    all_rows += build_table3_loto()
    all_rows += build_table4_dfp()

    if not all_rows:
        print("[ERROR] No data found. Check intermediate CSVs in:", INTER_DIR)
        return

    df = pd.DataFrame(all_rows)
    front = ["table", "group", "method"]
    df = df[front + [c for c in df.columns if c not in front]]

    # Save one combined file (with NaN columns where a table doesn't use that year)
    df.to_csv(OUT_FILE, index=False)
    print(f"[DONE] Saved {len(df)} rows to {OUT_FILE}\n")

    # Also save one clean CSV per table, with unused all-NaN columns dropped
    # (e.g. Table4_DFP has no 2017 columns since it's 2018-only)
    for table_name, sub in df.groupby("table", sort=False):
        sub_clean = sub.dropna(axis=1, how="all")
        table_path = OUT_FILE.parent / f"final_results_{table_name}.csv"
        sub_clean.to_csv(table_path, index=False)
        print(f"\n=== {table_name} (saved to {table_path.name}) ===")
        print(sub_clean.to_string(index=False))


if __name__ == "__main__":
    main()
