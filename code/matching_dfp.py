from pathlib import Path
import sys
import asyncio
import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

TEST_IDS = list(range(38, 45)) + list(range(61, 67))
ALL_IDS = list(range(1, 67))
TRAIN_IDS = sorted(set(ALL_IDS) - set(TEST_IDS))
YEAR = 2017


def get_processed_dir() -> Path:
    """Resolve the project's processed_data directory from common launch locations."""
    candidates = []

    cwd = Path.cwd().resolve()
    candidates.append(cwd / "data" / "processed_data")
    candidates.append(cwd.parent / "data" / "processed_data")
    candidates.append(Path(__file__).resolve().parent.parent / "data" / "processed_data")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: create under the most likely project root.
    fallback = cwd / "data" / "processed_data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


PROCESSED_DIR = get_processed_dir()


def normalize_rank_col_name(name: str) -> int:
    s = name.replace("_", "")
    return int(s.replace("donor", "").replace("score", ""))



def filter_pairwise_matching():
    infile = PROCESSED_DIR / f"pairwise_matching_{YEAR}.csv"
    outfile = PROCESSED_DIR / f"pairwise_matching_{YEAR}_dfp.csv"

    df = pd.read_csv(infile)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["donor"] = pd.to_numeric(df["donor"], errors="coerce")
    df = df.dropna(subset=["target", "donor"]).copy()
    df["target"] = df["target"].astype(int)
    df["donor"] = df["donor"].astype(int)

    df = df[df["target"].isin(TEST_IDS) & df["donor"].isin(TRAIN_IDS)].copy()
    df = df.sort_values(["target", "donor"]).reset_index(drop=True)
    df.to_csv(outfile, index=False)
    return outfile.name, df



def filter_pairwise_geo():
    infile = PROCESSED_DIR / "pairwise_geographic_distance.csv"
    outfile = PROCESSED_DIR / "pairwise_geographic_distance_dfp.csv"

    df = pd.read_csv(infile)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["donor"] = pd.to_numeric(df["donor"], errors="coerce")
    df["geo_distance"] = pd.to_numeric(df["geo_distance"], errors="coerce")
    df = df.dropna(subset=["target", "donor", "geo_distance"]).copy()
    df["target"] = df["target"].astype(int)
    df["donor"] = df["donor"].astype(int)

    df = df[df["target"].isin(TEST_IDS) & df["donor"].isin(TRAIN_IDS)].copy()
    df = df.sort_values(["target", "geo_distance", "donor"]).reset_index(drop=True)
    df.to_csv(outfile, index=False)
    return outfile.name, df



def filter_ranked_file(src_name: str, dst_name: str):
    src = PROCESSED_DIR / src_name
    dst = PROCESSED_DIR / dst_name

    df = pd.read_csv(src)
    if "target" not in df.columns:
        df = df.rename(columns={df.columns[0]: "target"})

    donor_cols = [c for c in df.columns if c.startswith("donor")]
    score_cols = [c for c in df.columns if c.startswith("score")]

    donor_cols = sorted(donor_cols, key=normalize_rank_col_name)
    score_cols = sorted(score_cols, key=normalize_rank_col_name)

    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    df = df[df["target"].isin(TEST_IDS)].copy()

    out_rows = []
    for _, row in df.iterrows():
        kept_donors = []
        kept_scores = []

        for i, dcol in enumerate(donor_cols):
            donor_val = pd.to_numeric(pd.Series([row[dcol]]), errors="coerce").iloc[0]
            if pd.isna(donor_val):
                continue
            donor_val = int(donor_val)

            if donor_val not in TRAIN_IDS:
                continue

            kept_donors.append(donor_val)

            if i < len(score_cols):
                sval = pd.to_numeric(pd.Series([row[score_cols[i]]]), errors="coerce").iloc[0]
                kept_scores.append(np.nan if pd.isna(sval) else float(sval))

        new_row = {"target": int(row["target"])}
        for j, donor in enumerate(kept_donors, start=1):
            new_row[f"donor{j}"] = donor
            if j <= len(kept_scores):
                new_row[f"score{j}"] = kept_scores[j - 1]

        out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows).sort_values("target").reset_index(drop=True)
    out_df.to_csv(dst, index=False)
    return dst.name, out_df



def main():
    print("Processed dir:", PROCESSED_DIR)
    print("Test IDs:", TEST_IDS)
    print("Train IDs:", TRAIN_IDS)

    created = []

    name, _ = filter_pairwise_matching()
    created.append(name)

    name, _ = filter_pairwise_geo()
    created.append(name)

    ranked_specs = [
        ("matching_weighted_ks.csv", "matching_weighted_ks_dfp.csv"),
        ("matching_mean_ks.csv", "matching_mean_ks_dfp.csv"),
        ("matching_marginal_energy.csv", "matching_marginal_energy_dfp.csv"),
        ("matching_sinkhorn_wasserstein.csv", "matching_sinkhorn_wasserstein_dfp.csv"),
        ("matching_geographic_distance.csv", "matching_geographic_distance_dfp.csv"),
    ]

    for src, dst in ranked_specs:
        name, _ = filter_ranked_file(src, dst)
        created.append(name)

    summary = pd.DataFrame({"file": created})
    summary_path = PROCESSED_DIR / "matching_files_created_dfp.csv"
    summary.to_csv(summary_path, index=False)

    print("Created files:")
    for f in created:
        print(" -", f)
    print(" -", summary_path.name)


if __name__ == "__main__":
    main()
