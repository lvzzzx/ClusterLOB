
import sys
import os
import argparse
import glob
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from clusterlob.pipeline import run_pipeline

def _build_summary_if_missing(run_dir: str, horizons: list[int]) -> str | None:
    summary_dir = os.path.join(run_dir, "summary")
    summary_path = os.path.join(summary_dir, "summary_all.parquet")
    if os.path.exists(summary_path):
        return summary_path

    window_paths = sorted(glob.glob(os.path.join(run_dir, "bucket_ofi_returns_window_*.parquet")))
    if not window_paths:
        return None

    df = pl.concat([pl.read_parquet(p) for p in window_paths], how="vertical")

    summaries = []
    for h in horizons:
        r_col = f"r_{h}"
        if r_col not in df.columns:
            continue
        df_h = df.filter(pl.col(r_col).is_not_null())
        if df_h.height == 0:
            continue

        grouped = df_h.group_by(
            [
                "cluster",
                "window_id",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "bucket_us",
                "k",
            ]
        ).agg(
            [
                pl.mean("ofi_size").alias("mean_ofi_size"),
                pl.std("ofi_size").alias("std_ofi_size"),
                pl.mean("ofi_count").alias("mean_ofi_count"),
                pl.std("ofi_count").alias("std_ofi_count"),
                pl.mean(r_col).alias("mean_r"),
                pl.std(r_col).alias("std_r"),
                pl.corr("ofi_size", r_col).alias("ic_size"),
                pl.corr("ofi_count", r_col).alias("ic_count"),
                pl.len().alias("n"),
            ]
        ).with_columns(pl.lit(h).alias("horizon"))

        summaries.append(grouped)

    if not summaries:
        return None

    summary_all = pl.concat(summaries, how="vertical")
    os.makedirs(summary_dir, exist_ok=True)
    summary_all.write_parquet(summary_path)
    return summary_path

def run_experiment_matrix(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
    train_days: int,
    test_days: int,
    step_days: int,
    base_out_dir: str,
    skip_pipeline: bool,
):
    # --- Experiment Configuration ---
    # Bucket Sizes in Microseconds
    bucket_configs = {
        "1s": 1_000_000,
        "5s": 5_000_000,
        "15s": 15_000_000,
        "30s": 30_000_000,
        "60s": 60_000_000
    }
    
    # Forward Horizons (Multiples of Bucket Size)
    horizons = [1, 2, 5, 10, 20]
    
    results = []

    print(f"--- Starting Timeframe Matrix Experiment ---")
    print(f"Date range: {start_date} -> {end_date}")
    print(f"Walk-forward: train={train_days}d, test={test_days}d, step={step_days}d")
    
    for label, bucket_us in bucket_configs.items():
        print(f"\n>> Running Bucket Size: {label} ({bucket_us} us)")
        
        # Define output directory for this specific run
        run_dir = os.path.join(base_out_dir, f"bucket_{label}")
        
        # 1. Run Pipeline (rolling walk-forward) unless skipped
        if not skip_pipeline:
            # using the refactored pipeline with rule-based alignment
            try:
                run_pipeline(
                    exchange=exchange,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    window=100,
                    k=3,
                    bucket_us=bucket_us,
                    depth=5,
                    horizons=horizons,
                    train_days=train_days,
                    test_days=test_days,
                    step_days=step_days,
                    out_dir=run_dir
                )
            except Exception as e:
                print(f"Pipeline failed for {label}: {e}")
                continue

        # 2. Analyze Results (fast path: summary_all.parquet)
        summary_path = _build_summary_if_missing(run_dir, horizons)
        if summary_path is None:
            print(f"Summary not found: {run_dir}")
            continue

        summary = pl.read_parquet(summary_path)

        # Align to requested horizons only
        summary = summary.filter(pl.col("horizon").is_in(horizons))

        if summary.height == 0:
            print("No matching summary rows found.")
            continue

        # Aggregate across all rolling windows in the range with stability metrics
        summary = summary.group_by(["cluster", "horizon"]).agg(
            [
                pl.mean("ic_size").alias("mean_ic_size"),
                pl.std("ic_size").alias("std_ic_size"),
                (pl.col("ic_size") > 0).mean().alias("pos_rate_size"),
                pl.mean("ic_count").alias("mean_ic_count"),
                pl.std("ic_count").alias("std_ic_count"),
                (pl.col("ic_count") > 0).mean().alias("pos_rate_count"),
                pl.len().alias("n_windows"),
                pl.mean("n").alias("mean_n"),
            ]
        ).with_columns(
            [
                pl.when(pl.col("std_ic_size") > 0)
                .then(pl.col("mean_ic_size") / (pl.col("std_ic_size") / pl.col("n_windows").sqrt()))
                .otherwise(None)
                .alias("tstat_ic_size"),
                pl.when(pl.col("std_ic_count") > 0)
                .then(pl.col("mean_ic_count") / (pl.col("std_ic_count") / pl.col("n_windows").sqrt()))
                .otherwise(None)
                .alias("tstat_ic_count"),
            ]
        )

        # Map cluster labels to signal names
        cluster_map = {
            -1: "ofi_all",
            0: "ofi_passive",
            1: "ofi_opp",
            2: "ofi_toxic",
        }

        for row in summary.iter_rows(named=True):
            signal = cluster_map.get(row["cluster"])
            if signal is None:
                continue

            for metric, ic_col in [("size", "mean_ic_size"), ("count", "mean_ic_count")]:
                std_col = "std_ic_size" if metric == "size" else "std_ic_count"
                tstat_col = "tstat_ic_size" if metric == "size" else "tstat_ic_count"
                pos_col = "pos_rate_size" if metric == "size" else "pos_rate_count"
                results.append({
                    "bucket_label": label,
                    "bucket_us": bucket_us,
                    "horizon_mult": int(row["horizon"]),
                    "horizon_time_s": (bucket_us * int(row["horizon"])) / 1e6,
                    "signal": f"{signal}_{metric}",
                    "mean_ic": row[ic_col],
                    "std_ic": row[std_col],
                    "tstat_ic": row[tstat_col],
                    "pos_rate": row[pos_col],
                    "n_windows": row["n_windows"],
                    "mean_n": row["mean_n"],
                })
                
    # --- Output Summary ---
    if not results:
        print("No results generated.")
        return

    res_df = pd.DataFrame(results)
    
    # Save Raw Data
    os.makedirs(base_out_dir, exist_ok=True)
    res_df.to_csv(os.path.join(base_out_dir, "ic_matrix.csv"), index=False)
    
    print("\n--- IC Matrix Summary (Top 5) ---")
    print(res_df.sort_values("mean_ic", ascending=False).head(5))
    
    # --- Visualization ---
    plot_matrix(res_df, base_out_dir)

def plot_matrix(df, out_dir):
    # We want a Heatmap: X=Horizon(Time), Y=BucketSize, Color=IC
    # One heatmap per Signal Type
    
    signals = df['signal'].unique()
    
    for sig in signals:
        plt.figure(figsize=(10, 6))
        
        # Pivot for Heatmap
        # We use 'horizon_time_s' for a unified X-axis
        pivot = df[df['signal'] == sig].pivot(
            index="bucket_label", 
            columns="horizon_mult", 
            values="ic"
        )
        
        # Reorder Y-axis (Bucket Size) logically
        order = ["1s", "5s", "15s", "30s", "60s"]
        pivot = pivot.reindex([o for o in order if o in pivot.index])
        
        sns.heatmap(pivot, annot=True, cmap="coolwarm", center=0, fmt=".3f")
        plt.title(f"Information Coefficient (IC) Heatmap: {sig}")
        plt.ylabel("Bucket Aggregation Size")
        plt.xlabel("Prediction Horizon (Multiples of Bucket)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heatmap_ic_{sig}.png"))
        plt.close()
        
    print(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--symbol", default="BTCUSDT")
    # Default to a 2-day sample from our known dataset
    parser.add_argument("--start-date", default="2024-05-01")
    parser.add_argument("--end-date", default="2024-05-31")
    parser.add_argument("--train-days", type=int, default=7)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument("--step-days", type=int, default=1)
    parser.add_argument("--out-dir", default="outputs/timeframe_matrix")
    parser.add_argument("--skip-pipeline", action="store_true", help="skip run_pipeline; summarize existing outputs")
    args = parser.parse_args()
    
    run_experiment_matrix(
        args.exchange,
        args.symbol,
        args.start_date,
        args.end_date,
        args.train_days,
        args.test_days,
        args.step_days,
        args.out_dir,
        args.skip_pipeline,
    )
