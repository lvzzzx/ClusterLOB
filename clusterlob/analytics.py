
import polars as pl
import os
import matplotlib.pyplot as plt
import argparse

def analyze_cluster_centroids(data_path: str):
    """
    Computes and prints the mean values of features for each cluster.
    """
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        return

    df = pl.read_parquet(data_path)
    print(f"Total trades: {df.height}")
    
    feature_cols = [
        "z_v_rel",
        "z_sbs",
        "z_obs",
        "z_spread",
        "z_t_m",
        "z_t_age",
    ]
    
    stats = df.group_by("cluster").agg(
        [
            pl.len().alias("count"),
            *[pl.col(c).mean().alias(f"mean_{c}") for c in feature_cols]
        ]
    ).sort("cluster")
    
    print("\nCluster Centroids (Means):")
    print(stats)
    return stats

def compute_ic_pnl(data_path: str, out_dir: str):
    """
    Computes Information Coefficient (IC) and runs a naive PnL simulation.
    """
    print(f"Loading data from {data_path}...")
    df = pl.read_parquet(data_path)
    
    # Filter for valid FRNB
    df = df.filter(pl.col("FRNB").is_not_null())
    
    print(f"Data loaded. Buckets with valid returns: {df.select('bucket_ts').n_unique()}")

    # Pivot to wide format
    # Using 'on' if available or handling pivot manually if needed, but keeping simple for now
    # Note: polars pivot signature changes. In recent versions `on` is preferred.
    # The user script used `index`, `columns`, `values` which is standard.
    # We'll use the syntax that worked in the last successful run (validate_alpha.py).
    
    # Polars > 0.20 uses 'on' instead of 'columns' and 'index'
    try:
        pivoted = df.pivot(
            values=["ofi_s", "ofi_c"],
            index=["date", "bucket_ts", "FRNB"],
            columns="cluster",
            aggregate_function="sum"
        ).fill_null(0)
    except TypeError:
         # Fallback for newer polars if the above fails (though previous run passed with DeprecationWarning)
         pivoted = df.pivot(
            values=["ofi_s", "ofi_c"],
            index=["date", "bucket_ts", "FRNB"],
            on="cluster",
            aggregate_function="sum"
        ).fill_null(0)
    
    pivoted = pivoted.sort("bucket_ts")
    
    # Define cluster columns
    # We dynamically find them to be safe
    ofi_s_cols = [c for c in pivoted.columns if c.startswith("ofi_s_")]
    ofi_c_cols = [c for c in pivoted.columns if c.startswith("ofi_c_")]
    
    # Calculate Benchmark OFI
    pivoted = pivoted.with_columns(
        [
            sum([pl.col(c) for c in ofi_s_cols]).alias("ofi_s_benchmark"),
            sum([pl.col(c) for c in ofi_c_cols]).alias("ofi_c_benchmark"),
        ]
    )
    
    # --- IC ---
    print("\n--- Information Coefficient (IC) ---")
    ic_results = {}
    target = "FRNB"
    
    feature_sets = {
        "OFI (Size)": ofi_s_cols + ["ofi_s_benchmark"],
        "OFI (Count)": ofi_c_cols + ["ofi_c_benchmark"]
    }
    
    for name, cols in feature_sets.items():
        print(f"\n{name} Correlation with {target}:")
        corrs = pivoted.select(
            [pl.corr(c, target).alias(c) for c in cols]
        ).to_dicts()[0]
        
        for k, v in corrs.items():
            print(f"  {k}: {v:.4f}")
            ic_results[k] = v

    # --- PnL ---
    print("\n--- PnL Simulation (Naive) ---")
    pnl_df = pivoted.select(["bucket_ts", "FRNB"])
    
    for c in ofi_s_cols + ["ofi_s_benchmark"]:
        pnl_col_name = f"pnl_{c}"
        pnl_df = pnl_df.with_columns(
            (pl.lit(pivoted[c]).sign() * pl.col("FRNB")).alias(pnl_col_name)
        )
    
    # Cumulative PnL
    cum_pnl = pnl_df.select(
        [
            pl.col("bucket_ts")] + 
            [pl.col(c).cum_sum().alias(f"cum_{c}") for c in pnl_df.columns if c.startswith("pnl_")]
    )
    
    pdf = cum_pnl.to_pandas()
    
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    for c in pdf.columns:
        if c.startswith("cum_pnl_ofi_s_"):
            label = c.replace("cum_pnl_ofi_s_", "Cluster ")
            color = 'black' if "benchmark" in label else None
            linestyle = '--' if "benchmark" in label else '-'
            linewidth = 2 if "benchmark" in label else 1.5
            
            plt.plot(pdf["bucket_ts"], pdf[c], label=label, linestyle=linestyle, linewidth=linewidth, color=color)
            
    plt.title("Cumulative PnL: Cluster OFI vs Future Returns (FRNB)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Log Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "cumulative_pnl.png")
    plt.savefig(plot_path)
    print(f"PnL plot saved to {plot_path}")
    
    with open(os.path.join(out_dir, "ic_stats.txt"), "w") as f:
        f.write("Information Coefficient (Correlation with FRNB):\n")
        for k, v in ic_results.items():
            f.write(f"{k}: {v:.6f}\n")
