
import polars as pl
import numpy as np
import argparse
import os

def run_resampled_backtest(
    data_path: str,
    out_dir: str,
    resample_minutes: int = 15,
    cost_bps: float = 3.0,
    threshold_quantile: float = 0.75,
    rolling_window_days: int = 1
):
    os.makedirs(out_dir, exist_ok=True)
    mpl_cache_dir = os.path.join(out_dir, ".mpl_cache")
    os.makedirs(mpl_cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    print(f"Loading 5-min data from {data_path}...")
    df = pl.read_parquet(data_path)
    
    # 1. Resample to Target Timeframe
    # Convert Int64 bucket_ts (microsecond timestamp?) or similar. 
    # Let's inspect bucket_ts. It seems to be seconds or ms based on prev output? 
    # Wait, usually bucket_ts is nanoseconds in Polars if datetime, but here Int64.
    # Assuming standard UNIX timestamp (seconds) or similar.
    # Actually, let's just use the 'date' and bucket_ts to reconstruct datetime if needed, 
    # but simplest is to just modulo the bucket_ts if it is unix epoch.
    
    # Check if bucket_ts is actually a datetime castable int.
    # In experiments/02... we saw it used directly.
    # Let's assume bucket_ts is UNIX SECONDS for now, or use Polars truncate.
    
    # We need to ensure we align the Pivot first before resampling?
    # No, Pivot first is easier, then resample the wide dataframe.
    
    print("Pivoting by Cluster...")
    df = df.with_columns(pl.col("cluster").cast(pl.Utf8))
    pivoted = df.pivot(
        values="ofi_s",
        index=["bucket_ts"],
        on="cluster",
        aggregate_function="sum"
    ).fill_null(0).sort("bucket_ts")
    
    # Get Returns for the same buckets
    returns_df = df.group_by("bucket_ts").agg(
        pl.col("FRNB").mean().alias("FRNB") # Mean is correct because it's the SAME bucket return for all clusters
    ).sort("bucket_ts")
    
    joined = pivoted.join(returns_df, on="bucket_ts")
    
    # Convert bucket_ts to Datetime for resampling
    # Assuming bucket_ts is Unix Seconds (standard for crypto data usually)
    # Let's verify by checking the difference between first two rows.
    # If 300, it's seconds. If 300000, ms.
    
    ts_diff = joined["bucket_ts"][1] - joined["bucket_ts"][0]
    if ts_diff == 300:
        time_unit = "s"
    elif ts_diff == 300000:
        time_unit = "ms"
    elif ts_diff == 300000000:
        time_unit = "us"
    else:
        # Fallback, assume seconds if unknown
        print(f"Warning: Unknown timestamp scale. Diff={ts_diff}. Assuming Seconds.")
        time_unit = "s"
        
    print(f"Detected Time Unit: {time_unit}")
    
    joined = joined.with_columns(
        pl.from_epoch(pl.col("bucket_ts"), time_unit=time_unit).alias("datetime")
    ).sort("datetime")
    
    print(f"Resampling to {resample_minutes} minutes...")
    # Resample
    # OFI sums, Returns sum (Log returns)
    resampled = (
        joined.group_by_dynamic("datetime", every=f"{resample_minutes}m")
        .agg([
            pl.col("0").sum(),
            pl.col("1").sum(),
            pl.col("2").sum(),
            pl.col("FRNB").sum()
        ])
    )
    
    # 2. Strategy Logic (Same as before)
    # Cluster 1 = Alpha
    signal_series = resampled["1"].cast(pl.Float64).to_numpy()
    frnb_series = resampled["FRNB"].cast(pl.Float64).to_numpy()
    
    # Rolling Window Calculation (buckets in window)
    # 1 Day = 1440 mins.
    buckets_per_day = int(1440 / resample_minutes)
    rolling_window = rolling_window_days * buckets_per_day
    
    print(f"Rolling Window: {rolling_window} buckets")
    
    # Calculate Thresholds
    # Using Rolling Quantile
    upper_threshold = (
        pl.Series(signal_series)
        .rolling_quantile(threshold_quantile, window_size=rolling_window)
        .shift(1)
        .fill_null(strategy="forward")
        .fill_null(np.percentile(signal_series, threshold_quantile * 100))
        .to_numpy()
    )
    
    lower_threshold = (
        pl.Series(signal_series)
        .rolling_quantile(1 - threshold_quantile, window_size=rolling_window)
        .shift(1)
        .fill_null(strategy="forward")
        .fill_null(np.percentile(signal_series, (1 - threshold_quantile) * 100))
        .to_numpy()
    )
    
    # Positions
    position = np.zeros(len(signal_series))
    position[signal_series > upper_threshold] = 1
    position[signal_series < lower_threshold] = -1
    
    # FIX LOOKAHEAD: Position determined at T trades Return at T+1
    # We use OFI(T) to predict Return(T+1)
    position = np.roll(position, 1)
    position[0] = 0
    
    # PnL
    gross_ret = position * frnb_series
    pos_change = np.abs(np.diff(position, prepend=0))
    costs = pos_change * (cost_bps / 10000.0)
    net_ret = gross_ret - costs
    
    # Stats
    cum_gross = np.cumsum(gross_ret)
    cum_net = np.cumsum(net_ret)
    n_trades = np.sum(pos_change)
    
    # Annualization Factor
    buckets_per_year = buckets_per_day * 365
    sharpe = np.mean(net_ret) / (np.std(net_ret) + 1e-9) * np.sqrt(buckets_per_year)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(cum_gross, label=f"Gross PnL (Cluster 1, {resample_minutes}m)", color='blue')
    plt.plot(cum_net, label=f"Net PnL (Cost {cost_bps}bps)", color='red')
    plt.plot(np.cumsum(frnb_series), label="Buy & Hold", color='gray', alpha=0.5, linestyle='--')
    plt.title(f"Resampled Backtest ({resample_minutes}m Buckets)\nThreshold: Top {int((1-threshold_quantile)*100)}%")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(out_dir, f"backtest_{resample_minutes}m.png")
    plt.savefig(plot_path)
    plt.close()
    
    print("\n--- Performance Stats ---")
    print(f"Timeframe: {resample_minutes} minutes")
    print(f"Total Buckets: {len(resampled)}")
    print(f"Time In Market: {np.mean(position != 0)*100:.1f}%")
    print(f"Turnover (Trades): {n_trades:.1f}")
    print(f"Avg Profit/Trade (Gross): {(np.sum(gross_ret)/(n_trades+1e-9))*10000:.2f} bps")
    print(f"Final Net Return: {cum_net[-1]*100:.2f}%")
    print(f"Annualized Sharpe: {sharpe:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/rolling_test_02_aligned/rolling_results.parquet")
    parser.add_argument("--out-dir", default="outputs/backtest_resampled")
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--cost-bps", type=float, default=3.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.75)
    args = parser.parse_args()
    
    run_resampled_backtest(
        args.data,
        args.out_dir,
        resample_minutes=args.minutes,
        cost_bps=args.cost_bps,
        threshold_quantile=args.threshold_quantile
    )
