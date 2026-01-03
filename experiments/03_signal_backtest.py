import polars as pl
import numpy as np
import argparse
import os
import tempfile

def run_bucket_backtest(
    data_path: str,
    out_dir: str,
    cost_bps: float = 4.0, # Binance VIP0 Taker Fee roughly
    entry_quantile: float = 0.80, # Enter when signal is very strong
    exit_quantile: float = 0.50, # Exit when signal weakens to median
    smooth_window: int = 1, # EMA smoothing span
    rolling_window: int = 288, # Approx 1 day of 5-min buckets
    use_rolling_thresholds: bool = True
):
    os.makedirs(out_dir, exist_ok=True)
    mpl_cache_dir = os.path.join(out_dir, ".mpl_cache")
    os.makedirs(mpl_cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print(f"Loading data from {data_path}...")
    df = pl.read_parquet(data_path)
    
    # 1. Pivot to get OFI_s for Cluster 1 (Opportunistic) aligned with Returns
    df = df.with_columns(pl.col("cluster").cast(pl.Utf8))
    pivoted = df.pivot(
        values="ofi_s",
        index=["bucket_ts"],
        on="cluster",
        aggregate_function="sum"
    ).fill_null(0).sort("bucket_ts")

    frnb = (
        df.group_by("bucket_ts")
        .agg(pl.col("FRNB").mean().alias("FRNB"))
        .sort("bucket_ts")
    )["FRNB"].to_numpy()
    
    # Cluster 1 is our Alpha Signal
    signal_col = "1" 
    signal_series_raw = pivoted[signal_col].to_numpy()
    
    # Apply Smoothing
    if smooth_window > 1:
        signal_series = (
            pl.Series(signal_series_raw)
            .ewm_mean(span=smooth_window, ignore_nulls=True)
            .to_numpy()
        )
        print(f"Applied EMA Smoothing (span={smooth_window})")
    else:
        signal_series = signal_series_raw

    # 2. Calculate Signal Strength Thresholds (Entry & Exit) 
    
    if use_rolling_thresholds:
        # Helper for rolling quantile
        def get_rolling_q(q):
            return (
                pl.Series(signal_series)
                .rolling_quantile(q, window_size=rolling_window)
                .shift(1)
                .fill_null(strategy="forward")
                .fill_null(np.percentile(signal_series, q * 100))
                .to_numpy()
            )
            
        up_entry = get_rolling_q(entry_quantile)
        up_exit = get_rolling_q(exit_quantile)
        
        low_entry = get_rolling_q(1 - entry_quantile)
        low_exit = get_rolling_q(1 - exit_quantile)
        
        print(
            f"Signal Thresholds (Rolling {rolling_window}):\n"
            f"  Long: Enter > p{entry_quantile:.2f}, Exit < p{exit_quantile:.2f}\n"
            f"  Short: Enter < p{1 - entry_quantile:.2f}, Exit > p{1 - exit_quantile:.2f}"
        )
    else:
        up_entry = np.full_like(signal_series, np.percentile(signal_series, entry_quantile * 100))
        up_exit = np.full_like(signal_series, np.percentile(signal_series, exit_quantile * 100))
        low_entry = np.full_like(signal_series, np.percentile(signal_series, (1 - entry_quantile) * 100))
        low_exit = np.full_like(signal_series, np.percentile(signal_series, (1 - exit_quantile) * 100))
        
        print(
            f"Signal Thresholds (Global):\n"
            f"  Long: Enter > {up_entry[0]:.2f}, Exit < {up_exit[0]:.2f}\n"
            f"  Short: Enter < {low_entry[0]:.2f}, Exit > {low_exit[0]:.2f}"
        )

    # 3. Hysteresis Loop
    position = np.zeros(len(signal_series))
    current_pos = 0.0
    
    for t in range(len(signal_series)):
        sig = signal_series[t]
        
        if current_pos == 0:
            if sig > up_entry[t]:
                current_pos = 1.0
            elif sig < low_entry[t]:
                current_pos = -1.0
        elif current_pos == 1:
            if sig < up_exit[t]:
                current_pos = 0.0
        elif current_pos == -1:
            if sig > low_exit[t]:
                current_pos = 0.0
        
        position[t] = current_pos
    
    # 4. PnL Calculation
    gross_ret = position * frnb
    
    # Cost is incurred when Position(t) != Position(t-1)
    pos_change = np.abs(np.diff(position, prepend=0))
    costs = pos_change * (cost_bps / 10000.0)
    
    net_ret = gross_ret - costs
    
    # 5. Metrics & Plotting
    cum_gross = np.cumsum(gross_ret)
    cum_net = np.cumsum(net_ret)
    cum_bh = np.cumsum(frnb)
    
    plt.figure(figsize=(12, 6))
    plt.plot(cum_gross, label="Gross PnL (Cluster 1)", color='blue')
    plt.plot(cum_net, label=f"Net PnL (Cost {cost_bps}bps)", color='red')
    plt.plot(cum_bh, label="Buy & Hold (BTC)", color='gray', alpha=0.5, linestyle='--')
    
    plt.title(f"Hysteresis + Smooth Backtest (Cluster 1 Alpha)\nEntry: p{entry_quantile:.2f} | Exit: p{exit_quantile:.2f} | Smooth: {smooth_window} | Cost: {cost_bps} bps")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(out_dir, "backtest_result.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Backtest plot saved to {plot_path}")
    
    # Stats
    n_trades = np.sum(pos_change)
    annual_factor = np.sqrt(288 * 365)
    sharpe = np.mean(net_ret) / (np.std(net_ret) + 1e-9) * annual_factor
    
    print("\n--- Performance Stats ---")
    print(f"Total Buckets: {len(pivoted)}")
    print(f"Time In Market: {np.mean(position != 0)*100:.1f}%")
    print(f"Turnover (Trades): {n_trades:.1f}")
    print(f"Avg Trade Duration: {(np.sum(np.abs(position)) / (n_trades + 1e-9)):.1f} buckets")
    print(f"Final Net Return: {cum_net[-1]*100:.2f}%")
    print(f"Annualized Sharpe: {sharpe:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/rolling_test_02_aligned/rolling_results.parquet")
    parser.add_argument("--out-dir", default="outputs/backtest_smooth")
    parser.add_argument("--cost-bps", type=float, default=3.0)
    parser.add_argument("--entry-quantile", type=float, default=0.80)
    parser.add_argument("--exit-quantile", type=float, default=0.50)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--rolling-window", type=int, default=288)
    parser.add_argument("--no-rolling-thresholds", action="store_true")
    args = parser.parse_args()
    
    run_bucket_backtest(
        args.data,
        args.out_dir,
        args.cost_bps,
        entry_quantile=args.entry_quantile,
        exit_quantile=args.exit_quantile,
        smooth_window=args.smooth_window,
        rolling_window=args.rolling_window,
        use_rolling_thresholds=not args.no_rolling_thresholds,
    )
