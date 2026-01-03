import sys
import os

# Add project root to path so we can import clusterlob
sys.path.append(os.getcwd())

from clusterlob.pipeline import run_pipeline
from clusterlob.analytics import analyze_cluster_centroids, compute_ic_pnl
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Futures 7-Day Clustering Experiment")
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-date", default="2024-05-01")
    parser.add_argument("--end-date", default="2024-05-07")
    parser.add_argument("--bucket-us", type=int, default=300000000) # 5 minutes
    parser.add_argument("--out-dir", default="outputs/crypto_futures_20240501_07")
    args = parser.parse_args()

    print(f"Running Pipeline for {args.exchange} {args.symbol}...")
    run_pipeline(
        exchange=args.exchange,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        window=100,
        k=3,
        bucket_us=args.bucket_us,
        depth=5,
        out_dir=args.out_dir
    )
    
    print("\nAnalyzing Clusters...")
    analyze_cluster_centroids(os.path.join(args.out_dir, "trade_features.parquet"))
    
    print("\nValidating Alpha...")
    compute_ic_pnl(
        os.path.join(args.out_dir, "bucket_ofi.parquet"),
        os.path.join(args.out_dir, "analysis")
    )

if __name__ == "__main__":
    main()
