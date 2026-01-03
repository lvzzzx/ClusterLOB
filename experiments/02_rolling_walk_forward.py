
import sys
import os
import argparse
from datetime import datetime, timedelta
import polars as pl
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.getcwd())

from clusterlob.pipeline import load_and_extract_features, bucket_ofi, add_bucket_returns, kmeans_cluster

def date_range(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)

def ensure_daily_features(exchange, symbol, start_date, end_date, window, depth, features_dir):
    """
    Step 1: Pre-compute features for each day if not already cached.
    """
    print(f"--- Stage 1: Pre-computing Daily Features ({start_date} to {end_date}) ---")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    os.makedirs(features_dir, exist_ok=True)
    
    for dt in date_range(start_dt, end_dt):
        day_str = dt.strftime("%Y-%m-%d")
        feat_path = os.path.join(features_dir, f"features_{day_str}.parquet")
        snap_path = os.path.join(features_dir, f"snaps_{day_str}.parquet")
        
        if os.path.exists(feat_path) and os.path.exists(snap_path):
            print(f"Skipping {day_str} (Found in cache)")
            continue
            
        print(f"Processing {day_str}...")
        try:
            feats, snaps = load_and_extract_features(
                exchange, symbol, day_str, day_str, window, depth
            )
            feats.write_parquet(feat_path)
            snaps.write_parquet(snap_path)
        except Exception as e:
            print(f"Error processing {day_str}: {e}")

def rolling_walk_forward(
    start_date: str,
    end_date: str,
    train_window_days: int,
    k: int,
    bucket_us: int,
    features_dir: str,
    out_dir: str,
):
    """
    Step 2: Rolling Train (7 days) -> Test (1 day) Loop
    """
    print(f"\n--- Stage 2: Rolling Walk-Forward ({train_window_days}-Day Train Window) ---")
    
    feature_cols = ["z_v_rel", "z_sbs", "z_obs", "z_spread", "z_t_m", "z_t_age"]
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    results = []
    
    # We iterate such that 'test_day' goes from start_date + window to end_date
    current_test_dt = start_dt + timedelta(days=train_window_days)
    
    while current_test_dt <= end_dt:
        test_day_str = current_test_dt.strftime("%Y-%m-%d")
        train_start_dt = current_test_dt - timedelta(days=train_window_days)
        train_end_dt = current_test_dt - timedelta(days=1)
        
        print(f"Train: [{train_start_dt.date()} .. {train_end_dt.date()}] -> Test: {test_day_str}")
        
        # 1. Load Training Data
        train_dfs = []
        for d in date_range(train_start_dt, train_end_dt):
            p = os.path.join(features_dir, f"features_{d.strftime('%Y-%m-%d')}.parquet")
            if os.path.exists(p):
                train_dfs.append(pl.read_parquet(p))
        
        if not train_dfs:
            print(f"  No training data for {test_day_str}, skipping.")
            current_test_dt += timedelta(days=1)
            continue
            
        train_data = pl.concat(train_dfs)
        
        # 2. Fit Model
        # Remove nulls/infs just like inside the pipeline
        finite_mask = pl.all_horizontal([pl.col(c).is_finite() for c in feature_cols])
        clean_train = train_data.drop_nulls(feature_cols).filter(finite_mask)
        X_train = clean_train.select(feature_cols).to_numpy()
        
        if X_train.shape[0] < 1000:
            print("  Insufficient training data, skipping.")
            current_test_dt += timedelta(days=1)
            continue
            
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        kmeans.fit(X_train)
        
        # --- Cluster Alignment ---
        # Map clusters to consistent labels:
        # 0: Passive (High Tm or remaining)
        # 1: Opportunistic (Alpha) -> Max (SBS - OBS)
        # 2: Directional (Toxic) -> Max (OBS - SBS)
        
        centroids = kmeans.cluster_centers_
        # feature_cols indices: 0:z_v_rel, 1:z_sbs, 2:z_obs, 3:z_spread, 4:z_t_m, 5:z_t_age
        
        # 1. Identify Opportunistic (Label 1)
        # Criteria: Maximize (z_sbs - z_obs)
        opp_scores = centroids[:, 1] - centroids[:, 2]
        opp_idx = np.argmax(opp_scores)
        
        # 2. Identify Directional (Label 2)
        # Criteria: Maximize (z_obs - z_sbs), excluding opp_idx
        dir_scores = centroids[:, 2] - centroids[:, 1]
        dir_scores_masked = dir_scores.copy()
        dir_scores_masked[opp_idx] = -np.inf
        dir_idx = np.argmax(dir_scores_masked)
        
        # 3. Identify Passive (Label 0)
        # Remaining cluster
        passive_idx = [i for i in range(k) if i != opp_idx and i != dir_idx][0]
        
        # Create mapping: original_idx -> aligned_label
        mapping = {
            passive_idx: 0,
            opp_idx: 1,
            dir_idx: 2
        }
        
        # 3. Load Test Data
        test_feat_path = os.path.join(features_dir, f"features_{test_day_str}.parquet")
        test_snap_path = os.path.join(features_dir, f"snaps_{test_day_str}.parquet")
        
        if not (os.path.exists(test_feat_path) and os.path.exists(test_snap_path)):
            print(f"  Missing test data for {test_day_str}, skipping.")
            current_test_dt += timedelta(days=1)
            continue
            
        test_feats = pl.read_parquet(test_feat_path)
        test_snaps = pl.read_parquet(test_snap_path)
        
        # 4. Predict on Test
        # We must filter test data same as train for valid prediction
        clean_test = test_feats.drop_nulls(feature_cols).filter(finite_mask)
        if clean_test.height == 0:
            current_test_dt += timedelta(days=1)
            continue

        X_test = clean_test.select(feature_cols).to_numpy()
        labels = kmeans.predict(X_test)
        
        # Apply alignment mapping
        aligned_labels = np.array([mapping[l] for l in labels])
        
        clustered_test = clean_test.with_columns(pl.Series("cluster", aligned_labels))
        
        # 5. Compute Alpha (OFI vs FRNB)
        bucketed = bucket_ofi(clustered_test, bucket_us)
        # We need snaps for return calculation
        bucketed_with_ret = add_bucket_returns(bucketed, test_snaps, bucket_us)
        
        # Filter valid returns
        valid_res = bucketed_with_ret.filter(pl.col("FRNB").is_not_null())
        
        # Append to results
        # We store the raw bucket data, tagged with the test date
        results.append(valid_res)
        
        current_test_dt += timedelta(days=1)

    if not results:
        print("No results generated.")
        return

    # --- Stage 3: Aggregation & Performance ---
    print("\n--- Stage 3: Performance Aggregation ---")
    all_res = pl.concat(results)
    
    os.makedirs(out_dir, exist_ok=True)
    all_res.write_parquet(os.path.join(out_dir, "rolling_results.parquet"))
    
    # Calculate PnL Curve
    # Pivot
    pivoted = all_res.pivot(
        values="ofi_s",
        index=["bucket_ts", "FRNB"],
        on="cluster",
        aggregate_function="sum"
    ).fill_null(0).sort("bucket_ts")
    
    # Identify Phi_2 (Opportunistic)
    # Clusters are now aligned:
    # 0: Passive
    # 1: Opportunistic (Alpha)
    # 2: Directional
    
    cluster_names = {
        0: "Passive (Noise)",
        1: "Opportunistic (Alpha)",
        2: "Directional (Toxic)"
    }
    
    plt.figure(figsize=(12, 6))
    
    for c in range(k):
        col = str(c)
        if col not in pivoted.columns: continue
        
        # PnL = Sign(OFI) * Return
        # We use series math directly
        pnl_series = (pivoted[col].sign() * pivoted["FRNB"]).cum_sum()
        plt.plot(pnl_series.to_numpy(), label=cluster_names.get(c, f"Cluster {c}"))
        
    plt.title(f"Rolling Walk-Forward PnL (Train={train_window_days}d)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "rolling_pnl.png"))
    print(f"Saved rolling PnL plot to {out_dir}/rolling_pnl.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--train-window", type=int, default=7)
    parser.add_argument("--features-dir", default="data/daily_features")
    parser.add_argument("--out-dir", default="outputs/rolling_experiment")
    args = parser.parse_args()
    
    ensure_daily_features(
        args.exchange, args.symbol, args.start_date, args.end_date,
        window=100, depth=5, features_dir=args.features_dir
    )
    
    rolling_walk_forward(
        args.start_date, args.end_date, args.train_window,
        k=3, bucket_us=300_000_000, 
        features_dir=args.features_dir, out_dir=args.out_dir
    )
