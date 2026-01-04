import argparse
import glob
import os
import sys

import polars as pl
import pandas as pd

# Add project root to path so we can import clusterlob
sys.path.append(os.getcwd())

from clusterlob.pipeline import (
    load_and_extract_features,
    fit_cluster_model,
    align_kmeans_labels,
    assign_clusters,
    bucket_ofi_all_and_clustered,
    add_bucket_returns,
    iter_dates,
    _load_cached_dates,
)


def _build_outsample_summary_if_missing(run_dir: str, horizons: list[int]) -> str | None:
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


def _aggregate_outsample(summary: pl.DataFrame) -> pl.DataFrame:
    return summary.group_by(["cluster", "horizon"]).agg(
        [
            pl.mean("ic_size").alias("mean_ic_size"),
            pl.std("ic_size").alias("std_ic_size"),
            pl.mean("ic_count").alias("mean_ic_count"),
            pl.std("ic_count").alias("std_ic_count"),
            pl.len().alias("n_windows"),
            pl.mean("n").alias("mean_n"),
        ]
    )


def _compute_insample_summary(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
    window: int,
    k: int,
    bucket_us: int,
    depth: int,
    horizons: list[int],
    cache_dir: str | None,
) -> pl.DataFrame:
    if cache_dir and os.path.exists(os.path.join(cache_dir, "features")):
        dates = iter_dates(start_date, end_date)
        feats, snaps = _load_cached_dates(cache_dir, dates)
    else:
        feats, snaps = load_and_extract_features(
            exchange=exchange,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            window=window,
            depth=depth,
        )

    feature_cols = [
        "z_v_rel",
        "z_sbs",
        "z_obs",
        "z_spread",
        "z_t_m",
        "z_t_age",
    ]

    model = fit_cluster_model(feats, feature_cols, k=k)
    label_map = align_kmeans_labels(model, feature_cols)
    clustered = assign_clusters(feats, model, feature_cols, label_map=label_map)

    bucketed = bucket_ofi_all_and_clustered(clustered, bucket_us)
    bucketed = add_bucket_returns(bucketed, snaps, bucket_us, horizons)

    summaries = []
    for h in horizons:
        r_col = f"r_{h}"
        if r_col not in bucketed.columns:
            continue
        df_h = bucketed.filter(pl.col(r_col).is_not_null())
        if df_h.height == 0:
            continue
        grouped = df_h.group_by("cluster").agg(
            [
                pl.corr("ofi_size", r_col).alias("mean_ic_size"),
                pl.corr("ofi_count", r_col).alias("mean_ic_count"),
                pl.len().alias("mean_n"),
            ]
        ).with_columns(
            [
                pl.lit(h).alias("horizon"),
                pl.lit(1).alias("n_windows"),
            ]
        )
        summaries.append(grouped)

    if not summaries:
        return pl.DataFrame()

    return pl.concat(summaries, how="vertical")


def compare_insample_outsample(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
    window: int,
    k: int,
    bucket_us: int,
    depth: int,
    horizons: list[int],
    out_dir: str,
    outsample_dir: str,
    outsample_start: str | None,
    outsample_end: str | None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Computing in-sample IC...")
    insample = _compute_insample_summary(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        window=window,
        k=k,
        bucket_us=bucket_us,
        depth=depth,
        horizons=horizons,
        cache_dir=os.path.join(outsample_dir, "cache"),
    )
    if insample.height == 0:
        raise RuntimeError("No in-sample summary computed.")

    summary_path = _build_outsample_summary_if_missing(outsample_dir, horizons)
    if summary_path is None:
        raise RuntimeError(f"No out-of-sample outputs found in {outsample_dir}")

    outsample = pl.read_parquet(summary_path)
    if outsample_start and outsample_end:
        outsample = outsample.filter(
            (pl.col("test_start") >= outsample_start)
            & (pl.col("test_start") <= outsample_end)
        )
    outsample = _aggregate_outsample(outsample)

    # Join on cluster + horizon
    joined = insample.join(
        outsample,
        on=["cluster", "horizon"],
        how="inner",
        suffix="_outsample",
    ).rename(
        {
            "mean_ic_size": "mean_ic_size_insample",
            "mean_ic_count": "mean_ic_count_insample",
            "mean_n": "mean_n_insample",
            "mean_ic_size_outsample": "mean_ic_size_outsample",
            "mean_ic_count_outsample": "mean_ic_count_outsample",
            "mean_n_outsample": "mean_n_outsample",
        }
    )

    # Compute ratios
    joined = joined.with_columns(
        [
            (pl.col("mean_ic_size_insample") / pl.col("mean_ic_size_outsample")).alias("ratio_size"),
            (pl.col("mean_ic_count_insample") / pl.col("mean_ic_count_outsample")).alias("ratio_count"),
        ]
    )

    out_path = os.path.join(out_dir, "insample_outsample_comparison.csv")
    joined.to_pandas().to_csv(out_path, index=False)

    print(f"Saved comparison to {out_path}")
    print("\nTop rows (by ratio_size):")
    print(joined.sort("ratio_size", descending=True).head(5).to_pandas())


def _parse_horizons(h: str) -> list[int]:
    return [int(x.strip()) for x in h.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare in-sample vs out-of-sample IC")
    parser.add_argument("--exchange", default="binance-futures")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--bucket-us", type=int, default=300_000_000)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--horizons", default="1,2,5")
    parser.add_argument("--out-dir", default="outputs/insample_outsample")
    parser.add_argument("--outsample-dir", required=True, help="pipeline out_dir with window outputs")
    parser.add_argument("--outsample-start", default=None, help="filter outsample test_start >= YYYY-MM-DD")
    parser.add_argument("--outsample-end", default=None, help="filter outsample test_start <= YYYY-MM-DD")
    args = parser.parse_args()

    compare_insample_outsample(
        exchange=args.exchange,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        k=args.k,
        bucket_us=args.bucket_us,
        depth=args.depth,
        horizons=_parse_horizons(args.horizons),
        out_dir=args.out_dir,
        outsample_dir=args.outsample_dir,
        outsample_start=args.outsample_start,
        outsample_end=args.outsample_end,
    )


if __name__ == "__main__":
    main()
