#!/usr/bin/env python3
"""Pointline Data Lake pipeline for ClusterLOB (crypto, trade-triggered L2 context)."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np
import polars as pl
from sklearn.cluster import KMeans

from pointline import research
from pointline.config import get_exchange_id


@dataclass(frozen=True)
class SymbolMeta:
    exchange_id: int
    symbol_id: int
    price_increment: float
    amount_increment: float


def _to_us(ts: datetime) -> int:
    return int(ts.timestamp() * 1_000_000)


def resolve_symbol_meta(exchange: str, symbol: str, asof: datetime) -> SymbolMeta:
    exchange_id = get_exchange_id(exchange)
    asof_us = _to_us(asof)
    dim = research.scan_table(
        "dim_symbol",
        exchange_id=exchange_id,
        columns=[
            "symbol_id",
            "exchange_symbol",
            "valid_from_ts",
            "valid_until_ts",
            "price_increment",
            "amount_increment",
        ],
    )
    row = (
        dim.filter(
            (pl.col("exchange_symbol") == symbol)
            & (pl.col("valid_from_ts") <= asof_us)
            & (pl.col("valid_until_ts") > asof_us)
        )
        .select(["symbol_id", "price_increment", "amount_increment"])
        .collect()
    )
    if row.height == 0:
        raise ValueError(f"symbol not found in dim_symbol: {exchange} {symbol} @ {asof}")
    symbol_id, price_inc, amount_inc = row.row(0)
    return SymbolMeta(
        exchange_id=int(exchange_id),
        symbol_id=int(symbol_id),
        price_increment=float(price_inc),
        amount_increment=float(amount_inc),
    )


def load_trades_lazy(
    exchange: str,
    symbol_id: int,
    start_date: str,
    end_date: str,
) -> pl.LazyFrame:
    trades = research.load_trades(
        exchange=exchange,
        symbol_id=symbol_id,
        start_date=start_date,
        end_date=end_date,
        lazy=True,
    )
    return trades.select(
        [
            "date",
            "exchange_id",
            "symbol_id",
            "ts_local_us",
            "side",
            "price_int",
            "qty_int",
        ]
    )


def load_snapshots_lazy(
    exchange: str,
    exchange_id: int,
    symbol_id: int,
    start_date: str,
    end_date: str,
) -> pl.LazyFrame:
    snaps = research.scan_table(
        "book_snapshot_25",
        exchange=exchange,
        columns=[
            "date",
            "exchange_id",
            "symbol_id",
            "ts_local_us",
            "bids_px",
            "bids_sz",
            "asks_px",
            "asks_sz",
        ],
    )
    start = pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
    end = pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
    return snaps.filter(
        (pl.col("date") >= start)
        & (pl.col("date") <= end)
        & (pl.col("exchange_id") == exchange_id)
        & (pl.col("symbol_id") == symbol_id)
    )


def enrich_snapshots(
    snaps: pl.LazyFrame,
    price_increment: float,
    amount_increment: float,
    depth: int,
) -> pl.LazyFrame:
    price_inc = pl.lit(price_increment)
    amount_inc = pl.lit(amount_increment)
    snaps = snaps.sort("ts_local_us").with_columns(
        [
            pl.col("bids_px").list.first().alias("bid_px1_int"),
            pl.col("asks_px").list.first().alias("ask_px1_int"),
            pl.col("bids_sz").list.first().alias("bid_sz1_int"),
            pl.col("asks_sz").list.first().alias("ask_sz1_int"),
            pl.col("bids_sz").list.slice(0, depth).list.sum().alias("bid_sz_depth_int"),
            pl.col("asks_sz").list.slice(0, depth).list.sum().alias("ask_sz_depth_int"),
        ]
    )

    snaps = snaps.with_columns(
        [
            (pl.col("bid_px1_int") * price_inc).alias("bid_px1"),
            (pl.col("ask_px1_int") * price_inc).alias("ask_px1"),
            (pl.col("bid_sz1_int") * amount_inc).alias("bid_sz1"),
            (pl.col("ask_sz1_int") * amount_inc).alias("ask_sz1"),
            (pl.col("bid_sz_depth_int") * amount_inc).alias("bid_sz_depth"),
            (pl.col("ask_sz_depth_int") * amount_inc).alias("ask_sz_depth"),
        ]
    )

    snaps = snaps.with_columns(
        ((pl.col("bid_px1") + pl.col("ask_px1")) * 0.5).alias("mid_px")
    )

    snaps = snaps.with_columns(
        [
            (pl.col("mid_px") != pl.col("mid_px").shift(1)).alias("mid_change"),
            (pl.col("bid_px1") != pl.col("bid_px1").shift(1)).alias("bid1_change"),
            (pl.col("ask_px1") != pl.col("ask_px1").shift(1)).alias("ask1_change"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("mid_change"))
            .then(pl.col("ts_local_us"))
            .otherwise(None)
            .alias("mid_change_ts"),
            pl.when(pl.col("bid1_change"))
            .then(pl.col("ts_local_us"))
            .otherwise(None)
            .alias("bid1_change_ts"),
            pl.when(pl.col("ask1_change"))
            .then(pl.col("ts_local_us"))
            .otherwise(None)
            .alias("ask1_change_ts"),
        ]
    ).with_columns(
        [
            pl.col("mid_change_ts").forward_fill().alias("last_mid_change_ts"),
            pl.col("bid1_change_ts").forward_fill().alias("last_bid1_change_ts"),
            pl.col("ask1_change_ts").forward_fill().alias("last_ask1_change_ts"),
        ]
    )

    return snaps.select(
        [
            "date",
            "exchange_id",
            "symbol_id",
            "ts_local_us",
            "bid_px1",
            "ask_px1",
            "bid_sz1",
            "ask_sz1",
            "bid_sz_depth",
            "ask_sz_depth",
            "mid_px",
            "last_mid_change_ts",
            "last_bid1_change_ts",
            "last_ask1_change_ts",
        ]
    )


def join_trade_context(
    trades: pl.LazyFrame,
    snaps: pl.LazyFrame,
) -> pl.LazyFrame:
    return trades.sort("ts_local_us").join_asof(
        snaps.sort("ts_local_us"),
        on="ts_local_us",
        by=["exchange_id", "symbol_id"],
        strategy="backward",
    )


def compute_features(
    df: pl.LazyFrame,
    window: int,
    amount_increment: float,
) -> pl.LazyFrame:
    df = df.filter(pl.col("side").is_in([0, 1])).with_columns(
        [
            pl.when(pl.col("side") == 0)
            .then(pl.lit(1))
            .when(pl.col("side") == 1)
            .then(pl.lit(-1))
            .otherwise(None)
            .alias("sign"),
            (pl.col("qty_int")).cast(pl.Float64).alias("qty_int_f"),
        ]
    )

    qty = pl.col("qty_int_f") * pl.lit(amount_increment)

    df = df.with_columns(
        [
            (pl.col("sign") * qty).alias("signsize"),
            (qty / (pl.col("bid_sz1") + pl.col("ask_sz1"))).alias("v_rel"),
            (
                pl.when(pl.col("sign") == 1)
                .then(pl.col("ask_sz_depth"))
                .otherwise(pl.col("bid_sz_depth"))
            ).alias("sbs"),
            (
                pl.when(pl.col("sign") == 1)
                .then(pl.col("bid_sz_depth"))
                .otherwise(pl.col("ask_sz_depth"))
            ).alias("obs"),
            (
                (pl.col("ask_px1") - pl.col("bid_px1"))
                / pl.col("mid_px")
                * 1e4
            ).alias("spread_bps"),
            (pl.col("ts_local_us") - pl.col("last_mid_change_ts")).alias("t_m_us"),
            (
                pl.col("ts_local_us")
                - pl.when(pl.col("sign") == 1)
                .then(pl.col("last_ask1_change_ts"))
                .otherwise(pl.col("last_bid1_change_ts"))
            ).alias("t_age_us"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("sbs") + 1).log().alias("log_sbs"),
            (pl.col("obs") + 1).log().alias("log_obs"),
        ]
    )

    df = df.sort("ts_local_us").with_columns(
        [
            ((pl.col("v_rel") - pl.col("v_rel").rolling_mean(window))
             / pl.col("v_rel").rolling_std(window)).alias("z_v_rel"),
            ((pl.col("log_sbs") - pl.col("log_sbs").rolling_mean(window))
             / pl.col("log_sbs").rolling_std(window)).alias("z_sbs"),
            ((pl.col("log_obs") - pl.col("log_obs").rolling_mean(window))
             / pl.col("log_obs").rolling_std(window)).alias("z_obs"),
            ((pl.col("spread_bps") - pl.col("spread_bps").rolling_mean(window))
             / pl.col("spread_bps").rolling_std(window)).alias("z_spread"),
            ((pl.col("t_m_us") - pl.col("t_m_us").rolling_mean(window))
             / pl.col("t_m_us").rolling_std(window)).alias("z_t_m"),
            ((pl.col("t_age_us") - pl.col("t_age_us").rolling_mean(window))
             / pl.col("t_age_us").rolling_std(window)).alias("z_t_age"),
        ]
    )

    return df


def kmeans_cluster(df: pl.DataFrame, feature_cols: Sequence[str], k: int) -> pl.DataFrame:
    finite_mask = pl.all_horizontal([pl.col(c).is_finite() for c in feature_cols])
    clean = df.drop_nulls(feature_cols).filter(finite_mask)
    X = clean.select(feature_cols).to_numpy()
    model = KMeans(n_clusters=k, n_init="auto", random_state=7)
    labels = model.fit_predict(X)
    return clean.with_columns(pl.Series("cluster", labels))


def bucket_ofi(
    df: pl.DataFrame,
    bucket_us: int,
) -> pl.DataFrame:
    return (
        df.with_columns(
            (pl.col("ts_local_us") // bucket_us * bucket_us).alias("bucket_ts")
        )
        .group_by(
            ["date", "exchange_id", "symbol_id", "bucket_ts", "cluster"],
            maintain_order=True,
        )
        .agg(
            [
                pl.col("signsize").sum().alias("ofi_s"),
                pl.col("sign").sum().alias("ofi_c"),
            ]
        )
    )


def add_bucket_returns(
    bucketed: pl.DataFrame,
    snaps: pl.DataFrame,
    bucket_us: int,
) -> pl.DataFrame:
    base = (
        bucketed.select(["date", "exchange_id", "symbol_id", "bucket_ts"])
        .unique()
        .sort("bucket_ts")
    )

    snaps_mid = snaps.select(
        ["date", "exchange_id", "symbol_id", "ts_local_us", "mid_px"]
    ).sort("ts_local_us")

    base = base.join_asof(
        snaps_mid,
        left_on="bucket_ts",
        right_on="ts_local_us",
        by=["exchange_id", "symbol_id"],
        strategy="backward",
    ).rename({"mid_px": "mid_px_start"})
    if "ts_local_us_right" in base.columns:
        base = base.drop("ts_local_us_right")
    if "date_right" in base.columns:
        base = base.drop("date_right")

    base = base.with_columns((pl.col("bucket_ts") + bucket_us).alias("bucket_end_ts"))
    base = base.join_asof(
        snaps_mid,
        left_on="bucket_end_ts",
        right_on="ts_local_us",
        by=["exchange_id", "symbol_id"],
        strategy="backward",
    ).rename({"mid_px": "mid_px_end"})
    if "ts_local_us_right" in base.columns:
        base = base.drop("ts_local_us_right")
    if "date_right" in base.columns:
        base = base.drop("date_right")

    base = base.with_columns((pl.col("bucket_ts") + 2 * bucket_us).alias("bucket_end_next_ts"))
    base = base.join_asof(
        snaps_mid,
        left_on="bucket_end_next_ts",
        right_on="ts_local_us",
        by=["exchange_id", "symbol_id"],
        strategy="backward",
    ).rename({"mid_px": "mid_px_end_next"})
    if "ts_local_us_right" in base.columns:
        base = base.drop("ts_local_us_right")
    if "date_right" in base.columns:
        base = base.drop("date_right")

    base = base.with_columns(
        [
            (pl.col("mid_px_end") / pl.col("mid_px_start")).log().alias("CONR"),
            (pl.col("mid_px_end_next") / pl.col("mid_px_end")).log().alias("FRNB"),
        ]
    )

    return bucketed.join(
        base,
        on=["date", "exchange_id", "symbol_id", "bucket_ts"],
        how="left",
    )



def load_and_extract_features(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
    window: int,
    depth: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    asof = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    meta = resolve_symbol_meta(exchange, symbol, asof)

    trades = load_trades_lazy(exchange, meta.symbol_id, start_date, end_date)
    snaps = load_snapshots_lazy(exchange, meta.exchange_id, meta.symbol_id, start_date, end_date)
    snaps = enrich_snapshots(snaps, meta.price_increment, meta.amount_increment, depth)

    joined = join_trade_context(trades, snaps)
    feats = compute_features(joined, window, meta.amount_increment).collect()
    
    return feats, snaps.collect()

def run_pipeline(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
    window: int,
    k: int,
    bucket_us: int,
    depth: int,
    out_dir: str,
) -> None:
    asof = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    meta = resolve_symbol_meta(exchange, symbol, asof)

    trades = load_trades_lazy(exchange, meta.symbol_id, start_date, end_date)
    snaps = load_snapshots_lazy(exchange, meta.exchange_id, meta.symbol_id, start_date, end_date)
    snaps = enrich_snapshots(snaps, meta.price_increment, meta.amount_increment, depth)

    joined = join_trade_context(trades, snaps)
    feats = compute_features(joined, window, meta.amount_increment).collect()

    feature_cols = [
        "z_v_rel",
        "z_sbs",
        "z_obs",
        "z_spread",
        "z_t_m",
        "z_t_age",
    ]
    clustered = kmeans_cluster(feats, feature_cols, k=k)

    bucketed = bucket_ofi(clustered, bucket_us)
    snaps_df = snaps.collect()
    bucketed = add_bucket_returns(bucketed, snaps_df, bucket_us)

    os.makedirs(out_dir, exist_ok=True)
    clustered.write_parquet(os.path.join(out_dir, "trade_features.parquet"))
    bucketed.write_parquet(os.path.join(out_dir, "bucket_ofi.parquet"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClusterLOB crypto pipeline (Pointline data lake)")
    parser.add_argument("--exchange", required=True, help="exchange name, e.g. binance")
    parser.add_argument("--symbol", required=True, help="exchange symbol, e.g. BTCUSDT")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--window", type=int, default=100, help="rolling window size (trades)")
    parser.add_argument("--k", type=int, default=3, help="k-means clusters")
    parser.add_argument("--bucket-us", type=int, default=1_000_000, help="OFI bucket size in us")
    parser.add_argument("--depth", type=int, default=5, help="depth levels for SBS/OBS")
    parser.add_argument("--out-dir", default="outputs/crypto", help="output directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        exchange=args.exchange,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        k=args.k,
        bucket_us=args.bucket_us,
        depth=args.depth,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
