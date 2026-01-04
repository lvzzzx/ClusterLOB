#!/usr/bin/env python3
"""Pointline Data Lake pipeline for ClusterLOB (crypto, trade-triggered L2 context)."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Sequence

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
    return trades.sort(["exchange_id", "symbol_id", "ts_local_us"]).join_asof(
        snaps.sort(["exchange_id", "symbol_id", "ts_local_us"]),
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


def fit_cluster_model(
    df: pl.DataFrame, feature_cols: Sequence[str], k: int
) -> KMeans:
    finite_mask = pl.all_horizontal([pl.col(c).is_finite() for c in feature_cols])
    clean = df.drop_nulls(feature_cols).filter(finite_mask)
    X = clean.select(feature_cols).to_numpy()
    if X.shape[0] == 0:
        raise ValueError("no finite rows available for clustering fit")
    model = KMeans(n_clusters=k, n_init="auto", random_state=7)
    model.fit(X)
    return model


def align_kmeans_labels(model: KMeans, feature_cols: Sequence[str]) -> dict[int, int]:
    if model.n_clusters != 3:
        raise ValueError("rule-based alignment requires k=3")
    try:
        idx_sbs = feature_cols.index("z_sbs")
        idx_obs = feature_cols.index("z_obs")
    except ValueError as exc:
        raise ValueError("alignment requires z_sbs and z_obs in feature_cols") from exc

    centers = model.cluster_centers_
    scores = [float(c[idx_sbs] - c[idx_obs]) for c in centers]
    max_idx = max(range(len(scores)), key=scores.__getitem__)
    min_idx = min(range(len(scores)), key=scores.__getitem__)
    remaining = [i for i in range(len(scores)) if i not in (max_idx, min_idx)]
    if len(remaining) != 1:
        raise ValueError("alignment expects exactly one remaining cluster")
    return {max_idx: 1, min_idx: 2, remaining[0]: 0}


def assign_clusters(
    df: pl.DataFrame,
    model: KMeans,
    feature_cols: Sequence[str],
    label_map: dict[int, int] | None = None,
) -> pl.DataFrame:
    finite_mask = pl.all_horizontal([pl.col(c).is_finite() for c in feature_cols])
    clean = df.drop_nulls(feature_cols).filter(finite_mask)
    X = clean.select(feature_cols).to_numpy()
    if X.shape[0] == 0:
        return clean.with_columns(pl.lit(None).cast(pl.Int64).alias("cluster"))
    labels = model.predict(X)
    if label_map is not None:
        labels = [label_map.get(int(label), int(label)) for label in labels]
    return clean.with_columns(pl.Series("cluster", labels))


def bucket_ofi_all_and_clustered(
    df: pl.DataFrame,
    bucket_us: int,
) -> pl.DataFrame:
    bucketed = df.with_columns(
        (pl.col("ts_local_us") // bucket_us * bucket_us).alias("bucket_ts")
    )

    per_cluster = (
        bucketed.group_by(
            ["date", "exchange_id", "symbol_id", "bucket_ts", "cluster"],
            maintain_order=True,
        )
        .agg(
            [
                pl.col("signsize").sum().alias("ofi_size"),
                pl.col("sign").sum().alias("ofi_count"),
            ]
        )
    )

    overall = (
        bucketed.group_by(
            ["date", "exchange_id", "symbol_id", "bucket_ts"],
            maintain_order=True,
        )
        .agg(
            [
                pl.col("signsize").sum().alias("ofi_size"),
                pl.col("sign").sum().alias("ofi_count"),
            ]
        )
        .with_columns(pl.lit(-1).cast(pl.Int64).alias("cluster"))
    )

    overall = overall.select(per_cluster.columns)
    return pl.concat([per_cluster, overall], how="vertical")


def add_bucket_returns(
    bucketed: pl.DataFrame,
    snaps: pl.DataFrame,
    bucket_us: int,
    horizons: Sequence[int],
) -> pl.DataFrame:
    base = (
        bucketed.select(["date", "exchange_id", "symbol_id", "bucket_ts"])
        .unique()
        .sort(["date", "exchange_id", "symbol_id", "bucket_ts"])
    )

    snaps_mid = snaps.select(
        ["date", "exchange_id", "symbol_id", "ts_local_us", "mid_px"]
    ).sort(["date", "exchange_id", "symbol_id", "ts_local_us"])

    base = base.join_asof(
        snaps_mid,
        left_on="bucket_ts",
        right_on="ts_local_us",
        by=["date", "exchange_id", "symbol_id"],
        strategy="backward",
    ).rename({"mid_px": "mid_px_start"})
    if "ts_local_us_right" in base.columns:
        base = base.drop("ts_local_us_right")

    base = base.with_columns((pl.col("bucket_ts") + bucket_us).alias("bucket_end_ts"))
    base = base.join_asof(
        snaps_mid,
        left_on="bucket_end_ts",
        right_on="ts_local_us",
        by=["date", "exchange_id", "symbol_id"],
        strategy="backward",
    ).rename({"mid_px": "mid_px_end"})
    if "ts_local_us_right" in base.columns:
        base = base.drop("ts_local_us_right")

    horizon_cols = []
    for h in horizons:
        horizon_ts_col = f"bucket_end_h{h}_ts"
        horizon_mid_col = f"mid_px_end_h{h}"
        base = base.with_columns(
            (pl.col("bucket_ts") + (1 + h) * bucket_us).alias(horizon_ts_col)
        )
        base = base.join_asof(
            snaps_mid,
            left_on=horizon_ts_col,
            right_on="ts_local_us",
            by=["date", "exchange_id", "symbol_id"],
            strategy="backward",
        ).rename({"mid_px": horizon_mid_col})
        if "ts_local_us_right" in base.columns:
            base = base.drop("ts_local_us_right")
        horizon_cols.append(horizon_mid_col)

    base = base.with_columns(
        [
            (pl.col("mid_px_end") / pl.col("mid_px_start")).log().alias("r_0"),
        ]
    )

    for h, mid_col in zip(horizons, horizon_cols):
        base = base.with_columns(
            (pl.col(mid_col) / pl.col("mid_px_end")).log().alias(f"r_{h}")
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

def _parse_date(d: str) -> date:
    return datetime.fromisoformat(d).date()


def _format_date(d: date) -> str:
    return d.isoformat()

def iter_dates(start_date: str, end_date: str) -> list[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    out = []
    cur = start
    while cur <= end:
        out.append(_format_date(cur))
        cur += timedelta(days=1)
    return out


def walk_forward_windows(
    start_date: str,
    end_date: str,
    train_days: int,
    test_days: int,
    step_days: int,
) -> Iterable[tuple[int, str, str, str, str]]:
    dates = iter_dates(start_date, end_date)

    window_id = 0
    idx = 0
    while True:
        train_start_idx = idx
        train_end_idx = idx + train_days - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_days - 1
        if test_end_idx >= len(dates):
            break
        train_start = dates[train_start_idx]
        train_end = dates[train_end_idx]
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]
        yield window_id, train_start, train_end, test_start, test_end
        window_id += 1
        idx += step_days


def _cache_daily_features(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
    window: int,
    depth: int,
    cache_dir: str,
) -> None:
    feats_dir = os.path.join(cache_dir, "features")
    snaps_dir = os.path.join(cache_dir, "snaps")
    os.makedirs(feats_dir, exist_ok=True)
    os.makedirs(snaps_dir, exist_ok=True)

    for d in iter_dates(start_date, end_date):
        feat_path = os.path.join(feats_dir, f"date={d}.parquet")
        snap_path = os.path.join(snaps_dir, f"date={d}.parquet")
        if os.path.exists(feat_path) and os.path.exists(snap_path):
            continue
        feats, snaps = load_and_extract_features(
            exchange=exchange,
            symbol=symbol,
            start_date=d,
            end_date=d,
            window=window,
            depth=depth,
        )
        feats.write_parquet(feat_path)
        snaps.write_parquet(snap_path)


def _load_cached_dates(cache_dir: str, dates: Sequence[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    feats_dir = os.path.join(cache_dir, "features")
    snaps_dir = os.path.join(cache_dir, "snaps")
    feat_paths = [os.path.join(feats_dir, f"date={d}.parquet") for d in dates]
    snap_paths = [os.path.join(snaps_dir, f"date={d}.parquet") for d in dates]

    feats = pl.concat([pl.read_parquet(p) for p in feat_paths], how="vertical")
    snaps = pl.concat([pl.read_parquet(p) for p in snap_paths], how="vertical")
    return feats, snaps


def run_pipeline(
    exchange: str,
    symbol: str,
    start_date: str,
    end_date: str,
    window: int,
    k: int,
    bucket_us: int,
    depth: int,
    horizons: Sequence[int],
    train_days: int,
    test_days: int,
    step_days: int,
    out_dir: str,
) -> None:
    feature_cols = [
        "z_v_rel",
        "z_sbs",
        "z_obs",
        "z_spread",
        "z_t_m",
        "z_t_age",
    ]
    os.makedirs(out_dir, exist_ok=True)

    cache_dir = os.path.join(out_dir, "cache")
    _cache_daily_features(
        exchange=exchange,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        window=window,
        depth=depth,
        cache_dir=cache_dir,
    )

    for (
        window_id,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in walk_forward_windows(start_date, end_date, train_days, test_days, step_days):
        train_dates = iter_dates(train_start, train_end)
        test_dates = iter_dates(test_start, test_end)
        feats_train, _ = _load_cached_dates(cache_dir, train_dates)
        feats_test, snaps_test = _load_cached_dates(cache_dir, test_dates)

        model = fit_cluster_model(feats_train, feature_cols, k=k)
        label_map = align_kmeans_labels(model, feature_cols)
        clustered = assign_clusters(feats_test, model, feature_cols, label_map=label_map)

        bucketed = bucket_ofi_all_and_clustered(clustered, bucket_us)
        bucketed = add_bucket_returns(bucketed, snaps_test, bucket_us, horizons)

        bucketed = bucketed.with_columns(
            [
                pl.lit(window_id).alias("window_id"),
                pl.lit(train_start).alias("train_start"),
                pl.lit(train_end).alias("train_end"),
                pl.lit(test_start).alias("test_start"),
                pl.lit(test_end).alias("test_end"),
                pl.lit(bucket_us).alias("bucket_us"),
                pl.lit(k).alias("k"),
            ]
        )

        out_path = os.path.join(out_dir, f"bucket_ofi_returns_window_{window_id}.parquet")
        bucketed.write_parquet(out_path)


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
    parser.add_argument("--horizons", default="1,2,5", help="comma-separated bucket horizons")
    parser.add_argument("--train-days", type=int, default=7, help="training window in days")
    parser.add_argument("--test-days", type=int, default=1, help="test window in days")
    parser.add_argument("--step-days", type=int, default=1, help="walk-forward step in days")
    parser.add_argument("--out-dir", default="outputs/crypto", help="output directory")
    return parser.parse_args()

def _parse_horizons(horizons: str) -> list[int]:
    return [int(h.strip()) for h in horizons.split(",") if h.strip()]


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
        horizons=_parse_horizons(args.horizons),
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
