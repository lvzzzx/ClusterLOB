# ClusterLOB: Crypto Microstructure Adaptation Strategy
**Project:** Binance BTC/USDT Trade-Only Clustering
**Researcher Persona:** Vector (Senior Quant, HFT/Microstructure)
**Market Regime:** Volatility-Driven / High-Noise / Tick-Constrained

---

## 1. Executive Summary
This document outlines the adaptation of the **ClusterLOB** methodology for the cryptocurrency market, specifically **Binance BTC/USDT**. Utilizing "Choice 2" (Trade-Only Clustering on L2), we aim to decompose the aggregate order flow into three behavioral regimes: **Directional ($\phi_1$)**, **Opportunistic ($\phi_2$)**, and **Noise/Market-Making ($\phi_3$)**.

The strategy leverages the `pointline` Data Lake to perform high-frequency feature engineering on Level 2 (L2) snapshots aligned to trade events, aiming to isolate the "Opportunistic" signal which identifies accumulation patterns that survive the fee barrier of crypto trading.

---

## 2. Theoretical Framework & Background

### 2.1 The Core Hypothesis: Information Asymmetry
Aggregate Order Flow Imbalance ($OFI$) is a sub-optimal signal because it aggregates heterogeneous intents. In HFT, we define the "Toxic" signal-to-noise ratio. By clustering trades based on their **Microstructure Context**, we can separate:
*   **Toxic Flow ($\phi_1$):** Informed traders with a very short alpha half-life (e.g., latency arbitrage).
*   **Latent Alpha ($\phi_2$):** Institutional "Smart Money" using execution algorithms (Icebergs, TWAP) to build positions.
*   **Noise ($\phi_3$):** Uninformed retail flow or passive market-making inventory rebalancing.

### 2.2 Why Crypto?
Crypto markets are characterized by massive retail participation and high noise. The original paper's focus on L3 (NASDAQ) is powerful but computationally expensive. Crypto's L2 environment, combined with high-conviction "Trade" events, allows for a more robust signal extraction that focuses on **Realized Information** (Action) rather than **Limit Orders** (Intent/Spoofing).

---

## 3. The Crypto Adaptation (L2 vs. L3)

The primary technical hurdle is the absence of **Level 3 (Order-Level)** data on Binance. We must pivot from tracking specific Order IDs to **trade-triggered L2 states** (book context immediately before the trade).

### 3.1 Feature Engineering Specification
We will derive core features ($X$) for every trade trigger using trade + L2 snapshot `asof` alignment. These features are designed to capture the "Book Response" to incoming trade flow.

| Feature | Notation | Crypto (L2) Implementation |
| :--- | :--- | :--- |
| **Relative Volume** | $V_{rel}$ | Ratio of trade quantity to total L1 volume: $V_{trade} / (V_{bid1}+V_{ask1})$. Captures "level consumption." |
| **Mid-Price Recency** | $T_m$ | Time since last mid-price change in snapshots: $t_{trade} - t_{last\_mid\_change}$. |
| **Level Update Age (Proxy)** | $T_{age}$ | Time since last update at the trade-side L1 price. If persistent L2 state is unavailable, drop this feature. |
| **Same-Side Shape** | $SBS$ | Cumulative volume of the first 5 price levels on the trade's side of the book (log-scaled). |
| **Opposite-Side Shape** | $OBS$ | Cumulative volume of the first 5 price levels on the opposite side (log-scaled). |
| **Spread Width** | $S$ | The bid-ask spread normalized by mid-price at execution (in bps). |

**Trade Sign / Size:**
*   **Trade Sign** $s_t \in \{-1, +1\}$ from aggressor flags (or tick rule fallback).
*   **Signed Size**: $SignSize = s_t \cdot V_{trade}$ used in OFI aggregation.

---

## 4. Implementation Roadmap

### Phase 1: Data Binding (Polars/Delta Lake)
*   Integrate `pointline.research` to scan `silver.trades` and `silver.book_snapshots_top25`.
*   Use `join_asof` to attach the most recent L2 snapshot **before** each trade (no lookahead).
*   **Fixed-Point Handling:** Retain `int64` encoding for `price_int` and `qty_int` throughout feature extraction.
*   **Aggressor Side:** Prefer exchange-provided side; fallback to tick rule only if needed.

### Phase 2: Unsupervised Learning Pipeline
1.  **Normalization:** Log-transform depth-related features ($SBS$, $OBS$) and apply Z-score standardization: $z = \frac{x - \mu}{\sigma}$. Leave ratios ($V_{rel}$) unlogged.
2.  **Clustering:** Apply K-means++ ($k=3$) to a rolling 24-hour window of trade events.
3.  **Labeling (Bucket-Based Horizons):** Use bucket returns rather than fixed seconds to avoid clock‑time mismatch in crypto. Define a bucket size $B$ (e.g., 5m). For each bucket:
    \[
    CONR = \log\left(\frac{P_{end}}{P_{start}}\right),\quad
    FRNB = \log\left(\frac{P_{end}^{next}}{P_{end}}\right)
    \]
    Then regress:
    \[
    r = \beta \cdot OFI(\phi_k)
    \]
    *   $\phi_1 \to$ High correlation with **CONR** (in‑bucket impact / toxic flow).
    *   $\phi_2 \to$ High correlation with **FRNB** (next‑bucket drift / latent flow).

**OFI Definition (Trade-Only):**
\[
OFI^{S}(\phi_k) = \sum_{i \in \phi_k} SignSize_i,\quad
OFI^{C}(\phi_k) = \sum_{i \in \phi_k} sign(SignSize_i)
\]

### Phase 3: Backtesting & Fee Logic
The alpha must survive the **Taker Fee Barrier** (parameterized by Binance tier).
*   **Execution Model:** Crossing the spread (Taker).
*   **Cost Model:** $PnL = (Sign \cdot \Delta Price) - (Fee + Slippage)$.
*   **Optimization:** Tune the $OFI(\phi_2)$ threshold to maximize Sharpe net of costs.

---

## 5. Risk & Success Metrics
*   **Primary Metric:** Net Annualized Sharpe Ratio (Target: > 1.5 after fees).
*   **Stability:** Cluster assignment consistency across "Asian" vs. "US" trading sessions.
*   **Adverse Selection:** Monitor the "mark-out" (price move after trade) to ensure we are not being sniped by $\phi_1$ flow.

---
**Vector System Status: Strategic Documentation Complete.**
**Ready for Phase 1 Implementation.**
