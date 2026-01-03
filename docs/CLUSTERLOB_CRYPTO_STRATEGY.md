# ClusterLOB: Crypto Microstructure Adaptation Strategy
**Project:** Binance BTC/USDT Trade-Only Clustering
**Researcher Persona:** Vector (Senior Quant, HFT/Microstructure)
**Market Regime:** Volatility-Driven / High-Noise / Tick-Constrained

---

## 1. Executive Summary
This document outlines the adaptation of the **ClusterLOB** methodology for the cryptocurrency market, specifically **Binance BTC/USDT**. Utilizing "Choice 2" (Trade-Only Clustering), we aim to decompose the aggregate order flow into three behavioral regimes: **Directional ($\phi_1$)**, **Opportunistic ($\phi_2$)**, and **Noise/Market-Making ($\phi_3$)**.

The strategy leverages the `pointline` Data Lake to perform high-frequency feature engineering on Level 2 (L2) data, aiming to isolate the "Opportunistic" signal which identifies institutional accumulation patterns that survive the high-fee barrier of crypto trading.

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

The primary technical hurdle is the absence of **Level 3 (Order-Level)** data on Binance. We must pivot from tracking specific Order IDs to tracking **Liquidity States**.

### 3.1 Feature Engineering Specification
We will derive 6 core features ($X$) for every trade trigger, utilizing the `pointline` Delta Lake. These features are designed to capture the "Book Response" to the incoming trade flow.

| Feature | Notation | Crypto (L2) Implementation |
| :--- | :--- | :--- |
| **Relative Volume** | $V_{rel}$ | Ratio of trade quantity to total volume at the touch: $V_{trade} / V_{L1}$. Captures "level consumption." |
| **Mid-Price Recency** | $T_m$ | Time elapsed since the last mid-price change: $t_{now} - t_{last\_mid\_change}$. |
| **Stable-Liquidity Age** | $T_{age}$ | Time elapsed since the *Price Level* itself was established (i.e., first quote at this price). |
| **Same-Side Shape** | $SBS$ | Cumulative volume of the first 5 price levels on the trade's side of the book (Log-scaled). |
| **Opposite-Side Shape** | $OBS$ | Cumulative volume of the first 5 price levels on the opposite side (Log-scaled). |
| **Spread Width** | $S$ | The bid-ask spread normalized by the mid-price at the moment of execution (in basis points). |

---

## 4. Implementation Roadmap

### Phase 1: Data Binding (Polars/Delta Lake)
*   Integrate `pointline.research` to scan `silver.trades` and `silver.book_snapshots_top25`.
*   Implement `join_asof` logic to synchronize trade triggers with the most recent L2 snapshot context.
*   **Fixed-Point Handling:** Retain `int64` encoding for `price_int` and `qty_int` throughout the feature extraction to maintain precision.

### Phase 2: Unsupervised Learning Pipeline
1.  **Normalization:** Log-transform $V$ and apply Z-score standardization: $z = \frac{x - \mu}{\sigma}$.
2.  **Clustering:** Apply K-means++ ($k=3$) to a rolling 24-hour window.
3.  **Labeling:** Perform predictive regression: $r_{t+\Delta t} = \beta \cdot OFI(\phi_k)$.
    *   $\phi_1 \to$ High correlation with $r_{t+1s}$.
    *   $\phi_2 \to$ High correlation with $r_{t+60s}$.

### Phase 3: Backtesting & Fee Logic
The alpha must survive the **Taker Fee Barrier** (~2-4 bps).
*   **Execution Model:** Crossing the spread (Taker).
*   **Cost Model:** $PnL = (Sign \cdot \Delta Price) - (Fee + Slippage)$.
*   **Optimization:** Tune the $OFI(\phi_2)$ threshold to maximize the Sharpe Ratio net of costs.

---

## 5. Risk & Success Metrics
*   **Primary Metric:** Net Annualized Sharpe Ratio (Target: > 1.5 after fees).
*   **Stability:** Cluster assignment consistency across "Asian" vs. "US" trading sessions.
*   **Adverse Selection:** Monitor the "mark-out" (price move after trade) to ensure we are not being sniped by $\phi_1$ flow.

---
**Vector System Status: Strategic Documentation Complete.**
**Ready for Phase 1 Implementation.**
