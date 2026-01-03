# Research Log: ClusterLOB - Binance-Futures BTC/USDT Adaptation

**Researcher:** Vector
**Date:** 2026-01-03
**Target:** Binance Futures (BTC/USDT)
**Objective:** Adapt ClusterLOB microstructure clustering (L2 Trade-Triggered) for high-frequency crypto derivatives.

---

## Experiment 1: 7-Day Microstructure Clustering (May 1–7, 2024)

### 1. Hypothesis
Crypto derivatives markets exhibit distinct liquidity regimes driven by leverage. We hypothesize that a 7-day training window provides sufficient data density to separate **Directional (Toxic)**, **Opportunistic (Latent Alpha)**, and **Market-Making (Passive)** flows using trade-triggered L2 features.

### 2. Experimental Setup
*   **Exchange:** `binance-futures`
*   **Symbol:** `BTCUSDT`
*   **Data Lake:** `pointline` (Schema: `trades`, `book_snapshot_25`)
*   **Period:** 2024-05-01 to 2024-05-07 (7 Days)
*   **Features:** $V_{rel}$ (Relative Vol), $SBS$ (Same-Side Depth), $OBS$ (Opposite-Side Depth), Spread, $T_m$ (Recency), $T_{age}$. All Z-Score standardized (rolling 100-trade window).
*   **Model:** K-Means++ ($k=3$)
*   **Event Volume:** 27,649,725 Trades

### 3. Results: Cluster Centroids

| Cluster ID | Label | Count | % Share | $z_{SBS}$ (Depth) | $z_{OBS}$ (OppDepth) | $z_{Tm}$ (Recency) | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Passive / Noise | 11.99M | 43.4% | ~0.00 | -0.12 | **+1.66** | **Low Urgency:** Executing after long intervals of mid-price stability. Likely passive accumulation or retail. |
| **1** | **Opportunistic ($\phi_2$)** | 5.25M | 19.0% | **+1.35** | **-1.25** | -0.60 | **Imbalance Exploitation:** Executing when own-side depth is thick and opposing depth is thin. Likely smart money positioning or sweeps. |
| **2** | Directional / Toxic | 10.40M | 37.6% | -0.75 | **+0.81** | -0.89 | **Adverse Selection:** Executing into a thick opposing book. High urgency, likely reacting to external latency signals or liquidations. |

### 4. Key Findings vs. Spot Market
*   **Volume Density:** Futures volume (27.6M) is ~12x higher than Spot (2.2M) for the same period.
*   **Feature Importance:** 
    *   **Spot:** Clusters separated primarily by **Spread** and **Time** (Volatility Regimes).
    *   **Futures:** Clusters separated primarily by **Order Book Imbalance** ($SBS$ vs $OBS$). Leverage participants are far more sensitive to depth available for execution.
*   **Stability:** The "Opportunistic" cluster (Cluster 1) represents a clear 19% minority, ideal for signal isolation.

## Experiment 2: Signal Validation (Alpha Check)

### 1. Methodology
We computed the **Information Coefficient (IC)** by correlating the Order Flow Imbalance (OFI) of each cluster (Size-weighted) with the 5-minute Future Return Next Bucket ($FRNB$).

### 2. Results (IC Values)

| Signal Source | IC ($R$) | Performance vs. Benchmark |
| :--- | :--- | :--- |
| **Cluster 1 ($\phi_2$ - Opportunistic)** | **0.0606** | **+75%** |
| Benchmark (Aggregate OFI) | 0.0346 | - |
| Cluster 0 (Passive) | 0.0205 | -41% |
| Cluster 2 (Directional) | -0.0065 | Negative (Mean Reversion) |

### 3. Conclusion
*   **$\phi_2$ is the Alpha:** Trades executing into favorable depth imbalance (High SBS / Low OBS) carry nearly **2x the predictive power** of the aggregate flow.
*   **Toxic Flow Reversion:** The negative IC on Cluster 2 suggests that highly aggressive "Directional" flow often marks local extremities, leading to mean reversion in the subsequent 5-minute window.
*   **Strategic Fit:** The system successfully isolates "Smart Money" accumulation from retail noise and toxic arbitrage.

---
**Status:** $\phi_2$ Alpha Confirmed. Pipeline validated.
**Next Steps:** Fee-inclusive backtesting and production deployment strategy.