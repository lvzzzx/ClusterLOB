# ClusterLOB Trading Strategy: Signal Generation Specification

**Researcher:** Vector
**Version:** 1.0
**Context:** Binance Futures (BTC/USDT)

---

## 1. Overview
This document specifies the **Real-Time Signal Generation Logic** derived from the ClusterLOB research. The strategy is a **Signal-Following** system that isolates and tracks the order flow of "Opportunistic" ($\phi_2$) market participants, who have been statistically proven (IC ~0.06) to predict short-term price movements ($FRNB$).

---

## 2. Theoretical Basis
*   **The Alpha Source:** Market participants classified as $\phi_2$ (Opportunistic) execute trades when the Order Book (L2) exhibits a favorable imbalance (High Same-Side Depth, Low Opposite-Side Depth).
*   **The Noise:** Aggregate volume includes "Passive" ($\phi_3$) and "Toxic/Directional" ($\phi_1$) flows which dilute or negatively correlate with future returns.
*   **The Logic:** By filtering the trade stream to accumulate only $\phi_2$ volume, we construct a "Smart Money" pressure gauge that ignores noise.

---

## 3. Signal Architecture

### 3.1. Input Stream (Per Trade Event)
For every trade $i$ occurring at time $t$:
*   **$V_i$:** Trade Size (Quantity).
*   **$S_i$:** Trade Side (+1 for Buy, -1 for Sell).
*   **$C_i$:** Cluster Classification ($\phi_1, \phi_2, \phi_3$) derived from the real-time K-Means inference on L2 features ($SBS, OBS, T_m, \dots$).

### 3.2. Flow Filtering
We define the **Signed Opportunistic Flow ($v_i$)** for trade $i$:

$$v_i = \begin{cases} V_i \cdot S_i & \text{if } C_i = \phi_2 \text{ (Opportunistic)} \\ 0 & \text{otherwise} \end{cases}$$

*Note: Trades from Cluster 0 (Passive) and Cluster 2 (Toxic) result in $v_i = 0$. They are treated as silence.*

### 3.3. The Accumulator (Leaky Bucket)
To model the "current pressure" and account for alpha decay, we use an **Exponential Decay Accumulator**.

$$ \text{Signal}_t = \text{Signal}_{t-1} \cdot \lambda + v_t $$

*   **$	ext{Signal}_t$**: The active trading signal strength at trade $t$.
*   **$\\\lambda$**: Decay factor (e.g., $0.95$ per event, or time-based decay $e^{-\Delta t / \tau}$). 
    *   *Recommended Half-Life ($\\tau$):* 30-60 seconds (matching the 5-minute alpha horizon).

---

## 4. Execution Logic (Schmitt Trigger)

To prevent signal flickering and reduce transaction costs, we employ a hysteresis-based trigger system.

### 4.1. Parameters
*   **$K_{open}$:** Threshold to enter a position (High Conviction).
*   **$K_{close}$:** Threshold to exit a position (Signal Fade).
*   *Constraint:* $K_{close} < K_{open}$

### 4.2. State Machine

| Current State | Condition | Action | New State |
| :--- | :--- | :--- | :--- |
| **FLAT** | $\text{Signal}_t > +K_{open}$ | **BUY (Long)** | **LONG** |
| **FLAT** | $\text{Signal}_t < -K_{open}$ | **SELL (Short)** | **SHORT** |
| **LONG** | $\text{Signal}_t < +K_{close}$ | **CLOSE LONG** | **FLAT** |
| **SHORT** | $\text{Signal}_t > -K_{close}$ | **CLOSE SHORT** | **FLAT** |

---

## 5. Execution Style & Risk Management

### 5.1. Execution (Passive Join)
Since $\phi_2$ trades imply a "Wall" of support (High Same-Side Depth):
*   **Entry:** Place Limit Orders at the Best Bid (for Long) or Best Ask (for Short).
*   **Rationale:** We join the "Smart Money" wall. The high depth protects us from immediate adverse selection.
*   **Benefit:** Capture Maker Rebates (or reduce Taker fees), critical for HFT profitability.

### 5.2. Toxic Flow Protection (Circuit Breaker)
We monitor the **Toxic Flow Accumulator** ($\\text{Signal}_{\\phi_1}$) separately.
*   **Rule:** If $|\text{Signal}_{\\phi_1}| > K_{danger}$:
    *   Cancel all resting limit orders.
    *   Widen spreads immediately.
    *   *Reasoning:* A surge in Toxic flow indicates high volatility/breakout potential where passive orders are likely to be run over.

---

## 6. Daily Maintenance
*   **Retraining:** Every 24 hours (00:00 UTC).
*   **Alignment:** Run the centroid-alignment algorithm to ensure Cluster ID $X$ maps correctly to the $\phi_2$ definition (Max $SBS - OBS$).
*   **Deployment:** Hot-swap the updated K-Means centroids into the inference engine.
