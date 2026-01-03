# ClusterLOB Trading Strategy: Signal Generation Specification

**Researcher:** Vector
**Version:** 2.0 (Post-Validation Pivot)
**Context:** Binance Futures (BTC/USDT)

---

## 1. Overview
This document specifies the **Real-Time Signal Generation Logic** derived from the ClusterLOB research. Following extensive backtesting (Experiments 1-6), the strategy has pivoted from a directional taker approach (unviable due to fees and alpha decay) to a **Signal-Augmented Market Maker** system. The goal is to use microstructure clustering to optimize queue positioning and quote skew.

---

## 2. Theoretical Basis
*   **The Alpha Source ($\\phi_2$):** "Opportunistic" flow (Cluster 1) predicts short-term (< 5 min) price momentum (+3.44% Gross PnL). It represents "Smart Money" accumulation.
*   **The Risk Source ($\\phi_1$):** "Directional" flow (Cluster 2) represents "Toxic" volatility. While mean-reverting at 5m, it signals high adverse selection risk for resting orders.
*   **The Strategy:** We do not chase price (Taker). We provide liquidity (Maker) and use the signals to **Skew** our prices (to capture $\\phi_2$ moves) and **Widen** our spreads (to avoid $\\phi_1$ toxicity).

---

## 3. Signal Architecture

### 3.1. Input Stream (Per Trade Event)
For every trade $i$ occurring at time $t$:
*   **$V_i$:** Trade Size (Quantity).
*   **$S_i$:** Trade Side (+1 for Buy, -1 for Sell).
*   **$C_i$:** Cluster Classification ($\\phi_1, \\phi_2, \\phi_3$) derived from real-time K-Means inference.

### 3.2. Dual Signal Accumulators
We maintain two separate accumulators with short half-lives ($\\tau \\approx 10s$) to match the instantaneous nature of the alpha.

**A. Alpha Skew Signal ($Sig_{skew}$):**
Accumulates only **$\\phi_2$ (Opportunistic)** flow.
$$v_{skew, i} = \\begin{cases} V_i \\cdot S_i & \\text{if } C_i = \\phi_2 \\\
0 & \\text{otherwise} \\end{cases}$$
$$ Sig_{skew, t} = Sig_{skew, t-1} \\cdot e^{-\\Delta t / \\tau} + v_{skew, t} $$

**B. Toxic Hazard Signal ($Sig_{hazard}$):**
Accumulates magnitude of **$\\phi_1$ (Directional)** flow.
$$v_{hazard, i} = \\begin{cases} |V_i| & \\text{if } C_i = \\phi_1 \\\
0 & \\text{otherwise} \\end{cases}$$
$$ Sig_{hazard, t} = Sig_{hazard, t-1} \\cdot e^{-\\Delta t / \\tau} + v_{hazard, t} $$

---

## 4. Execution Logic: Dynamic Skew & Spread

The core logic is to adjust the **Fair Value (FV)** and **Spread Width** relative to the Mid Price ($P_{mid}$).

### 4.1. Fair Value Adjustment (Skew)
We define a theoretical "Microstructure Fair Price" ($P_{micro}$) derived from the Alpha Skew Signal.

$$ P_{micro} = P_{mid} + \\beta \\cdot \\tanh(\\alpha \\cdot Sig_{skew}) $$

*   **Logic:** If $Sig_{skew}$ is positive (Buying Pressure), we shift our internal valuation *up*.
*   **Effect:** We quote higher Bids (more likely to be hit by sellers, acquiring Longs) and higher Asks (less likely to be hit by buyers, holding inventory).
*   **Result:** The strategy naturally accumulates inventory *with* the smart money flow.

### 4.2. Spread Widening (Defense)
We modulate the spread width ($W$) based on the Toxic Hazard Signal.

$$ W_{dynamic} = W_{base} \\cdot (1 + \\gamma \\cdot \\max(0, Sig_{hazard} - K_{tol})) $$

*   **Logic:** If $Sig_{hazard}$ exceeds a tolerance threshold $K_{tol}$, we widen quotes.
*   **Effect:** We demand a higher premium to provide liquidity during toxic/volatile periods, reducing the probability of adverse selection.

---

## 5. Quote Placement Specs

### 5.1. Bid & Ask Calculation
$$ P_{bid} = P_{micro} - \\frac{W_{dynamic}}{2} $$
$$ P_{ask} = P_{micro} + \\frac{W_{dynamic}}{2} $$

### 5.2. Inventory Management Overlay
To preventing risk ruin, we apply a standard inventory penalty term ($\\theta$) to the Fair Value:
$$ P_{final} = P_{micro} - \\theta \\cdot (Q_{inventory} - Q_{target}) $$

### 5.3. Summary of Interactions
1.  **Opportunistic Buying ($Sig_{skew} > 0$):** $P_{micro}$ rises. Bid moves closer to Mid. Ask moves away. We are eager to Buy.
2.  **Toxic Dump ($Sig_{hazard} \\uparrow$):** $W_{dynamic}$ expands. Both Bid and Ask move away from Mid. We retreat to safety.
3.  **Inventory Long ($Q > 0$):** $P_{final}$ drops. Bid moves down. Ask moves down. We are eager to Sell.

---

## 6. Daily Maintenance
*   **Retraining:** Every 24 hours (00:00 UTC) to update K-Means centroids.
*   **Calibration:** Recalibrate $\\beta$ (Skew Sensitivity) and $\\gamma$ (Hazard Sensitivity) based on previous day's volatility and alpha yield.