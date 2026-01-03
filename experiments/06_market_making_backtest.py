import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple

# --- 1. Core Logic (From Experiment 05) ---

@dataclass
class TradeEvent:
    timestamp: float # Seconds
    price: float
    size: float
    side: int # +1 Buy, -1 Sell
    cluster_id: int 

class SignalAccumulator:
    """
    Real-time Signal Accumulator with Exponential Decay.
    Optimized for O(1) updates per trade.
    """
    def __init__(self, half_life_seconds: float):
        self.tau = half_life_seconds
        self.decay_const = np.log(2) / self.tau
        
        self.skew_val = 0.0
        self.hazard_val = 0.0
        self.last_update_ts = 0.0
        
        # Volatility Tracking (Welford's approx)
        self.rolling_vol_std = 1.0 

    def update(self, event: TradeEvent):
        dt = event.timestamp - self.last_update_ts
        if dt < 0: dt = 0 # Safety for disordered timestamps
        
        decay = np.exp(-self.decay_const * dt)
        
        # Decay
        self.skew_val *= decay
        self.hazard_val *= decay
        
        # Update Volatility Baseline (Slow moving average of trade sizes)
        alpha_vol = 0.001 
        self.rolling_vol_std = (1 - alpha_vol) * self.rolling_vol_std + alpha_vol * event.size
        
        # Accumulate
        if event.cluster_id == 2: # Alpha (Phi_2) - Inverted for Crypto
            self.skew_val += event.size * event.side
        elif event.cluster_id == 1: # Toxic (Phi_1) - Inverted for Crypto
            self.hazard_val += abs(event.size)
            
        self.last_update_ts = event.timestamp

    def get_normalized_signals(self) -> Tuple[float, float]:
        norm_factor = max(self.rolling_vol_std * 50.0, 1e-9) # Scaling factor tuned for "typical" large moves
        return self.skew_val / norm_factor, self.hazard_val / norm_factor

class MarketMakerStrategy:
    def __init__(self, 
                 beta_skew: float, 
                 gamma_hazard: float, 
                 theta_inventory: float,
                 base_spread_bps: float,
                 min_tick: float):
        
        self.beta = beta_skew
        self.gamma = gamma_hazard
        self.theta = theta_inventory
        self.base_spread_bps = base_spread_bps
        self.min_tick = min_tick
        
        # State
        self.active_bid = None
        self.active_ask = None
        
    def get_quotes(self, mid_price: float, norm_skew: float, norm_hazard: float, inventory: float):
        # 1. Alpha Skew
        alpha_adjust = self.beta * np.tanh(norm_skew) * self.min_tick # Scaled by ticks
        p_micro = mid_price + alpha_adjust
        
        # 2. Dynamic Inventory Interaction
        signal_confidence = abs(np.tanh(norm_skew))
        dynamic_theta = self.theta * (1.0 - signal_confidence)
        inv_penalty = dynamic_theta * inventory * self.min_tick
        
        p_final = p_micro - inv_penalty
        
        # 3. Hazard Spread
        # Base spread is in bps, convert to price
        half_spread_price = (mid_price * (self.base_spread_bps / 10000.0)) / 2.0
        
        # Hazard expands spread
        hazard_factor = max(0, norm_hazard - 0.2) # Threshold
        w_dynamic = half_spread_price * (1.0 + self.gamma * hazard_factor)
        
        raw_bid = p_final - w_dynamic
        raw_ask = p_final + w_dynamic
        
        # Round to Tick
        bid = round(raw_bid / self.min_tick) * self.min_tick
        ask = round(raw_ask / self.min_tick) * self.min_tick
        
        # Safety: Ensure no cross
        if bid >= ask:
            bid = ask - self.min_tick
            
        return bid, ask

# --- 2. Backtest Engine ---

def run_backtest(
    file_path: str,
    output_dir: str,
    requote_interval_ms: int = 200,
    latency_ms: int = 10,
    maker_fee_bps: float = -0.05, # Rebate (negative fee)
    taker_fee_bps: float = 2.0 # Only relevant if we implemented taking, keeping for ref
):
    print(f"Loading data from {file_path}...")
    df = pl.read_parquet(file_path).sort("ts_local_us")

    # Filter valid columns
    needed_cols = ["ts_local_us", "mid_px", "bid_px1", "ask_px1", "qty_int_f", "sign", "cluster"]
    df = df.select(needed_cols)
    
    # Constants
    MIN_TICK = 0.1 # BTCUSDT tick size
    AMOUNT_INC = 0.001 # BTCUSDT min size (approx)
    
    # Init Strategy
    accumulator = SignalAccumulator(half_life_seconds=10.0)
    strategy = MarketMakerStrategy(
        beta_skew=50.0,      # Max 50 ticks skew
        gamma_hazard=5.0,    # Max 5x spread expansion
        theta_inventory=2.0, # 2 ticks per unit inventory
        base_spread_bps=2.0, # 2 bps base spread (tight)
        min_tick=MIN_TICK
    )
    
    # State Variables
    inventory = 0.0
    cash = 0.0
    
    active_bid = None
    active_ask = None
    pending_bid = None
    pending_ask = None
    pending_effective_ts = None
    
    last_requote_ts = 0.0
    
    # History for plotting
    history_ts = []
    history_pnl = []
    history_inv = []
    history_mid = []
    
    # Iterate
    # Using iter_rows for simplicity in Python (slow but correct for logic verification)
    # For production backtesting on millions of rows, we'd vectorise this in Rust/Polars exprs.
    
    print("Starting simulation loop...")
    
    # Convert to numpy for slightly faster iteration
    data = df.to_numpy()
    # indices: 0=ts, 1=mid, 2=bid1, 3=ask1, 4=qty, 5=sign, 6=cluster
    
    start_ts_us = data[0, 0]
    
    trades_count = 0
    fills_count = 0
    buy_trades = 0
    sell_trades = 0
    buy_fills = 0
    sell_fills = 0
    fill_qty_total = 0.0
    fill_notional = 0.0
    fees_total = 0.0
    edge_qty_sum = 0.0
    edge_tick_qty_sum = 0.0
    quote_updates = 0
    quote_width_sum = 0.0
    quote_width_count = 0
    base_width_sum = 0.0
    hazard_width_sum = 0.0
    hazard_width_count = 0
    norm_skew_sum = 0.0
    norm_skew_abs_sum = 0.0
    norm_hazard_sum = 0.0
    norm_hazard_abs_sum = 0.0
    norm_obs = 0
    inv_min = 0.0
    inv_max = 0.0
    inv_abs_sum = 0.0
    inv_obs = 0
    inventory_sum = 0.0
    mid_sum = 0.0
    inv_mid_prod_sum = 0.0
    inv_sq_sum = 0.0
    mid_sq_sum = 0.0
    
    for row in data:
        ts_us = row[0]
        mid = row[1]
        bid1 = row[2]
        ask1 = row[3]
        qty = row[4] * AMOUNT_INC # Convert int to float qty
        side = row[5] # 1 (Buy) or -1 (Sell) -> This is the TAKER side
        cluster = row[6]
        
        ts_sec = (ts_us - start_ts_us) / 1e6
        
        # Apply pending quote updates once latency has elapsed
        if pending_effective_ts is not None and ts_us >= pending_effective_ts:
            active_bid = pending_bid
            active_ask = pending_ask
            pending_bid = None
            pending_ask = None
            pending_effective_ts = None

        # 1. Fill Logic (Conservative Cross)
        # Incoming BUY (side=1) matches our ASK
        if side == 1:
            buy_trades += 1
            # Trade happened at ask1. We get filled if our active_ask is <= ask1.
            # To be more conservative (assuming we are bottom of queue), 
            # we could require active_ask < ask1, but active_ask <= ask1 is standard.
            if active_ask is not None and ask1 >= active_ask:
                fill_qty = min(qty, 1.0) # Cap fill size per trade to realistic limit
                
                # We SELL
                inventory -= fill_qty
                cash += (fill_qty * active_ask)
                
                # Fee (Maker rebate is negative cost, i.e., profit)
                fee = (fill_qty * active_ask) * (maker_fee_bps / 10000.0)
                cash -= fee
                
                fills_count += 1
                sell_fills += 1
                fill_qty_total += fill_qty
                fill_notional += (fill_qty * active_ask)
                fees_total += fee
                edge = max(0.0, active_ask - mid)
                edge_qty_sum += edge * fill_qty
                edge_tick_qty_sum += (edge / MIN_TICK) * fill_qty
                
        # Incoming SELL (side=-1) matches our BID
        elif side == -1:
            sell_trades += 1
            # Trade happened at bid1. We get filled if our active_bid is >= bid1.
            if active_bid is not None and bid1 <= active_bid:
                fill_qty = min(qty, 1.0)
                
                # We BUY
                inventory += fill_qty
                cash -= (fill_qty * active_bid)
                
                # Fee
                fee = (fill_qty * active_bid) * (maker_fee_bps / 10000.0)
                cash -= fee
                
                fills_count += 1
                buy_fills += 1
                fill_qty_total += fill_qty
                fill_notional += (fill_qty * active_bid)
                fees_total += fee
                edge = max(0.0, mid - active_bid)
                edge_qty_sum += edge * fill_qty
                edge_tick_qty_sum += (edge / MIN_TICK) * fill_qty
        trades_count += 1
        inv_min = min(inv_min, inventory)
        inv_max = max(inv_max, inventory)
        inv_abs_sum += abs(inventory)
        inv_obs += 1
        inventory_sum += inventory
        mid_sum += mid
        inv_mid_prod_sum += inventory * mid
        inv_sq_sum += inventory * inventory
        mid_sq_sum += mid * mid

        # 2. Update Signal Accumulator (after fill checks to avoid look-ahead)
        event = TradeEvent(ts_sec, mid, qty, side, cluster)
        accumulator.update(event)

        # 3. Requote Logic (200ms Interval)
        if (ts_us - last_requote_ts) >= (requote_interval_ms * 1000):
            norm_skew, norm_hazard = accumulator.get_normalized_signals()

            # Calculate new quotes
            new_bid, new_ask = strategy.get_quotes(mid, norm_skew, norm_hazard, inventory)

            # Enqueue quote update with latency
            pending_bid = new_bid
            pending_ask = new_ask
            pending_effective_ts = ts_us + (latency_ms * 1000)

            last_requote_ts = ts_us
            quote_updates += 1
            quote_width_sum += (new_ask - new_bid)
            quote_width_count += 1
            base_width = mid * (strategy.base_spread_bps / 10000.0)
            hazard_width = max(0.0, (new_ask - new_bid) - base_width)
            base_width_sum += base_width
            hazard_width_sum += hazard_width
            hazard_width_count += 1
            norm_skew_sum += norm_skew
            norm_skew_abs_sum += abs(norm_skew)
            norm_hazard_sum += norm_hazard
            norm_hazard_abs_sum += abs(norm_hazard)
            norm_obs += 1

            # Record Metrics periodically
            if len(history_ts) == 0 or (ts_sec - history_ts[-1] > 60): # Every minute
                mtm = cash + (inventory * mid)
                history_ts.append(ts_sec)
                history_pnl.append(mtm)
                history_inv.append(inventory)
                history_mid.append(mid)
        
    # --- Analysis ---
    
    print(f"Simulation Complete. Processed {trades_count} trades.")
    print(f"Total Fills: {fills_count} (Fill Rate: {fills_count/trades_count*100:.2f}%)")
    if buy_trades > 0:
        print(f"Buy Trades: {buy_trades} | Sell Fills: {sell_fills} (Hit Rate: {sell_fills/buy_trades*100:.2f}%)")
    if sell_trades > 0:
        print(f"Sell Trades: {sell_trades} | Buy Fills: {buy_fills} (Hit Rate: {buy_fills/sell_trades*100:.2f}%)")
    if fill_qty_total > 0:
        avg_fill_size = fill_qty_total / max(fills_count, 1)
        avg_fill_px = fill_notional / fill_qty_total
        avg_edge = edge_qty_sum / fill_qty_total
        avg_edge_ticks = edge_tick_qty_sum / fill_qty_total
        print(f"Filled Qty: {fill_qty_total:.4f} BTC | Avg Fill Size: {avg_fill_size:.4f} BTC")
        print(f"Fill Notional: {fill_notional:.2f} USDT | VWAP: {avg_fill_px:.2f}")
        print(f"Avg Edge vs Mid: {avg_edge:.4f} USDT ({avg_edge_ticks:.2f} ticks)")
    if quote_width_count > 0:
        print(f"Quote Updates: {quote_updates} | Avg Quote Width: {(quote_width_sum/quote_width_count):.4f} USDT")
        print(f"Avg Base Width: {(base_width_sum/max(hazard_width_count,1)):.4f} USDT | Avg Hazard Width: {(hazard_width_sum/max(hazard_width_count,1)):.4f} USDT")
    if norm_obs > 0:
        print(f"Norm Skew Avg/AbsAvg: {(norm_skew_sum/norm_obs):.6f}/{(norm_skew_abs_sum/norm_obs):.6f}")
        print(f"Norm Hazard Avg/AbsAvg: {(norm_hazard_sum/norm_obs):.6f}/{(norm_hazard_abs_sum/norm_obs):.6f}")
    if inv_obs > 0:
        print(f"Inventory Min/Max: {inv_min:.4f}/{inv_max:.4f} BTC | Avg Abs Inv: {(inv_abs_sum/inv_obs):.4f} BTC")
        inv_mean = inventory_sum / inv_obs
        mid_mean = mid_sum / inv_obs
        inv_var = (inv_sq_sum / inv_obs) - (inv_mean * inv_mean)
        mid_var = (mid_sq_sum / inv_obs) - (mid_mean * mid_mean)
        cov = (inv_mid_prod_sum / inv_obs) - (inv_mean * mid_mean)
        if inv_var > 0 and mid_var > 0:
            corr = cov / (np.sqrt(inv_var) * np.sqrt(mid_var))
            print(f"Inventory-Mid Corr (simple): {corr:.6f}")
    print(f"Total Maker Fees (rebate < 0): {fees_total:.2f} USDT")
    
    final_mid = data[-1, 1]
    final_pnl = cash + (inventory * final_mid)
    
    print(f"\n--- Results ---")
    print(f"Final PnL: {final_pnl:.2f} USDT")
    print(f"Final Inventory: {inventory:.4f} BTC")
    
    # Plots
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # PnL
    plt.subplot(3, 1, 1)
    plt.plot(history_ts, history_pnl, label='Cumulative PnL (USDT)')
    plt.title(f"Backtest Results (Requote {requote_interval_ms}ms)")
    plt.legend()
    plt.grid(True)
    
    # Inventory
    plt.subplot(3, 1, 2)
    plt.plot(history_ts, history_inv, label='Inventory (BTC)', color='orange')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True)
    
    # Price
    plt.subplot(3, 1, 3)
    plt.plot(history_ts, history_mid, label='BTC Price', color='green')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "backtest_summary.png"))
    print(f"Plot saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/crypto_futures_20240501_07/trade_features.parquet")
    parser.add_argument("--out-dir", default="outputs/backtest_maker_01")
    parser.add_argument("--interval", type=int, default=200)
    args = parser.parse_args()
    
    run_backtest(args.data, args.out_dir, args.interval)
