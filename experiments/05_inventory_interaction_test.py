import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os

# Set random seed for reproducibility
np.random.seed(42)

@dataclass
class TradeEvent:
    timestamp: float # Seconds
    price: float
    size: float
    side: int # +1 Buy, -1 Sell
    cluster_id: int # 0: Noise, 1: Toxic (Phi_1), 2: Alpha (Phi_2)

class SignalAccumulator:
    """
    Real-time Signal Accumulator with Exponential Decay.
    Optimized for O(1) updates per trade.
    """
    def __init__(self, half_life_seconds: float, decay_dt_step: float = 1.0):
        self.tau = half_life_seconds
        # Pre-calculate decay factor for the standard time step (e.g., 1s) to save exp() calls
        # if updates are perfectly regular. For irregular, we calculate on fly.
        self.decay_const = np.log(2) / self.tau
        
        self.skew_val = 0.0
        self.hazard_val = 0.0
        self.last_update_ts = 0.0
        
        # Track volume stats for normalization (Vector's Requirement 1)
        self.vol_sum = 0.0
        self.vol_sq_sum = 0.0
        self.vol_count = 0
        self.rolling_vol_std = 1.0 # Default to 1 to avoid div/0

    def _update_vol_stats(self, size: float):
        # Simple Welford's online algorithm or simple rolling window approx could go here.
        # For this prototype, we'll use a fast exponential moving variance approach
        # to normalize the signal dynamically.
        alpha_vol = 0.001 # Slow decay for volatility baseline
        self.rolling_vol_std = (1 - alpha_vol) * self.rolling_vol_std + alpha_vol * size

    def update(self, event: TradeEvent) -> Tuple[float, float]:
        """
        Updates signals based on trade event.
        Returns (skew_signal, hazard_signal).
        """
        dt = event.timestamp - self.last_update_ts
        decay = np.exp(-self.decay_const * dt)
        
        # Decay previous state
        self.skew_val *= decay
        self.hazard_val *= decay
        
        # Update Volatility Baseline
        self._update_vol_stats(event.size)
        
        # Accumulate New Flow
        if event.cluster_id == 2: # Phi_2: Alpha/Opportunistic
            self.skew_val += event.size * event.side
            
        elif event.cluster_id == 1: # Phi_1: Toxic/Directional
            # Toxic flow increases Hazard signal (magnitude)
            self.hazard_val += abs(event.size)
            
        self.last_update_ts = event.timestamp
        return self.skew_val, self.hazard_val

    def get_normalized_signals(self) -> Tuple[float, float]:
        """
        Returns signals normalized by rolling volume volatility.
        Prevents tanh saturation during high-vol events.
        """
        # Normalize by a multiple of typical trade size/volatility
        norm_factor = max(self.rolling_vol_std * 10, 1e-9) 
        return self.skew_val / norm_factor, self.hazard_val / norm_factor

class MarketMakerStrategy:
    def __init__(self, 
                 beta_skew: float, 
                 gamma_hazard: float, 
                 theta_inventory: float,
                 base_spread: float,
                 interaction_mode: str = 'FIXED'):
        
        self.beta = beta_skew
        self.gamma = gamma_hazard
        self.theta = theta_inventory
        self.base_spread = base_spread
        self.mode = interaction_mode
        
        self.inventory = 0.0
        self.cash = 0.0
        self.pnl_history = []
        self.inventory_history = []
        
    def get_quotes(self, mid_price: float, raw_skew: float, raw_hazard: float) -> Tuple[float, float, float]:
        """
        Calculates Bid, Ask, and Micro-Fair-Value.
        """
        # 1. Calculate Normalized Skew (Sig_skew_norm) to feed into tanh
        # Assuming input raw_skew is already 'reasonable' or we apply alpha scaling here
        # For this sim, we assume raw_skew is passed from get_normalized_signals
        
        # Alpha Logic
        alpha_adjust = self.beta * np.tanh(raw_skew) # Bounded [-beta, +beta]
        p_micro = mid_price + alpha_adjust
        
        # 2. Inventory Logic (The Interaction Test)
        inv_penalty = 0.0
        if self.mode == 'FIXED':
            inv_penalty = self.theta * self.inventory
            
        elif self.mode == 'DYNAMIC':
            # Vector's Logic: 
            # If Signal Confidence is High (abs(tanh(skew)) -> 1), reduce penalty.
            # If Signal is Weak (0), full penalty.
            signal_confidence = abs(np.tanh(raw_skew)) # 0 to 1
            dynamic_theta = self.theta * (1.0 - signal_confidence)
            inv_penalty = dynamic_theta * self.inventory
            
        p_final = p_micro - inv_penalty
        
        # 3. Hazard Logic (Spread Widening)
        # Hazard is purely additive to spread
        hazard_factor = max(0, raw_hazard - 0.5) # Threshold 0.5 arbitrary for sim
        w_dynamic = self.base_spread * (1.0 + self.gamma * hazard_factor)
        
        bid = p_final - (w_dynamic / 2.0)
        ask = p_final + (w_dynamic / 2.0)
        
        return bid, ask, p_final

def run_simulation(n_events=10000, interaction_mode='FIXED'):
    # Simulation Parameters
    mid_price = 10000.0
    volatility = 20.0 # Price brownian motion
    
    accumulator = SignalAccumulator(half_life_seconds=10.0)
    strategy = MarketMakerStrategy(
        beta_skew=5.0,      # Aggressive Skew ($5 max)
        gamma_hazard=2.0,   # 2x Spread expansion on hazard
        theta_inventory=0.5,# $0.50 price shift per unit of inventory
        base_spread=2.0,    # $2.00 Base Spread
        interaction_mode=interaction_mode
    )
    
    # Storage
    timestamps = []
    mids = []
    inventories = []
    pnls = []
    skew_signals = []
    
    t = 0.0
    
    # Generate Synthetic Trade Stream
    # We create a "Trend" mode where Price moves AND Cluster 2 buys happen together
    # to test if the strategy captures the move vs dumping inventory.
    
    trend_active = False
    trend_direction = 0
    
    for i in range(n_events):
        dt = np.random.exponential(1.0) # Avg 1 sec between trades
        t += dt
        
        # 1. Evolve Price (GBM + Trend)
        trend_drift = 0.0
        if i % 200 == 0: # Switch regimes every ~200 trades
             if np.random.rand() < 0.3:
                 trend_active = True
                 trend_direction = 1 if np.random.rand() > 0.5 else -1
             else:
                 trend_active = False
                 
        if trend_active:
            trend_drift = trend_direction * 2.0 # Strong drift
            
        price_change = np.random.normal(0, volatility * np.sqrt(dt/86400)) + trend_drift
        mid_price += price_change
        
        # 2. Generate Trade Event
        # If Trending, more likely to see Cluster 2 (Alpha) trades in direction of trend
        is_alpha_trade = False
        is_toxic_trade = False
        
        if trend_active:
            # 60% chance of Alpha trade driving the trend
            if np.random.rand() < 0.6:
                cluster = 2
                side = trend_direction
                size = np.random.lognormal(0, 0.5) * 5.0 # Large aggressive
            else:
                cluster = 0
                side = 1 if np.random.rand() > 0.5 else -1
                size = np.random.lognormal(0, 0.5)
        else:
            # Mean reverting / Noise
            rand = np.random.rand()
            if rand < 0.1: # 10% Toxic
                cluster = 1
                side = 1 if np.random.rand() > 0.5 else -1
                size = np.random.lognormal(0, 1.0) * 10.0 # Huge toxic
            elif rand < 0.3: # 20% Opportunistic (randomly appearing)
                cluster = 2
                side = 1 if np.random.rand() > 0.5 else -1
                size = np.random.lognormal(0, 0.5) * 2.0
            else: # Noise
                cluster = 0
                side = 1 if np.random.rand() > 0.5 else -1
                size = np.random.lognormal(0, 0.5)
                
        event = TradeEvent(t, mid_price, size, side, cluster)
        
        # 3. Update Signals
        accumulator.update(event)
        norm_skew, norm_hazard = accumulator.get_normalized_signals()
        
        # 4. Market Maker Action (Before filling this trade, we effectively quoted)
        # In a real backtest, we'd check if previous quotes were hit. 
        # Here we simplify: if trade crosses our quote, we fill.
        
        bid, ask, fv = strategy.get_quotes(mid_price, norm_skew, norm_hazard)
        
        filled_side = 0
        fill_price = 0.0
        
        # Incoming SELL (side = -1) hits our BID
        if event.side == -1 and event.price <= bid:
            filled_side = 1 # We Buy
            fill_price = bid
        # Incoming BUY (side = 1) hits our ASK
        elif event.side == 1 and event.price >= ask:
            filled_side = -1 # We Sell
            fill_price = ask
            
        if filled_side != 0:
            strategy.inventory += (filled_side * event.size)
            strategy.cash -= (filled_side * event.size * fill_price)
            
        # Mark to Market PnL
        mtm_value = strategy.cash + (strategy.inventory * mid_price)
        
        timestamps.append(t)
        mids.append(mid_price)
        inventories.append(strategy.inventory)
        pnls.append(mtm_value)
        skew_signals.append(norm_skew)

    return pd.DataFrame({
        'time': timestamps,
        'mid': mids,
        'inventory': inventories,
        'pnl': pnls,
        'skew': skew_signals
    })

def main():
    out_dir = "outputs/interaction_test"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Running FIXED Interaction Simulation...")
    df_fixed = run_simulation(n_events=20000, interaction_mode='FIXED')
    
    print("Running DYNAMIC Interaction Simulation...")
    df_dynamic = run_simulation(n_events=20000, interaction_mode='DYNAMIC')
    
    # Analysis
    # We want to see if Dynamic held more inventory during the Trend (high Skew)
    # and if that resulted in higher PnL.
    
    fixed_pnl = df_fixed['pnl'].iloc[-1]
    dyn_pnl = df_dynamic['pnl'].iloc[-1]
    
    print("\n--- Results ---")
    print(f"Fixed Inventory PnL:   ${fixed_pnl:.2f}")
    print(f"Dynamic Inventory PnL: ${dyn_pnl:.2f}")
    print(f"Improvement:           {(dyn_pnl - fixed_pnl)/abs(fixed_pnl)*100:.2f}%")
    
    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Price & Signal
    plt.subplot(3, 1, 1)
    plt.plot(df_fixed['time'], df_fixed['mid'], label='Mid Price', color='black', alpha=0.5)
    plt.ylabel('Price')
    plt.title('Market Simulation: Price vs Signals')
    
    ax1b = plt.gca().twinx()
    ax1b.fill_between(df_fixed['time'], df_fixed['skew'], color='green', alpha=0.1, label='Skew Signal')
    ax1b.set_ylabel('Alpha Skew')
    
    # Inventory Comparison
    plt.subplot(3, 1, 2)
    plt.plot(df_fixed['time'], df_fixed['inventory'], label='Fixed Inv', color='red', linewidth=1)
    plt.plot(df_dynamic['time'], df_dynamic['inventory'], label='Dynamic Inv', color='blue', linewidth=1)
    plt.ylabel('Inventory')
    plt.legend()
    plt.title('Inventory Management Comparison')
    
    # PnL Comparison
    plt.subplot(3, 1, 3)
    plt.plot(df_fixed['time'], df_fixed['pnl'], label='Fixed PnL', color='red')
    plt.plot(df_dynamic['time'], df_dynamic['pnl'], label='Dynamic PnL', color='blue')
    plt.ylabel('Cumulative PnL')
    plt.legend()
    plt.title('Performance Comparison')
    
    save_path = os.path.join(out_dir, "interaction_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
