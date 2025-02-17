#!/usr/bin/env python3
"""
Improved Dual Network Trading System with Enhanced Risk Management,
Detailed Metrics, CLI, and Optimal Strategy Network Training

Features:
  - Prediction Network: Predicts next–step percentage change with uncertainty.
  - Improved Strategy Network: Now uses a deeper architecture with BatchNorm, Dropout,
    orthogonal initialization, and a risk–aware leverage adjustment.
  - Enhanced Reward: Implements an improved reward function incorporating intermediate
    rewards (e.g. unrealized P&L), risk–adjusted metrics (e.g. via prediction uncertainty),
    and an entropy bonus.
  - Environment & Metrics:
      • Tracks portfolio history to calculate Sharpe ratio and maximum drawdown.
      • Reward on closing a position (as % profit/loss × leverage) with heavy penalties on liquidation.
  - Experience Replay with stratified sampling.
  - Learning Rate Scheduling for both networks.
  - Data Cleaning: Subcommand to generate “clean” data from raw data.
  - CLI: Interactive command–line interface with subcommands.
  - Both vectorized and non–vectorized strategy training options.
  - Optimized for MacBook M3 (using MPS if available).
  - Supports resuming from checkpoints and saving the best model.
  - Option to train indefinitely (until Ctrl-C is pressed).
  - Each simulation episode now uses 72 rows of history (context) and 288 rows for simulation.

Note:
  • Old checkpoints from previous versions are not fully compatible with the new model architectures.
"""

import os
import random
import argparse
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F  # For improved network functions

# Allow the NumPy scalar global (required on PyTorch 2.6+)
torch.serialization.add_safe_globals(["numpy._core.multiarray.scalar"])

# Set device (using MPS on a MacBook M3 if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[{datetime.now()}] Using device: {device}")

# Fixed subset size for mini-batch sampling in strategy training
FIXED_SUBSET_SIZE = 200

##############################################
# Checkpoint Fix Helpers
##############################################
def fix_volume_checkpoint(state_dict):
    """For a plain StrategyNetwork checkpoint, add 'base_net.' prefix if needed."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("base_net."):
            new_state_dict["base_net." + k] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def fix_prediction_checkpoint(state_dict):
    """Remove any '_orig_mod.' prefix (from torch.compile) from the keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        new_state_dict[new_key] = v
    return new_state_dict

##############################################
# Utility Functions
##############################################
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

def generate_clean_data(raw_csv, clean_csv):
    print(f"[{datetime.now()}] Generating clean data from {raw_csv}...")
    data = pd.read_csv(raw_csv)
    data["timestamp"] = pd.to_datetime(data["open_time"], unit='ms')
    data["hour"] = data["timestamp"].dt.hour + data["timestamp"].dt.minute / 60.0
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["weekday"] = data["timestamp"].dt.weekday
    data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)
    data["rsi"] = compute_rsi(data["close"], window=14)
    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    cols = ["timestamp", "close"]
    if "high" in data.columns and "low" in data.columns:
        cols.extend(["high", "low"])
    if "volume" in data.columns:
        cols.append("volume")
    cols.extend(["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "rsi", "macd"])
    clean_data = data[cols].dropna().reset_index(drop=True)
    clean_data.to_csv(clean_csv, index=False)
    print(f"[{datetime.now()}] Clean data saved to {clean_csv}")

def load_and_preprocess_data(csv_path, seq_length=72):
    print(f"[{datetime.now()}] Loading and preprocessing data from {csv_path}...")
    data = pd.read_csv(csv_path, parse_dates=["timestamp"])
    data.sort_values("timestamp", inplace=True)
    print(f"[{datetime.now()}] Calculating technical features (returns, ma20, ma50)...")
    data["returns"] = data["close"].pct_change()
    data["ma20"] = data["close"].rolling(20).mean()
    data["ma50"] = data["close"].rolling(50).mean()
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(data[["returns", "ma20", "ma50"]])
    data[["returns", "ma20", "ma50"]] = scaler.transform(data[["returns", "ma20", "ma50"]])
    sim_length = 288  # simulation steps per episode
    episodes = []
    print(f"[{datetime.now()}] Building episodes (history: {seq_length}, simulation length: {sim_length})...")
    for start in range(0, len(data) - seq_length - sim_length, sim_length):
        episode = data.iloc[start : start + seq_length + sim_length].reset_index(drop=True)
        episodes.append(episode)
    print(f"[{datetime.now()}] Created {len(episodes)} episodes.")
    features = list(data.columns)
    return episodes, features, scaler

def generate_prediction_data(csv_path, seq_length=72):
    print(f"[{datetime.now()}] Generating prediction data from {csv_path}...")
    data = pd.read_csv(csv_path, parse_dates=["timestamp"])
    data.sort_values("timestamp", inplace=True)
    data.reset_index(drop=True, inplace=True)
    states = []
    targets = []
    for i in range(seq_length, len(data) - 1):
        window = data["close"].iloc[i - seq_length : i].values
        returns = (window[1:] - window[:-1]) / window[:-1]
        states.append(returns)
        current_price = window[-1]
        next_price = data["close"].iloc[i]
        target = (next_price - current_price) / current_price
        targets.append(target)
    states = np.array(states)
    targets = np.array(targets)
    print(f"[{datetime.now()}] Generated {len(states)} training samples for prediction network.")
    return states, targets

##############################################
# Helper Functions for Stratified Sampling
##############################################
def is_bull_episode(episode):
    return episode["returns"].mean() > 0

def is_bear_episode(episode):
    return episode["returns"].mean() < 0

def sample_stratified(episodes_subset, batch_size):
    bull = [ep for ep in episodes_subset if is_bull_episode(ep)]
    bear = [ep for ep in episodes_subset if is_bear_episode(ep)]
    half = batch_size // 2
    batch = []
    if len(bull) >= half and len(bear) >= half:
        batch += random.sample(bull, half)
        batch += random.sample(bear, half)
        if len(batch) < batch_size:
            remainder = batch_size - len(batch)
            batch += random.sample(episodes_subset, remainder)
    else:
        batch = random.sample(episodes_subset, batch_size)
    return batch

##############################################
# 1. Prediction Network & Training
##############################################
class PredictionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_out_size = hidden_size * (2 if bidirectional else 1)
        self.layernorm = nn.LayerNorm(lstm_out_size)
        self.fc_mean = nn.Linear(lstm_out_size, 1)
        self.fc_log_std = nn.Linear(lstm_out_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.layernorm(out)
        mean = self.fc_mean(out)
        log_std = self.fc_log_std(out)
        return mean.squeeze(-1), log_std.squeeze(-1)

def train_prediction_network(csv_path, seq_length=72, num_epochs=10, batch_size=64,
                             lr=0.0001, checkpoint_path="pred_net.pt", resume=False,
                             best_checkpoint_path="best_pred_net.pt", infinite=False):
    print(f"[{datetime.now()}] Starting training of Prediction Network...")
    if resume and os.path.exists(best_checkpoint_path):
        print(f"[{datetime.now()}] Found best prediction checkpoint. Loading from {best_checkpoint_path}...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model = PredictionNetwork(input_size=1).to(device)
        state_dict = fix_prediction_checkpoint(checkpoint["model_state_dict"])
        model.load_state_dict(state_dict)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint.get("best_loss", float('inf'))
        print(f"[{datetime.now()}] Resumed Prediction Network from epoch {start_epoch}")
    else:
        model = PredictionNetwork(input_size=1).to(device)
        start_epoch = 0
        best_loss = float('inf')
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    states, targets = generate_prediction_data(csv_path, seq_length=seq_length)
    states = states[..., np.newaxis]
    dataset = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(targets))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epoch = start_epoch
    while True:
        epoch += 1
        total_loss = 0.0
        all_preds = []
        all_targets = []
        for batch_states, batch_targets in dataloader:
            batch_states = batch_states.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            # Use autocast for MPS (float16)
            with torch.autocast("mps", dtype=torch.float16):
                pred_mean, pred_log_std = model(batch_states)
                pred_std = torch.exp(pred_log_std) + 1e-6
                loss = 0.5 * ((batch_targets - pred_mean) ** 2 / (pred_std ** 2)) + pred_log_std
                loss = loss.mean()
            torch.cuda.zero_grad()  # For MPS, ensure proper gradient zeroing
            loss.backward()
            # Gradient clipping added here:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch_states.size(0)
            all_preds.append(pred_mean.detach().cpu().numpy())
            all_targets.append(batch_targets.detach().cpu().numpy())
        avg_loss = total_loss / len(dataset)
        preds = np.concatenate(all_preds)
        targets_arr = np.concatenate(all_targets)
        mae = np.mean(np.abs(preds - targets_arr))
        correct_direction = np.mean(np.sign(preds) == np.sign(targets_arr)) * 100
        print(f"[{datetime.now()}] Epoch {epoch}: Avg Loss: {avg_loss:.6f}, MAE: {mae:.6f}, Correct Direction: {correct_direction:.2f}%")
        scheduler.step(avg_loss)
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
        }
        torch.save(checkpoint_dict, checkpoint_path)
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_dict["best_loss"] = best_loss
            torch.save(checkpoint_dict, best_checkpoint_path)
            print(f"[{datetime.now()}] New best Prediction Network saved to {best_checkpoint_path}")
        if not infinite and epoch >= start_epoch + num_epochs:
            break
    return model

##############################################
# 2. Improved Strategy Network & Environment
##############################################
class ImprovedStrategyNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc_adv = nn.Linear(hidden_size, 5)   # Advantage stream
        self.fc_val = nn.Linear(hidden_size, 1)    # Value stream
        self.fc_leverage = nn.Linear(hidden_size, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, pred_std=None):
        h = self.fc1(x)
        h = self.batch_norm1(h)
        h = F.relu(h)
        h = self.dropout1(h)
        
        h = self.fc2(h)
        h = self.batch_norm2(h)
        h = F.relu(h)
        h = self.dropout2(h)
        
        adv = self.fc_adv(h)
        val = self.fc_val(h)
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        
        lev = F.softplus(self.fc_leverage(h))
        if pred_std is not None:
            lev = lev * torch.exp(-pred_std)
        return q_val, lev.squeeze(-1)

class TradingEnvStrategy:
    def __init__(self, episode, features, pred_net, seq_length=72, initial_cash=100.0, sim_steps=288,
                 max_drawdown_limit=0.2):
        self.episode = episode
        self.prices = episode["close"].values
        if "high" in episode.columns and "low" in episode.columns:
            self.highs = episode["high"].values
            self.lows = episode["low"].values
        else:
            self.highs = self.prices
            self.lows = self.prices
        self.seq_length = seq_length
        self.initial_cash = initial_cash
        self.sim_steps = sim_steps
        self.pred_net = pred_net
        self.fee = 0.001 # TODO
        # New parameters for risk management:
        self.holding_reward_coef = 0.001  # Reward for holding an open position (unrealized PnL)
        self.max_drawdown_limit = max_drawdown_limit  # e.g. 20% max drawdown allowed
        self.reset()
    
    def reset(self):
        self.t = self.seq_length
        self.portfolio = self.initial_cash
        self.position = "flat"  # "flat", "long", or "short"
        self.entry_price = None
        self.trades = 0
        self.wins = 0
        self.liquidations = 0
        self.portfolio_history = [self.portfolio]
        self.peak = self.portfolio  # For max drawdown tracking
        return self._get_state()
    
    def _get_state(self):
        window = self.prices[self.t - self.seq_length:self.t]
        returns = (window[1:] - window[:-1]) / window[:-1]
        returns = returns[np.newaxis, :, np.newaxis]
        with torch.no_grad():
            pred_mean, pred_log_std = self.pred_net(torch.FloatTensor(returns).to(device))
        pred_std = torch.exp(pred_log_std)
        pos = 0.0
        if self.position == "long":
            pos = 1.0
        elif self.position == "short":
            pos = -1.0
        state = torch.cat([
            pred_mean.view(1),
            pred_std.view(1),
            torch.tensor([pos], dtype=torch.float32, device=device).view(1)
        ], dim=0)
        return state.unsqueeze(0)
    
    def step(self, action, leverage):
        # Ensure leverage is between 1 and 100
        leverage = max(1.0, min(leverage, 100.0))
        current_price = self.prices[self.t]
        next_index = self.t + 1
        if next_index < len(self.prices):
            next_price = self.prices[next_index]
            next_high = self.highs[next_index]
            next_low = self.lows[next_index]
        else:
            next_price = current_price
            next_high = current_price
            next_low = current_price
        safety_margin = 1.1
        reward = 0.0

        # Process actions
        if action == 0:  # Open Short
            if self.position == "flat":
                self.position = "short"
                self.entry_price = current_price
        elif action == 1:  # Close Long
            if self.position == "long":
                ret = (current_price - self.entry_price) / self.entry_price
                reward = ret * leverage
                self.portfolio *= (1 + reward) * (1 - self.fee)
                self.trades += 1
                if reward > 0:
                    self.wins += 1
                self.position = "flat"
                self.entry_price = None
        elif action == 2:  # Do Nothing
            # If holding a position, add a small holding reward based on unrealized PnL
            if self.position != "flat" and self.entry_price is not None:
                if self.position == "long":
                    holding_ret = (current_price - self.entry_price) / self.entry_price
                elif self.position == "short":
                    holding_ret = (self.entry_price - current_price) / self.entry_price
                reward += self.holding_reward_coef * holding_ret
        elif action == 3:  # Close Short
            if self.position == "short":
                ret = (self.entry_price - current_price) / self.entry_price
                reward = ret * leverage
                self.portfolio *= (1 + reward) * (1 - self.fee)
                self.trades += 1
                if reward > 0:
                    self.wins += 1
                self.position = "flat"
                self.entry_price = None
        elif action == 4:  # Open Long
            if self.position == "flat":
                self.position = "long"
                self.entry_price = current_price

        # Check for liquidation conditions
        if self.position == "long":
            liq_price = self.entry_price * (1 - (1.0 / leverage) * safety_margin)
            if next_low < liq_price:
                reward = -1.0
                self.portfolio = 0.0
                self.liquidations += 1
                self.position = "flat"
                self.entry_price = None
        elif self.position == "short":
            liq_price = self.entry_price * (1 + (1.0 / leverage) * safety_margin)
            if next_high > liq_price:
                reward = -1.0
                self.portfolio = 0.0
                self.liquidations += 1
                self.position = "flat"
                self.entry_price = None

        self.t += 1
        self.portfolio_history.append(self.portfolio)
        # Update peak portfolio value for drawdown tracking
        self.peak = max(self.peak, self.portfolio)
        done = (self.t >= (self.seq_length + self.sim_steps)) or (self.portfolio <= 0.0)
        # Enforce maximum drawdown limit if specified
        if not done and self.max_drawdown_limit is not None and self.peak > 0:
            current_drawdown = (self.peak - self.portfolio) / self.peak
            if current_drawdown > self.max_drawdown_limit:
                reward = -1.0
                done = True
        # Clip the reward to be between -1 and 1
        reward = max(-1.0, min(reward, 1.0))
        next_state = None if done else self._get_state()
        return next_state, reward, done, {"portfolio": self.portfolio}
    
    def compute_metrics(self):
        history = np.array(self.portfolio_history)
        if len(history) < 2:
            return 0.0, 0.0
        returns = np.diff(history) / history[:-1]
        sharpe = returns.mean() / (returns.std() + 1e-6)
        peak = np.maximum.accumulate(history)
        drawdowns = (peak - history) / peak
        max_drawdown = drawdowns.max()
        return sharpe, max_drawdown

##############################################
# Improved Reward Function
##############################################
def improved_reward_function(ret, leverage, pred_std, action_entropy, params):
    # Scale base reward and clip
    base_reward = np.clip(ret * leverage, -1.0, 1.0)
    
    # Smoother risk adjustment using tanh
    risk_adj = np.tanh(abs(ret) / (pred_std + 1e-6))
    
    # Progressive leverage penalty
    leverage_penalty = params['leverage_penalty'] * np.tanh((leverage / 50.0) ** 2)
    
    # Entropy bonus for exploration
    entropy_bonus = params['entropy_coef'] * action_entropy
    
    # Combined reward with better scaling
    total_reward = base_reward + params['risk_coef'] * risk_adj + entropy_bonus - leverage_penalty
    
    return np.clip(total_reward, -1.0, 1.0)

##############################################
# 3. Strategy Network Training with Improvements
##############################################
def train_strategy_network_vectorized(episodes, features, pred_net, num_epochs=50, seq_length=72,
                                      gamma=0.99, lr=0.0001, batch_size=20, checkpoint_path="strat_net.pt",
                                      resume=False, infinite=False, best_checkpoint_path="best_strat_net.pt",
                                      rolling_window_size=None, use_volume=False, strat_net=None):
    print(f"[{datetime.now()}] Starting vectorized training of Strategy Network with improved sampling...")
    if strat_net is None:
        if use_volume:
            strat_net = VolumeAwareStrategyNet(ImprovedStrategyNetwork(input_size=3).to(device))
            strat_net = strat_net.to(device)
        else:
            strat_net = ImprovedStrategyNetwork(input_size=3).to(device)
    optimizer = optim.Adam(strat_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    start_epoch = 0
    best_reward = -float('inf')
    if resume and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if use_volume:
                checkpoint["model_state_dict"] = fix_volume_checkpoint(checkpoint["model_state_dict"])
            strat_net.load_state_dict(checkpoint["model_state_dict"], strict=False)
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: optimizer state dict load failed: {e}")
            start_epoch = checkpoint["epoch"]
            best_reward = checkpoint.get("best_reward", best_reward)
            print(f"[{datetime.now()}] Resumed Strategy Network from epoch {start_epoch}")
            strat_net = strat_net.to(device)
        except Exception as e:
            print(f"Error loading strategy checkpoint: {e}")
    if rolling_window_size is not None:
        episodes = episodes[-rolling_window_size:]
    replay_buffer = deque(maxlen=10000)
    for ep in episodes:
        replay_buffer.append(ep)
    
    reward_params = {
        'entropy_coef': 0.005,
        'leverage_penalty': 0.005,
        'risk_coef': 0.1,
    }
    
    epoch = start_epoch
    while True:
        epoch += 1
        current_subset_size = min(len(replay_buffer), FIXED_SUBSET_SIZE)
        episodes_subset = random.sample(list(replay_buffer), current_subset_size)
        random.shuffle(episodes_subset)
        num_batches = current_subset_size // batch_size
        total_loss = 0.0
        total_reward = 0.0
        for _ in range(num_batches):
            batch_episodes = sample_stratified(episodes_subset, batch_size)
            price_close = np.stack([ep["close"].values for ep in batch_episodes], axis=0)
            if "high" in batch_episodes[0].columns and "low" in batch_episodes[0].columns:
                price_high = np.stack([ep["high"].values for ep in batch_episodes], axis=0)
                price_low = np.stack([ep["low"].values for ep in batch_episodes], axis=0)
            else:
                price_high = price_close
                price_low = price_close
            B, L = price_close.shape
            sim_steps = min(L - seq_length, 288)
            portfolios = np.full((B,), 100.0, dtype=np.float32)
            positions = np.array(["flat"] * B)
            entry_prices = np.zeros(B, dtype=np.float32)
            pred_std_batch = [1.0] * B
            log_probs_list = []
            rewards_list = []
            for t in range(seq_length, seq_length + sim_steps):
                states = []
                pred_std_list = []
                for b in range(B):
                    window = price_close[b, t - seq_length:t]
                    returns = (window[1:] - window[:-1]) / window[:-1]
                    returns = returns[np.newaxis, :, np.newaxis]
                    with torch.no_grad():
                        pred_mean, pred_log_std = pred_net(torch.FloatTensor(returns).to(device))
                    pred_std_val = torch.exp(pred_log_std).item()
                    pred_std_list.append(pred_std_val)
                    pos = 0.0
                    if positions[b] == "long":
                        pos = 1.0
                    elif positions[b] == "short":
                        pos = -1.0
                    state = torch.tensor([pred_mean.item(), pred_std_val, pos],
                                         dtype=torch.float32, device=device)
                    states.append(state)
                pred_std_batch = pred_std_list
                state_tensor = torch.stack(states)
                if use_volume:
                    action_logits, lev = strat_net(state_tensor, volume=None)
                else:
                    action_logits, lev = strat_net(state_tensor, pred_std=torch.tensor(pred_std_batch, dtype=torch.float32, device=device))
                dist = torch.distributions.Categorical(logits=action_logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                entropies = dist.entropy()
                step_rewards = []
                for b in range(B):
                    current_price = price_close[b, t]
                    r = 0.0
                    safety_margin = 1.1
                    lvr = max(1.0, min(lev[b].item(), 100.0))
                    if positions[b] == "long":
                        liq_price = entry_prices[b] * (1 - (1.0 / lvr) * safety_margin)
                        if price_low[b, t] < liq_price:
                            r = -1.0
                            portfolios[b] = 0.0
                            positions[b] = "flat"
                            entry_prices[b] = 0.0
                    elif positions[b] == "short":
                        liq_price = entry_prices[b] * (1 + (1.0 / lvr) * safety_margin)
                        if price_high[b, t] > liq_price:
                            r = -1.0
                            portfolios[b] = 0.0
                            positions[b] = "flat"
                            entry_prices[b] = 0.0
                    act = actions[b].item()
                    if act == 0 and positions[b] == "flat":
                        positions[b] = "short"
                        entry_prices[b] = current_price
                    elif act == 1 and positions[b] == "long":
                        ret = (current_price - entry_prices[b]) / entry_prices[b]
                        entropy_val = entropies[b].item()
                        r = improved_reward_function(ret, lvr, pred_std_batch[b], entropy_val, reward_params)
                        portfolios[b] *= (1 + r) * (1 - 0.001)
                        positions[b] = "flat"
                        entry_prices[b] = 0.0
                    elif act == 3 and positions[b] == "short":
                        ret = (entry_prices[b] - current_price) / entry_prices[b]
                        entropy_val = entropies[b].item()
                        r = improved_reward_function(ret, lvr, pred_std_batch[b], entropy_val, reward_params)
                        portfolios[b] *= (1 + r) * (1 - 0.001)
                        positions[b] = "flat"
                        entry_prices[b] = 0.0
                    elif act == 4 and positions[b] == "flat":
                        positions[b] = "long"
                        entry_prices[b] = current_price
                    step_rewards.append(r)
                rewards_list.append(torch.tensor(step_rewards, dtype=torch.float32, device=device))
                log_probs_list.append(log_probs)
            if len(log_probs_list) == 0:
                continue
            log_probs_tensor = torch.stack(log_probs_list, dim=1)
            rewards_tensor = torch.stack(rewards_list, dim=1)
            discounted_returns = torch.zeros_like(rewards_tensor)
            for b in range(B):
                R = 0.0
                for t in reversed(range(rewards_tensor.shape[1])):
                    R = rewards_tensor[b, t] + gamma * R
                    discounted_returns[b, t] = R
            returns_mean = discounted_returns.mean()
            returns_std = discounted_returns.std() + 1e-9
            discounted_returns = (discounted_returns - returns_mean) / returns_std
            policy_loss = - (log_probs_tensor * discounted_returns).mean()
            optimizer.zero_grad()
            policy_loss.backward()
            # Apply gradient clipping for strategy network training
            torch.nn.utils.clip_grad_norm_(strat_net.parameters(), max_norm=1.0)
            optimizer.step()
            batch_reward = np.mean(portfolios - 100.0)
            total_loss += policy_loss.item()
            total_reward += batch_reward
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_reward = total_reward / num_batches if num_batches > 0 else 0.0
        print(f"[{datetime.now()}] Strategy Epoch {epoch} | Loss: {avg_loss:.4f} | Avg Reward: {avg_reward:.2f}")
        scheduler.step(avg_reward)
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": strat_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "best_reward": best_reward,
        }
        torch.save(checkpoint_dict, checkpoint_path)
        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint_dict["best_reward"] = best_reward
            torch.save(checkpoint_dict, best_checkpoint_path)
            print(f"[{datetime.now()}] New best Strategy Network saved to {best_checkpoint_path}")
        if not infinite and epoch >= start_epoch + num_epochs:
            break
    return strat_net

def train_strategy_network(episodes, features, pred_net, num_epochs=50, seq_length=72,
                           gamma=0.99, lr=0.0001, batch_size=20, checkpoint_path="strat_net.pt",
                           resume=False, infinite=False, best_checkpoint_path="best_strat_net.pt",
                           rolling_window_size=None, use_volume=False, strat_net=None):
    print(f"[{datetime.now()}] Starting non–vectorized training of Strategy Network with improved sampling...")
    if strat_net is None:
        if use_volume:
            strat_net = VolumeAwareStrategyNet(ImprovedStrategyNetwork(input_size=3).to(device))
            strat_net = strat_net.to(device)
        else:
            strat_net = ImprovedStrategyNetwork(input_size=3).to(device)
    optimizer = optim.Adam(strat_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)
    start_epoch = 0
    best_reward = -float('inf')
    if resume and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if use_volume:
                checkpoint["model_state_dict"] = fix_volume_checkpoint(checkpoint["model_state_dict"])
            strat_net.load_state_dict(checkpoint["model_state_dict"], strict=False)
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: optimizer state dict load failed: {e}")
            start_epoch = checkpoint["epoch"]
            best_reward = checkpoint.get("best_reward", best_reward)
            print(f"[{datetime.now()}] Resumed non–vectorized Strategy Network from epoch {start_epoch}")
            strat_net = strat_net.to(device)
        except Exception as e:
            print(f"Error loading strategy checkpoint: {e}")
    replay_buffer = deque(maxlen=10000)
    if rolling_window_size is not None:
        episodes = episodes[-rolling_window_size:]
    for ep in episodes:
        replay_buffer.append(ep)
    
    reward_params = {
        'entropy_coef': 0.001,
        'leverage_penalty': 0.01,
        'risk_coef': 0.1,
    }
    
    epoch = start_epoch
    while True:
        epoch += 1
        current_subset_size = min(len(replay_buffer), FIXED_SUBSET_SIZE)
        current_subset = random.sample(list(replay_buffer), current_subset_size)
        total_loss = 0.0
        total_reward = 0.0
        total_trades = 0
        total_wins = 0
        total_liquidations = 0
        num_eps = 0
        for episode_df in current_subset:
            if use_volume:
                base_env = TradingEnvStrategy(episode_df, features, pred_net, seq_length=seq_length)
                env = VolumeAwareEnvWrapper(base_env)
                state, volume = env.reset()
            else:
                env = TradingEnvStrategy(episode_df, features, pred_net, seq_length=seq_length)
                state = env.reset()
                volume = None
            log_probs = []
            rewards = []
            while True:
                if use_volume:
                    action_logits, lev = strat_net(state, volume=volume)
                else:
                    pred_std_val = state[0, 1].unsqueeze(0)
                    action_logits, lev = strat_net(state, pred_std=pred_std_val)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)
                if use_volume:
                    next_state, next_volume, reward, done, info = env.step(action.item(), lev.item())
                else:
                    next_state, reward, done, info = env.step(action.item(), lev.item())
                    next_volume = None
                rewards.append(reward)
                if done:
                    break
                state = next_state
                volume = next_volume
            profit_pct = (env.portfolio / env.initial_cash) - 1
            sharpe, max_drawdown = env.compute_metrics()
            print(f"Episode {num_eps+1}: Final Portfolio: {env.portfolio:.2f} ({profit_pct*100:.2f}%), "
                  f"Position: {env.position}, Trades: {env.trades}, Wins: {env.wins}, "
                  f"Liquidations: {env.liquidations}, Sharpe: {sharpe:.2f}, Max Drawdown: {max_drawdown*100:.2f}%")
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.FloatTensor(returns).to(device)
            if returns.std() > 0:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            loss = - sum(lp * R for lp, R in zip(log_probs, returns))
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping added here for non–vectorized training
            torch.nn.utils.clip_grad_norm_(strat_net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_reward += profit_pct
            total_trades += env.trades
            total_wins += env.wins
            total_liquidations += env.liquidations
            num_eps += 1
        avg_loss = total_loss / num_eps if num_eps > 0 else 0.0
        avg_reward = total_reward / num_eps if num_eps > 0 else 0.0
        win_ratio = total_wins / total_trades if total_trades > 0 else 0.0
        print(f"[{datetime.now()}] Non–vectorized Strategy Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"Avg Profit: {avg_reward*100:.2f}% | Trades/Episode: {total_trades/num_eps:.1f} | "
              f"Win Ratio: {win_ratio:.2f} | Liquidations/Episode: {total_liquidations/num_eps:.1f}")
        scheduler.step(avg_reward)
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": strat_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "best_reward": best_reward,
        }
        torch.save(checkpoint_dict, checkpoint_path)
        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint_dict["best_reward"] = best_reward
            torch.save(checkpoint_dict, best_checkpoint_path)
            print(f"[{datetime.now()}] New best non–vectorized Strategy Network saved to {best_checkpoint_path}")
        if not infinite and epoch >= start_epoch + num_epochs:
            break
    return strat_net

##############################################
# 4. Volume-Aware Wrappers
##############################################
class VolumeAwarePredictionNet(nn.Module):
    def __init__(self, base_net):
        super().__init__()
        self.base_net = base_net
        self.volume_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x, volume=None):
        mean, log_std = self.base_net(x)
        if volume is not None:
            if not torch.is_tensor(volume):
                volume = torch.tensor([volume], dtype=torch.float32, device=x.device)
            elif volume.dim() == 1:
                volume = volume.unsqueeze(1)
            vol_effect = self.volume_net(volume)
            log_std = log_std + vol_effect.squeeze(-1)
        return mean, log_std

class VolumeAwareStrategyNet(nn.Module):
    def __init__(self, base_net):
        super().__init__()
        self.base_net = base_net
        self.volume_impact = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.leverage_adjust = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, state, pred_std=None, volume=None):
        q_values, leverage = self.base_net(state, pred_std)
        if volume is not None:
            if not torch.is_tensor(volume):
                volume = torch.tensor([volume], dtype=torch.float32, device=state.device)
            elif volume.dim() == 1:
                volume = volume.unsqueeze(1)
            impact = self.volume_impact(volume)
            adjust = self.leverage_adjust(volume)
            q_values = q_values * (1 + torch.sigmoid(impact))
            leverage = leverage * (1 + torch.sigmoid(adjust))
        return q_values, leverage

class VolumeAwareEnvWrapper:
    def __init__(self, base_env):
        self.base_env = base_env
        self.has_volume = "volume" in base_env.episode.columns

    def __getattr__(self, name):
        return getattr(self.base_env, name)

    def reset(self):
        state = self.base_env.reset()
        volume = None
        if self.has_volume:
            volume = self.base_env.episode["volume"].iloc[self.base_env.t]
            volume = torch.tensor([volume], dtype=torch.float32, device=device)
        return state, volume

    def step(self, action, leverage):
        next_state, reward, done, info = self.base_env.step(action, leverage)
        volume = None
        if not done and self.has_volume:
            volume = self.base_env.episode["volume"].iloc[self.base_env.t]
            volume = torch.tensor([volume], dtype=torch.float32, device=device)
        return next_state, volume, reward, done, info

def wrap_models_with_volume(pred_checkpoint, strat_checkpoint):
    pred_net = PredictionNetwork(input_size=1).to(device)
    try:
        checkpoint = torch.load(pred_checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = fix_prediction_checkpoint(checkpoint["model_state_dict"])
            pred_net.load_state_dict(state_dict)
        else:
            pred_net.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading Prediction Network: {e}")
        exit(1)
    pred_net.eval()
    volume_pred_net = VolumeAwarePredictionNet(pred_net)
    volume_pred_net = volume_pred_net.to(device)

    strat_net = ImprovedStrategyNetwork(input_size=3).to(device)
    try:
        checkpoint = torch.load(strat_checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint["model_state_dict"] = fix_volume_checkpoint(checkpoint["model_state_dict"])
            strat_net.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            strat_net.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading Strategy Network: {e}")
        exit(1)
    strat_net.eval()
    volume_strat_net = VolumeAwareStrategyNet(strat_net)
    volume_strat_net = volume_strat_net.to(device)
    return volume_pred_net, volume_strat_net

##############################################
# 5. Unified Interface for Training Both Networks
##############################################
def train_dual_networks(data_csv="BTCUSDT_5m_clean.csv", seq_length=72,
                        pred_epochs=10, strat_epochs=50,
                        pred_batch_size=64, strat_batch_size=20,
                        pred_lr=0.0001, strat_lr=0.0001,
                        pred_checkpoint="pred_net.pt", strat_checkpoint="strat_net.pt",
                        vectorized=False, resume=False, infinite=False,
                        rolling_window_size=None, use_volume=False):
    print(f"[{datetime.now()}] Starting dual network training...")
    pred_net = train_prediction_network(data_csv, seq_length=seq_length, num_epochs=pred_epochs,
                                        batch_size=pred_batch_size, lr=pred_lr,
                                        checkpoint_path=pred_checkpoint, resume=resume,
                                        best_checkpoint_path="best_pred_net.pt", infinite=infinite)
    episodes, features, scaler = load_and_preprocess_data(data_csv, seq_length=seq_length)
    strat_net = None
    if use_volume:
        base_strat_net = ImprovedStrategyNetwork(input_size=3).to(device)
        try:
            checkpoint = torch.load(strat_checkpoint, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                checkpoint["model_state_dict"] = fix_volume_checkpoint(checkpoint["model_state_dict"])
                base_strat_net.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                base_strat_net.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading strategy checkpoint: {e}")
            exit(1)
        base_strat_net.eval()
        strat_net = VolumeAwareStrategyNet(base_strat_net)
        strat_net = strat_net.to(device)
    if vectorized:
        strat_net = train_strategy_network_vectorized(
            episodes, features, pred_net, num_epochs=strat_epochs, seq_length=seq_length,
            gamma=0.99, lr=strat_lr, batch_size=strat_batch_size, checkpoint_path=strat_checkpoint,
            resume=resume, infinite=infinite, best_checkpoint_path="best_strat_net.pt",
            rolling_window_size=rolling_window_size, use_volume=use_volume, strat_net=strat_net
        )
    else:
        strat_net = train_strategy_network(
            episodes, features, pred_net, num_epochs=strat_epochs, seq_length=seq_length,
            gamma=0.99, lr=strat_lr, batch_size=strat_batch_size, checkpoint_path=strat_checkpoint,
            resume=resume, infinite=infinite, best_checkpoint_path="best_strat_net.pt",
            rolling_window_size=rolling_window_size, use_volume=use_volume, strat_net=strat_net
        )
    print(f"[{datetime.now()}] Dual network training complete.")
    return pred_net, strat_net

##############################################
# 6. Interactive Command–Line Interface
##############################################
def main():
    parser = argparse.ArgumentParser(
        description="Dual Network Trading System (Prediction & Improved Strategy Networks)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("help", help="Show this help message")
    subparsers.add_parser("clear", help="Clear the console")
    
    gen_clean_parser = subparsers.add_parser("generate_clean", help="Generate clean data from raw data")
    gen_clean_parser.add_argument("--raw_csv", type=str, required=True, help="Path to raw CSV data")
    gen_clean_parser.add_argument("--clean_csv", type=str, default="clean_data.csv", help="Output path for clean data")
    
    train_dual_parser = subparsers.add_parser("train_dual", help="Train both networks")
    train_dual_parser.add_argument("--data_csv", type=str, default="BTCUSDT_5m_clean.csv", help="Path to CSV data file")
    train_dual_parser.add_argument("--seq_length", type=int, default=72, help="History sequence length")
    train_dual_parser.add_argument("--pred_epochs", type=int, default=10, help="Epochs for prediction network")
    train_dual_parser.add_argument("--strat_epochs", type=int, default=50, help="Epochs for strategy network")
    train_dual_parser.add_argument("--pred_batch_size", type=int, default=64)
    train_dual_parser.add_argument("--strat_batch_size", type=int, default=20)
    train_dual_parser.add_argument("--pred_lr", type=float, default=0.0001)
    train_dual_parser.add_argument("--strat_lr", type=float, default=0.0001)
    train_dual_parser.add_argument("--pred_checkpoint", type=str, default="pred_net.pt")
    train_dual_parser.add_argument("--strat_checkpoint", type=str, default="strat_net.pt")
    train_dual_parser.add_argument("--vectorized", action="store_true", help="Use vectorized simulation for strategy training")
    train_dual_parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoints")
    train_dual_parser.add_argument("--infinite", action="store_true", help="Train indefinitely until Ctrl-C is pressed")
    train_dual_parser.add_argument("--rolling_window_size", type=int, default=None, help="If set, use only the most recent N episodes")
    train_dual_parser.add_argument("--use_volume", action="store_true", help="Use volume-aware wrappers if volume data is available")
    
    train_pred_parser = subparsers.add_parser("train_pred", help="Train the prediction network only")
    train_pred_parser.add_argument("--data_csv", type=str, default="BTCUSDT_5m_clean.csv")
    train_pred_parser.add_argument("--seq_length", type=int, default=72)
    train_pred_parser.add_argument("--pred_epochs", type=int, default=10)
    train_pred_parser.add_argument("--pred_batch_size", type=int, default=64)
    train_pred_parser.add_argument("--pred_lr", type=float, default=0.0001)
    train_pred_parser.add_argument("--pred_checkpoint", type=str, default="pred_net.pt")
    train_pred_parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    train_pred_parser.add_argument("--infinite", action="store_true", help="Train indefinitely until Ctrl-C is pressed")
    train_pred_parser.add_argument("--use_volume", action="store_true", help="Use volume-aware wrappers if volume data is available")
    
    train_strat_parser = subparsers.add_parser("train_strat", help="Train the strategy network only")
    train_strat_parser.add_argument("--data_csv", type=str, default="BTCUSDT_5m_clean.csv")
    train_strat_parser.add_argument("--seq_length", type=int, default=72)
    train_strat_parser.add_argument("--strat_epochs", type=int, default=50)
    train_strat_parser.add_argument("--strat_batch_size", type=int, default=20)
    train_strat_parser.add_argument("--strat_lr", type=float, default=0.0001)
    train_strat_parser.add_argument("--strat_checkpoint", type=str, default="strat_net.pt")
    train_strat_parser.add_argument("--pred_checkpoint", type=str, default="pred_net.pt")
    train_strat_parser.add_argument("--vectorized", action="store_true", help="Use vectorized simulation for strategy training")
    train_strat_parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    train_strat_parser.add_argument("--infinite", action="store_true", help="Train indefinitely until Ctrl-C is pressed")
    train_strat_parser.add_argument("--rolling_window_size", type=int, default=None, help="If set, use only the most recent N episodes")
    train_strat_parser.add_argument("--use_volume", action="store_true", help="Use volume-aware wrappers if volume data is available")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
    elif args.command == "help":
        parser.print_help()
    elif args.command == "clear":
        clear_console()
    elif args.command == "generate_clean":
        generate_clean_data(args.raw_csv, args.clean_csv)
    elif args.command == "train_dual":
        train_dual_networks(
            data_csv=args.data_csv,
            seq_length=args.seq_length,
            pred_epochs=args.pred_epochs,
            strat_epochs=args.strat_epochs,
            pred_batch_size=args.pred_batch_size,
            strat_batch_size=args.strat_batch_size,
            pred_lr=args.pred_lr,
            strat_lr=args.strat_lr,
            pred_checkpoint=args.pred_checkpoint,
            strat_checkpoint=args.strat_checkpoint,
            vectorized=args.vectorized,
            resume=args.resume,
            infinite=args.infinite,
            rolling_window_size=args.rolling_window_size,
            use_volume=args.use_volume
        )
    elif args.command == "train_pred":
        pred_net = train_prediction_network(
            csv_path=args.data_csv,
            seq_length=args.seq_length,
            num_epochs=args.pred_epochs,
            batch_size=args.pred_batch_size,
            lr=args.pred_lr,
            checkpoint_path=args.pred_checkpoint,
            resume=args.resume,
            infinite=args.infinite,
            best_checkpoint_path="best_pred_net.pt"
        )
        if getattr(args, "use_volume", False):
            pred_net = VolumeAwarePredictionNet(pred_net)
            print(f"[{datetime.now()}] Prediction network wrapped with volume-aware layers.")
    elif args.command == "train_strat":
        pred_net = PredictionNetwork(input_size=1).to(device)
        try:
            checkpoint = torch.load(args.pred_checkpoint, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = fix_prediction_checkpoint(checkpoint["model_state_dict"])
                pred_net.load_state_dict(state_dict)
            else:
                pred_net.load_state_dict(checkpoint)
            pred_net.eval()
            print(f"[{datetime.now()}] Loaded Prediction Network from {args.pred_checkpoint}")
        except Exception as e:
            print(f"Error loading Prediction Network from {args.pred_checkpoint}: {e}")
            exit(1)

        episodes, features, scaler = load_and_preprocess_data(args.data_csv, seq_length=args.seq_length)
        strat_net = None
        if args.use_volume:
            base_strat_net = ImprovedStrategyNetwork(input_size=3).to(device)
            try:
                checkpoint = torch.load(args.strat_checkpoint, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    checkpoint["model_state_dict"] = fix_volume_checkpoint(checkpoint["model_state_dict"])
                    base_strat_net.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    base_strat_net.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Error loading strategy checkpoint: {e}")
                exit(1)
            base_strat_net.eval()
            strat_net = VolumeAwareStrategyNet(base_strat_net)
            strat_net = strat_net.to(device)
        if args.vectorized:
            train_strategy_network_vectorized(
                episodes, features, pred_net,
                num_epochs=args.strat_epochs,
                seq_length=args.seq_length,
                gamma=0.99,
                lr=args.strat_lr,
                batch_size=args.strat_batch_size,
                checkpoint_path=args.strat_checkpoint,
                resume=args.resume,
                infinite=args.infinite,
                best_checkpoint_path="best_strat_net.pt",
                rolling_window_size=args.rolling_window_size,
                use_volume=args.use_volume,
                strat_net=strat_net
            )
        else:
            train_strategy_network(
                episodes, features, pred_net,
                num_epochs=args.strat_epochs,
                seq_length=args.seq_length,
                gamma=0.99,
                lr=args.strat_lr,
                batch_size=args.strat_batch_size,
                checkpoint_path=args.strat_checkpoint,
                resume=args.resume,
                infinite=args.infinite,
                best_checkpoint_path="best_strat_net.pt",
                rolling_window_size=args.rolling_window_size,
                use_volume=args.use_volume,
                strat_net=strat_net
            )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
