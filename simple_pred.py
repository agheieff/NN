#!/usr/bin/env python3
"""
Reworked Prediction and Strategy Networks

- The PredictionNetwork outputs a direction probability and a confidence value.
  It is trained with a standard binary cross–entropy loss for direction and a simple
  mean–squared error loss for confidence. The target confidence is defined as:
      target_conf = y * p_detach + (1 - y) * (1 - p_detach)
- The StrategyNetwork receives the prediction outputs (direction and confidence)
  and outputs a continuous position (between –max_leverage and +max_leverage).
- The strategy is trained end–to–end by simulating a trading episode and using
  a loss based solely on the (negative) final balance.
- Unnecessary parts and overly–complicated math have been removed.
"""

import os
import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------------------------ Data Loading and Preprocessing ------------------------

def load_and_preprocess(csv_path, window_size=288, max_samples=500000):
    """
    Load CSV data and create samples using a sliding window.
    For each sample (window of candles):
      - Normalize open, high, low, close by dividing by the last candle's close and subtracting 1.
      - Normalize volume by the maximum volume in the window.
      - Flatten the (open, high, low, close, volume) features.
      - Append two global features: weekday sine and cosine (from the timestamp of the last candle, if available).
      - The target is 1 if the candle immediately after the window has a close greater than the window's last close.
    Also returns raw prices and volumes for strategy simulation.
    Note: raw_prices (and volumes) are returned with one extra element so that
          for N samples (X), prices has length N+1.
    """
    print(f"[{datetime.now()}] Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    num_samples = total_rows - window_size
    if num_samples > max_samples:
        start_idx = np.random.randint(0, num_samples - max_samples)
        df = df.iloc[start_idx:start_idx + max_samples + window_size].reset_index(drop=True)
        num_samples = max_samples

    features_list = []
    targets_list = []

    # Determine which timestamp column to use, if any.
    if 'timestamp' in df.columns:
        time_col = 'timestamp'
        time_unit = None
    elif 'open_time' in df.columns:
        time_col = 'open_time'
        time_unit = 'ms'
    else:
        time_col = None

    for i in range(num_samples):
        window = df.iloc[i:i+window_size].copy()
        target_row = df.iloc[i+window_size]
        baseline = window['close'].iloc[-1]

        # Normalize prices in the window relative to the baseline.
        for col in ['open', 'high', 'low', 'close']:
            window[col] = window[col] / baseline - 1

        # Normalize volume by the maximum in the window.
        max_vol = window['volume'].max() + 1e-6
        window['volume'] = window['volume'] / max_vol

        # Flatten the candle features (order: open, high, low, close, volume).
        candle_features = window[['open', 'high', 'low', 'close', 'volume']].values.flatten()

        # Global features: weekday sine and cosine.
        if time_col is not None:
            ts = pd.to_datetime(window.iloc[-1][time_col], unit=time_unit)
            weekday = ts.weekday()  # Monday=0, Sunday=6.
            weekday_sin = math.sin(2 * math.pi * weekday / 7)
            weekday_cos = math.cos(2 * math.pi * weekday / 7)
        else:
            weekday_sin, weekday_cos = 0.0, 0.0

        global_features = np.array([weekday_sin, weekday_cos], dtype=np.float32)
        features = np.concatenate([candle_features, global_features])
        features_list.append(features.astype(np.float32))

        # Binary target: 1 if next candle's close > baseline.
        target = 1.0 if target_row['close'] > baseline else 0.0
        targets_list.append(np.float32(target))

    X = torch.FloatTensor(np.stack(features_list))
    y = torch.FloatTensor(np.array(targets_list)).unsqueeze(1)

    # For strategy simulation, use the raw close prices and volumes.
    # We return one extra element so that for each sample there is a subsequent price.
    raw_prices = torch.FloatTensor(df['close'].values[window_size - 1:])
    raw_volumes = torch.FloatTensor(df['volume'].values[window_size - 1:])

    print(f"[{datetime.now()}] Loaded {len(X)} samples with {X.shape[1]} features each")
    return X, y, raw_prices, raw_volumes

# ------------------------ Prediction Network ------------------------

class PredictionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(PredictionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # Two heads: one for direction probability and one for confidence.
        self.fc_direction = nn.Linear(hidden_dim // 2, 1)
        self.fc_confidence = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Direction: probability between 0 and 1.
        p = torch.sigmoid(self.fc_direction(x))
        # Confidence: value between 0 and 1.
        c = torch.sigmoid(self.fc_confidence(x))
        return p, c

def prediction_loss(p, c, y, lambda_conf=1.0):
    """
    Loss for the prediction network:
      - Binary cross–entropy for direction prediction.
      - Mean–squared error for confidence, where the target confidence is defined as:
          target_conf = y * p_detach + (1 - y) * (1 - p_detach)
    """
    bce_loss = F.binary_cross_entropy(p, y)
    with torch.no_grad():
        target_conf = y * p.detach() + (1 - y) * (1 - p.detach())
    mse_loss = F.mse_loss(c, target_conf)
    return bce_loss + lambda_conf * mse_loss

def train_prediction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, _, _ = load_and_preprocess(args.train_file, window_size=args.window_size, max_samples=args.max_samples)
    input_dim = X.shape[1]
    model = PredictionNetwork(input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_acc = 0.0
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Loaded checkpoint from epoch {start_epoch}, best accuracy {best_acc:.2f}%")

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        X_batch = X.to(device)
        y_batch = y.to(device)
        p, c = model(X_batch)
        loss = prediction_loss(p, c, y_batch)
        loss.backward()
        optimizer.step()

        preds = (p > 0.5).float()
        acc = (preds == y_batch).float().mean().item() * 100

        saved_str = ""
        if acc > best_acc:
            best_acc = acc
            if args.checkpoint:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'hidden_dim': args.hidden_dim,
                    'input_dim': input_dim
                }, args.checkpoint)
                saved_str = " (saved)"
        if epoch % args.print_every == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2f}%{saved_str}")
    print("Prediction network training completed.")

def evaluate_prediction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, _, _ = load_and_preprocess(args.val_file, window_size=args.window_size, max_samples=args.max_samples)
    input_dim = X.shape[1]
    model = PredictionNetwork(input_dim, hidden_dim=args.hidden_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        p, c = model(X)
        preds = (p > 0.5).float()
        acc = (preds == y).float().mean().item() * 100
    print(f"Validation Accuracy: {acc:.2f}%")
    confidences = c.cpu().numpy().flatten()
    print("Confidence stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(confidences.min(), confidences.max(), confidences.mean()))

# ------------------------ Strategy Network ------------------------

class StrategyNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, max_leverage=1.0):
        """
        Simple strategy network that takes in the prediction outputs (direction and confidence)
        and outputs a continuous position signal (scaled by max_leverage).
        """
        super(StrategyNetwork, self).__init__()
        self.max_leverage = max_leverage
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Ensures output is between -1 and 1.
        )

    def forward(self, pred_features):
        return self.max_leverage * self.net(pred_features)

def run_strategy_episode(pred_model, strat_model, X, prices, initial_balance=100.0):
    """
    Simulate a trading episode over the sequential data.
    For each time step t, the strategy receives the prediction network's outputs for X[t]
    and computes a position. The balance is updated as:
         balance *= (1 + position * return)
    where return = (price[t+1] - price[t]) / price[t]
    """
    device = prices.device
    balance = torch.tensor(initial_balance, device=device, dtype=torch.float32)
    balance_history = [balance]
    T = X.shape[0]
    for t in range(T - 1):
        x_t = X[t:t+1]  # Shape: (1, feature_dim)
        with torch.no_grad():
            p, c = pred_model(x_t)  # p, c shape: (1, 1)
        pred_features = torch.cat([p, c], dim=1)  # Shape: (1, 2)
        position = strat_model(pred_features)      # Shape: (1, 1)
        price_t = prices[t]
        price_next = prices[t + 1]
        ret = (price_next - price_t) / price_t
        balance = balance * (1 + position.squeeze() * ret)
        balance_history.append(balance)
    return balance, balance_history

def train_strategy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load training data.
    X, _, raw_prices, _ = load_and_preprocess(args.train_file, window_size=args.window_size, max_samples=args.max_samples)
    X = X.to(device)
    raw_prices = raw_prices.to(device)

    # Load fixed prediction model.
    pred_checkpoint = torch.load(args.pred_checkpoint, map_location=device)
    input_dim = X.shape[1]
    pred_hidden_dim = pred_checkpoint.get('hidden_dim', 128)
    pred_model = PredictionNetwork(input_dim, hidden_dim=pred_hidden_dim).to(device)
    pred_model.load_state_dict(pred_checkpoint['model_state_dict'])
    pred_model.eval()  # Freeze prediction network during strategy training.

    # Initialize strategy network.
    strat_model = StrategyNetwork(input_dim=2, hidden_dim=args.strat_hidden_dim, max_leverage=args.max_leverage).to(device)
    optimizer = optim.Adam(strat_model.parameters(), lr=args.lr)

    start_epoch = 0
    best_balance = 0.0
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        strat_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_balance = checkpoint.get('best_balance', 0.0)
        print(f"Loaded strategy checkpoint from epoch {start_epoch}, best balance {best_balance:.2f}")

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        strat_model.train()
        optimizer.zero_grad()
        total_samples = X.shape[0]
        episode_length = args.episode_length
        if total_samples < episode_length + 1:
            episode_length = total_samples - 1
        start_idx = np.random.randint(0, total_samples - episode_length)
        X_ep = X[start_idx:start_idx + episode_length]
        prices_ep = raw_prices[start_idx:start_idx + episode_length + 1]  # Extra price for return computation.
        final_balance, _ = run_strategy_episode(pred_model, strat_model, X_ep, prices_ep, initial_balance=args.initial_balance)
        loss = -final_balance  # Maximize final balance.
        loss.backward()
        optimizer.step()

        if epoch % args.print_every == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Final Balance={final_balance.item():.2f}")

        if final_balance.item() > best_balance:
            best_balance = final_balance.item()
            if args.checkpoint:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': strat_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_balance': best_balance,
                    'strat_hidden_dim': args.strat_hidden_dim
                }, args.checkpoint)
    print("Strategy training completed.")

def evaluate_strategy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, _, raw_prices, _ = load_and_preprocess(args.val_file, window_size=args.window_size, max_samples=args.max_samples)
    X = X.to(device)
    raw_prices = raw_prices.to(device)

    # Load prediction model.
    pred_checkpoint = torch.load(args.pred_checkpoint, map_location=device)
    input_dim = X.shape[1]
    pred_hidden_dim = pred_checkpoint.get('hidden_dim', 128)
    pred_model = PredictionNetwork(input_dim, hidden_dim=pred_hidden_dim).to(device)
    pred_model.load_state_dict(pred_checkpoint['model_state_dict'])
    pred_model.eval()

    # Load strategy model.
    strat_checkpoint = torch.load(args.checkpoint, map_location=device)
    strat_hidden_dim = strat_checkpoint.get('strat_hidden_dim', 64)
    strat_model = StrategyNetwork(input_dim=2, hidden_dim=strat_hidden_dim, max_leverage=args.max_leverage).to(device)
    strat_model.load_state_dict(strat_checkpoint['model_state_dict'])
    strat_model.eval()

    final_balance, balance_history = run_strategy_episode(pred_model, strat_model, X, raw_prices, initial_balance=args.initial_balance)
    print(f"Strategy Evaluation: Final Balance = {final_balance.item():.2f}")
    return final_balance, balance_history

# ------------------------ Dataset Splitting ------------------------

def split_dataset(args):
    print(f"[{datetime.now()}] Loading CSV data from {args.csv} ...")
    df = pd.read_csv(args.csv)
    total_rows = len(df)
    train_rows = int(total_rows * args.train_percentage / 100)
    train_df = df.iloc[:train_rows]
    val_df = df.iloc[train_rows:]
    train_df.to_csv(args.train_file, index=False)
    val_df.to_csv(args.val_file, index=False)
    print(f"[{datetime.now()}] Dataset split: {len(train_df)} training rows, {len(val_df)} validation rows.")

# ------------------------ Main Entry Point ------------------------

def main():
    parser = argparse.ArgumentParser(description="Reworked Prediction and Strategy Networks")
    subparsers = parser.add_subparsers(dest="command", help="Commands to run")

    # Split command.
    split_parser = subparsers.add_parser("split", help="Split CSV into training and validation sets")
    split_parser.add_argument("--csv", type=str, required=True, help="Path to source CSV file")
    split_parser.add_argument("--train_file", type=str, default="train.csv", help="Output training CSV file")
    split_parser.add_argument("--val_file", type=str, default="val.csv", help="Output validation CSV file")
    split_parser.add_argument("--train_percentage", type=float, default=80, help="Percentage of data for training")

    # Train prediction network.
    train_pred_parser = subparsers.add_parser("train_pred", help="Train the prediction network")
    train_pred_parser.add_argument("--train_file", type=str, default="BTCUSDT_5m_train.csv", help="Training CSV file")
    train_pred_parser.add_argument("--checkpoint", type=str, default="pred_model.pt", help="Checkpoint file for prediction model")
    train_pred_parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    train_pred_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_pred_parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    train_pred_parser.add_argument("--window_size", type=int, default=288, help="Number of candles per input sample")
    train_pred_parser.add_argument("--max_samples", type=int, default=500000, help="Maximum samples to load")
    train_pred_parser.add_argument("--print_every", type=int, default=100, help="Print progress every n epochs")

    # Validate prediction network.
    val_pred_parser = subparsers.add_parser("val_pred", help="Validate the prediction network")
    val_pred_parser.add_argument("--val_file", type=str, default="val.csv", help="Validation CSV file")
    val_pred_parser.add_argument("--checkpoint", type=str, default="pred_model.pt", help="Checkpoint file for prediction model")
    val_pred_parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    val_pred_parser.add_argument("--window_size", type=int, default=288, help="Number of candles per input sample")
    val_pred_parser.add_argument("--max_samples", type=int, default=500000, help="Maximum samples to load")

    # Train strategy network.
    train_strat_parser = subparsers.add_parser("train_strat", help="Train the strategy network")
    train_strat_parser.add_argument("--train_file", type=str, default="train.csv", help="Training CSV file")
    train_strat_parser.add_argument("--pred_checkpoint", type=str, default="pred_model.pt", help="Prediction model checkpoint")
    train_strat_parser.add_argument("--checkpoint", type=str, default="strat_model.pt", help="Checkpoint file for strategy model")
    train_strat_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_strat_parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    train_strat_parser.add_argument("--strat_hidden_dim", type=int, default=64, help="Strategy network hidden dimension size")
    train_strat_parser.add_argument("--window_size", type=int, default=288, help="Number of candles per input sample")
    train_strat_parser.add_argument("--max_samples", type=int, default=500000, help="Maximum samples to load")
    train_strat_parser.add_argument("--episode_length", type=int, default=5000, help="Number of samples per episode")
    train_strat_parser.add_argument("--initial_balance", type=float, default=100.0, help="Initial balance for simulation")
    train_strat_parser.add_argument("--max_leverage", type=float, default=1.0, help="Maximum leverage factor")
    train_strat_parser.add_argument("--print_every", type=int, default=10, help="Print progress every n epochs")

    # Validate strategy network.
    val_strat_parser = subparsers.add_parser("val_strat", help="Validate the strategy network")
    val_strat_parser.add_argument("--val_file", type=str, default="val.csv", help="Validation CSV file")
    val_strat_parser.add_argument("--pred_checkpoint", type=str, default="pred_model.pt", help="Prediction model checkpoint")
    val_strat_parser.add_argument("--checkpoint", type=str, default="strat_model.pt", help="Strategy model checkpoint")
    val_strat_parser.add_argument("--window_size", type=int, default=288, help="Number of candles per input sample")
    val_strat_parser.add_argument("--max_samples", type=int, default=500000, help="Maximum samples to load")
    val_strat_parser.add_argument("--initial_balance", type=float, default=100.0, help="Initial balance for simulation")
    val_strat_parser.add_argument("--max_leverage", type=float, default=1.0, help="Maximum leverage factor")

    args = parser.parse_args()

    if args.command == "split":
        split_dataset(args)
    elif args.command == "train_pred":
        train_prediction(args)
    elif args.command == "val_pred":
        evaluate_prediction(args)
    elif args.command == "train_strat":
        train_strategy(args)
    elif args.command == "val_strat":
        evaluate_strategy(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
