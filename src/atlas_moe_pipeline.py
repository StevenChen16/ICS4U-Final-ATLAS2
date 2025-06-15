"""
ATLAS MoE Training Pipeline

This module integrates the MoE functionality with the existing ATLAS training pipeline,
providing backward compatibility while enabling advanced expert routing.

Author: Steven Chen
Date: 2025-06-14
"""

import os
import warnings
import joblib
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import from original atlas2 module
from .atlas2 import (
    load_ticker_data, create_binary_labels, extract_windows_with_stride,
    transform_3d_to_images, time_series_split, visualize_sample,
    visualize_training_history, plot_binary_metrics, plot_predictions_over_time,
    device
)

# Import MoE components
from .atlas_moe import AtlasMoEModel, AtlasMoEConfig, create_atlas_moe_model
from .data_fingerprint import DataFingerprinter
from .auto_tuning import get_auto_config

# Set matplotlib configuration
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")


class ATLASMoEDataset(torch.utils.data.Dataset):
    """Custom dataset for ATLAS MoE that includes price data for fingerprinting"""
    
    def __init__(self, images, labels, price_windows):
        """
        Args:
            images: Transformed image data [N, C, H, W]
            labels: Binary labels [N]
            price_windows: Raw price windows for fingerprinting [N, seq_len, price_features]
        """
        self.images = images
        self.labels = labels
        self.price_windows = price_windows
        
        assert len(images) == len(labels) == len(price_windows)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.price_windows[idx]


def prepare_moe_dataset(all_windows, all_labels, all_indices, image_data, window_size=50):
    """
    Prepare dataset with price windows for MoE fingerprinting
    
    Args:
        all_windows: Raw price windows [N, window_size, features]
        all_labels: Labels [N]
        all_indices: Window indices [N]
        image_data: Transformed images [N, H, W, C]
        window_size: Size of price windows
        
    Returns:
        Tuple of processed data
    """
    # Convert to OHLCV format for fingerprinting
    # Assuming features are: [Close, Open, High, Low, Volume, ...]
    price_windows = np.zeros((len(all_windows), window_size, 5))  # OHLCV
    
    for i, window in enumerate(all_windows):
        if window.shape[1] >= 5:
            # Reorder to OHLCV: [Open, High, Low, Close, Volume]
            price_windows[i, :, 0] = window[:, 1]  # Open
            price_windows[i, :, 1] = window[:, 2]  # High  
            price_windows[i, :, 2] = window[:, 3]  # Low
            price_windows[i, :, 3] = window[:, 0]  # Close
            price_windows[i, :, 4] = window[:, 4]  # Volume
        else:
            # Fallback: use available features
            available_features = min(window.shape[1], 5)
            price_windows[i, :, :available_features] = window[:, :available_features]
    
    return price_windows


def train_moe_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=30,
    device=device,
    patience=10,
    best_model_save_path="models/atlas_moe_model_best.pth",
    last_model_save_path="models/atlas_moe_model_last.pth",
    load_balance_weight=0.01,
):
    """
    Train ATLAS MoE model with load balancing loss
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], 
              "load_balance_loss": []}
    
    best_val_loss = float("inf")
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_load_balance = 0.0
        
        for images, labels, price_windows in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            images = images.to(device)
            labels = labels.to(device)
            price_windows = price_windows.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, price_windows)
            predictions = outputs['predictions']
            load_balance_loss = outputs['load_balance_loss']
            
            # Combined loss
            main_loss = criterion(predictions, labels.float().view(-1, 1))
            total_loss = main_loss + load_balance_weight * load_balance_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += main_loss.item() * images.size(0)
            train_load_balance += load_balance_loss.item()
            predicted = (predictions > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted.view(-1) == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        avg_load_balance = train_load_balance / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_load_balance = 0.0
        
        with torch.no_grad():
            for images, labels, price_windows in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
            ):
                images = images.to(device)
                labels = labels.to(device)
                price_windows = price_windows.to(device)
                
                outputs = model(images, price_windows)
                predictions = outputs['predictions']
                load_balance_loss = outputs['load_balance_loss']
                
                loss = criterion(predictions, labels.float().view(-1, 1))
                
                val_loss += loss.item() * images.size(0)
                val_load_balance += load_balance_loss.item()
                predicted = (predictions > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted.view(-1) == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        avg_val_load_balance = val_load_balance / len(val_loader)
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["load_balance_loss"].append(avg_load_balance)
        
        # Print progress
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LB Loss: {avg_load_balance:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_info': model.get_model_info(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_model_save_path)
            print(f"Saved best model, validation loss: {val_loss:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Save last model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_info': model.get_model_info(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, last_model_save_path)
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping: {patience} epochs without improvement")
            break
    
    return history


def evaluate_moe_model(model, test_loader, criterion, device=device):
    """Evaluate MoE model and generate detailed metrics"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    all_probs = []
    all_labels = []
    all_routing_info = []
    
    with torch.no_grad():
        for images, labels, price_windows in tqdm(test_loader, desc="Evaluating model"):
            images = images.to(device)
            labels = labels.to(device) 
            price_windows = price_windows.to(device)
            
            outputs = model(images, price_windows)
            predictions = outputs['predictions']
            
            loss = criterion(predictions, labels.float().view(-1, 1))
            
            test_loss += loss.item() * images.size(0)
            predicted = (predictions > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted.view(-1) == labels).sum().item()
            
            # Collect predictions and routing info
            all_probs.extend(predictions.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store routing information
            routing_info = {
                'routing_weights': outputs['routing_weights'].cpu().numpy(),
                'expert_utilization': outputs['expert_utilization'].cpu().numpy(),
                'gating_type': outputs['gating_type']
            }
            all_routing_info.append(routing_info)
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    
    # Analyze expert usage
    print("\nExpert Usage Analysis:")
    if all_routing_info:
        avg_utilization = np.mean([info['expert_utilization'] for info in all_routing_info], axis=0)
        expert_names = ['crypto_hft', 'equity_intraday', 'equity_daily', 'futures_trend', 'low_vol_etf']
        
        for i, (name, util) in enumerate(zip(expert_names, avg_utilization)):
            print(f"  {name}: {util:.3f}")
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Generate confusion matrix and metrics
    predictions = (all_probs > 0.5).astype(int)
    cm = confusion_matrix(all_labels, predictions)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, predictions, target_names=['Down', 'Up']))
    
    # Plot metrics
    plot_binary_metrics(all_labels, all_probs)
    
    return test_loss, test_acc, all_probs, all_labels, all_routing_info


def run_atlas_moe_pipeline(
    ticker_list=["AAPL", "MSFT", "GOOGL"],
    data_dir="data_short",
    window_size=50,
    stride=10,
    image_size=50,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    patience=10,
    validation_size=0.2,
    gap_size=10,
    threshold=0.5,
    enable_auto_tuning=False,
    # MoE specific parameters
    enable_moe=True,
    top_k=2,
    use_learnable_gate=True,
    load_balance_weight=0.01,
):
    """
    Run the complete ATLAS MoE pipeline
    
    Returns:
        tuple: (Trained model, Test data, Training history)
    """
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("=" * 60)
    print("ATLAS MoE Training Pipeline")
    print("=" * 60)
    
    # Auto-tuning integration (if enabled)
    if enable_auto_tuning:
        print("\n🚀 ATLAS Auto-Tuning Enabled")
        try:
            auto_config = get_auto_config(ticker_list, data_dir)
            # Update parameters with auto-tuned values
            window_size = auto_config.get('window_size', window_size)
            stride = auto_config.get('stride', stride)
            threshold = auto_config.get('threshold', threshold)
            batch_size = auto_config.get('batch_size', batch_size)
            learning_rate = auto_config.get('learning_rate', learning_rate)
            epochs = auto_config.get('epochs', epochs)
            patience = auto_config.get('patience', patience)
            print("✅ Auto-tuning completed!")
        except Exception as e:
            print(f"⚠️ Auto-tuning failed: {e}, using manual parameters")
    
    # Data preprocessing (reuse from atlas2.py)
    print(f"\n📊 Processing {len(ticker_list)} stocks...")
    
    all_windows = []
    all_labels = []
    all_indices = []
    
    selected_features = [
        "Close", "Open", "High", "Low", "Volume",
        "MA5", "MA20", "MACD", "RSI", "Upper", "Lower",
        "CRSI", "Kalman_Trend", "FFT_21"
    ]
    
    for ticker in ticker_list:
        print(f"Processing {ticker}...")
        try:
            stock_data = load_ticker_data(ticker, data_dir=data_dir)
            windows, labels, indices = extract_windows_with_stride(
                stock_data, window_size, stride, selected_features
            )
            
            all_windows.extend(windows)
            all_labels.extend(labels)
            all_indices.extend(indices)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)
    
    print(f"Total samples: {len(all_windows)}")
    
    # Label encoding
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(all_labels)
    class_names = label_encoder.classes_
    
    # Transform to images
    print("\n🖼️ Converting to images...")
    transformed_images = transform_3d_to_images(all_windows, image_size=image_size)
    
    # Prepare price windows for MoE
    print("\n🧠 Preparing MoE dataset...")
    price_windows = prepare_moe_dataset(all_windows, all_labels, all_indices, 
                                       transformed_images, window_size)
    
    # Time series split
    print("\n📈 Splitting data...")
    X_train, X_test, y_train, y_test, train_indices, test_indices = time_series_split(
        transformed_images, all_indices, numeric_labels, test_size=0.2, gap_size=gap_size
    )
    
    # Split price windows accordingly
    train_mask = np.isin(all_indices, train_indices)
    test_mask = np.isin(all_indices, test_indices)
    
    price_train = price_windows[train_mask]
    price_test = price_windows[test_mask]
    
    # Further split training into train/val
    train_size = int((1 - validation_size) * len(X_train))
    
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    price_val = price_train[train_size:]
    
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    price_train = price_train[:train_size]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Prepare PyTorch data
    X_train = np.transpose(X_train, (0, 3, 1, 2))  # [B, C, H, W]
    X_val = np.transpose(X_val, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    
    # Create datasets
    train_dataset = ATLASMoEDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
        torch.FloatTensor(price_train)
    )
    
    val_dataset = ATLASMoEDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
        torch.FloatTensor(price_val)
    )
    
    test_dataset = ATLASMoEDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test),
        torch.FloatTensor(price_test)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    print(f"\n🤖 Creating {'MoE' if enable_moe else 'single'} model...")
    
    input_shape = (transformed_images.shape[1], transformed_images.shape[2], 
                  transformed_images.shape[3])
    
    model = create_atlas_moe_model(
        input_shape=input_shape,
        enable_moe=enable_moe,
        top_k=top_k,
        use_learnable_gate=use_learnable_gate,
        dropout_rate=0.5
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Train model
    print(f"\n🚀 Training for {epochs} epochs...")
    history = train_moe_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=epochs, device=device, patience=patience,
        load_balance_weight=load_balance_weight
    )
    
    # Visualize training
    visualize_training_history(history)
    
    # Load best model and evaluate
    print("\n📊 Evaluating best model...")
    checkpoint = torch.load("models/atlas_moe_model_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_probs, test_labels, routing_info = evaluate_moe_model(
        model, test_loader, criterion, device
    )
    
    # Save model info
    model_info = {
        'class_names': class_names,
        'input_shape': input_shape,
        'window_size': window_size,
        'test_accuracy': test_acc,
        'selected_features': selected_features,
        'enable_moe': enable_moe,
        'moe_config': model.get_model_info() if enable_moe else None
    }
    
    joblib.dump(model_info, "models/atlas_moe_model_info.pkl")
    
    print(f"\n✅ Training completed! Test accuracy: {test_acc:.4f}")
    
    return model, (X_test, y_test, test_indices, test_probs, routing_info), history


# Example usage
if __name__ == "__main__":
    # Test MoE pipeline
    model, test_data, history = run_atlas_moe_pipeline(
        ticker_list=["AAPL", "MSFT", "GOOGL", "AMZN"],
        data_dir="data_short",
        epochs=10,  # Short for testing
        enable_moe=True,
        use_learnable_gate=True,
        top_k=2
    )