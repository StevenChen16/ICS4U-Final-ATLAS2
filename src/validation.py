#!/usr/bin/env python3
"""
ATLAS Model Validation System - for Minute-Level Data
Comprehensive validation methods for time series prediction models

This version supports:
- Single file validation for minute-level data
- Automatic ticker detection from filenames
- Flexible time scales (minutes, hours, days)
- Chinese stock market data

Author: based on ATLAS system
Date: June 2025

Usage:
    python -m src.validation.py --data "data_minute/processed_000001_XSHE.csv" --model "models/atlas_model.pth"
    python -m src.validation.py --data "data_minute/" --model "models/atlas_model.pth" --auto-detect
"""

import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import ATLAS modules
try:
    from src.atlas2 import (
        ATLASModel, create_specialized_kernels,
        transform_3d_to_images, load_data_from_csv,
        time_series_split, extract_windows_with_stride,
        create_binary_labels
    )
    from src.data import download_and_prepare_data
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ATLASValidator:
    """validation system supporting minute-level data and flexible configurations"""
    
    def __init__(self, model_path=None, model_info_path=None, data_source=None, time_unit='minutes'):
        """
        Initialize validator
        
        Args:
            model_path (str): Path to trained model
            model_info_path (str): Path to model info file  
            data_source (str): Data directory or single file path
            time_unit (str): Time unit for validation ('minutes', 'hours', 'days')
        """
        self.model_path = model_path
        self.model_info_path = model_info_path
        self.data_source = data_source
        self.time_unit = time_unit
        self.model = None
        self.model_info = None
        self.device = device
        
        # Detect data type and available tickers
        self.data_files, self.ticker_list = self._detect_data_source()
        
        if model_path and model_info_path:
            self._load_model()
    
    def _detect_data_source(self):
        """Detect whether data_source is file or directory and extract tickers"""
        if not self.data_source:
            return {}, []
            
        data_path = Path(self.data_source)
        
        if data_path.is_file() and data_path.suffix == '.csv':
            # Single file mode
            print(f"📁 Single file mode: {data_path.name}")
            ticker = self._extract_ticker_from_filename(data_path.name)
            return {ticker: str(data_path)}, [ticker]
            
        elif data_path.is_dir():
            # Directory mode
            print(f"📁 Directory mode: {data_path}")
            csv_files = list(data_path.glob("*.csv"))
            data_files = {}
            tickers = []
            
            for file_path in csv_files:
                ticker = self._extract_ticker_from_filename(file_path.name)
                data_files[ticker] = str(file_path)
                tickers.append(ticker)
                
            print(f"📊 Found {len(tickers)} data files: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
            return data_files, tickers
        else:
            print(f"❌ Invalid data source: {self.data_source}")
            return {}, []
    
    def _extract_ticker_from_filename(self, filename):
        """Extract ticker symbol from filename"""
        # Remove extension
        base_name = Path(filename).stem
        
        # Common patterns for Chinese stocks
        patterns = [
            r'processed_(\d{6}_\w+)',  # processed_000001_XSHE
            r'(\d{6}\.S[HZ])',         # 000001.SH, 000001.SZ  
            r'(\d{6}_\w+)',            # 000001_XSHE
            r'(\d{6})',                # 000001
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, base_name)
            if match:
                return match.group(1)
        
        # Fallback: use filename as ticker
        return base_name
    
    def _load_model(self):
        """Load trained model and info"""
        print("Loading ATLAS model for validation...")
        
        self.model_info = joblib.load(self.model_info_path)
        
        kernels = create_specialized_kernels()
        self.model = ATLASModel(
            input_shape=self.model_info['input_shape'],
            kernels=kernels,
            dropout_rate=0.0
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Window size: {self.model_info.get('window_size', 'Unknown')}")
        print(f"   Input shape: {self.model_info.get('input_shape', 'Unknown')}")
        
    def _get_time_multiplier(self):
        """Get multiplier for different time units"""
        if self.time_unit == 'minutes':
            return 1
        elif self.time_unit == 'hours':
            return 60
        elif self.time_unit == 'days':
            return 1440  # 24 * 60 minutes
        else:
            return 1
    
    def holdout_validation(self, test_ratio=0.2, gap_periods=100, verbose=True):
        """
        holdout validation with flexible time units
        
        Args:
            test_ratio (float): Proportion of data for testing
            gap_periods (int): Gap between training and test data (in time_unit)
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Validation results
        """
        print(f"\n{'='*60}")
        print(f"ATLAS Holdout Validation")
        print(f"Tickers: {', '.join(self.ticker_list)}")
        print(f"Test ratio: {test_ratio:.1%}, Gap: {gap_periods} {self.time_unit}")
        print(f"{'='*60}")
        
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        ticker_results = {}
        
        for ticker in tqdm(self.ticker_list, desc="Processing tickers"):
            try:
                ticker_result = self._validate_single_ticker(
                    ticker, test_ratio, gap_periods, verbose=False
                )
                
                if ticker_result:
                    ticker_results[ticker] = ticker_result
                    all_predictions.extend(ticker_result['predictions'])
                    all_true_labels.extend(ticker_result['true_labels'])
                    all_probabilities.extend(ticker_result['probabilities'])
                    
            except Exception as e:
                print(f"❌ Error processing {ticker}: {str(e)}")
                continue
        
        # Calculate overall metrics
        if all_predictions:
            overall_metrics = self._calculate_metrics(
                all_true_labels, all_predictions, all_probabilities
            )
            
            results = {
                'method': 'holdout',
                'overall_metrics': overall_metrics,
                'ticker_results': ticker_results,
                'config': {
                    'test_ratio': test_ratio,
                    'gap_periods': gap_periods,
                    'time_unit': self.time_unit,
                    'tickers': self.ticker_list
                }
            }
            
            if verbose:
                self._print_validation_results(results)
            
            return results
        else:
            print("❌ No successful validations")
            return None
    
    def _validate_single_ticker(self, ticker, test_ratio, gap_periods, verbose=False):
        """single ticker validation"""
        try:
            file_path = self.data_files.get(ticker)
            if not file_path or not os.path.exists(file_path):
                if verbose:
                    print(f"⚠️ Data file not found for {ticker}")
                return None
            
            # Load data
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                # Handle different date column names
                date_cols = ['Date', 'Datetime', 'date', 'datetime', 'timestamp', 'time']
                date_col = None
                for col in date_cols:
                    if col in data.columns:
                        date_col = col
                        break
                        
                if date_col:
                    data[date_col] = pd.to_datetime(data[date_col])
                    data.set_index(date_col, inplace=True)
            else:
                data = load_data_from_csv(file_path)
            
            print(f"📊 {ticker}: Loaded {len(data)} records from {data.index[0]} to {data.index[-1]}")
            
            # Prepare features and labels
            selected_features = self.model_info.get('selected_features', [
                'Close', 'Open', 'High', 'Low', 'Volume',
                'MA5', 'MA20', 'MACD', 'RSI', 'Upper', 'Lower',
                'CRSI', 'Kalman_Trend', 'FFT_21'
            ])
            
            # Filter available features
            available_features = [f for f in selected_features if f in data.columns]
            if len(available_features) < len(selected_features):
                missing = set(selected_features) - set(available_features)
                print(f"⚠️ {ticker}: Missing features {missing}, using {len(available_features)} available features")
            
            window_size = self.model_info.get('window_size', 50)
            
            # For minute-level data, use smaller stride
            stride = 5 if self.time_unit == 'minutes' else 10
            
            windows, labels, indices = extract_windows_with_stride(
                data, window_size=window_size, stride=stride, features=available_features
            )
            
            if len(windows) < 50:  # Minimum required samples
                print(f"⚠️ {ticker}: Insufficient windows ({len(windows)}), skipping")
                return None
            
            print(f"📈 {ticker}: Extracted {len(windows)} windows")
            
            # Convert to images
            transformed_images = transform_3d_to_images(windows, image_size=50)
            
            # Encode labels
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            numeric_labels = label_encoder.fit_transform(labels)
            
            # Time series split with gap in data points (not days)
            X_train, X_test, y_train, y_test, train_idx, test_idx = time_series_split(
                transformed_images, indices, numeric_labels, 
                test_size=test_ratio, gap_size=gap_periods
            )
            
            if len(X_test) == 0:
                print(f"⚠️ {ticker}: No test data after split, skipping")
                return None
            
            print(f"🔬 {ticker}: Test set size: {len(X_test)} samples")
            
            # Prepare for model
            X_test = np.transpose(X_test, (0, 3, 1, 2))
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(X_test_tensor)
                probabilities = outputs.cpu().numpy().flatten()
                predictions = (probabilities > 0.5).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, predictions, probabilities)
            
            return {
                'ticker': ticker,
                'predictions': predictions.tolist(),
                'true_labels': y_test.tolist(),
                'probabilities': probabilities.tolist(),
                'metrics': metrics,
                'test_size': len(y_test),
                'test_indices': test_idx.tolist(),
                'data_info': {
                    'total_records': len(data),
                    'total_windows': len(windows),
                    'available_features': len(available_features),
                    'time_range': f"{data.index[0]} to {data.index[-1]}"
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Error validating {ticker}: {str(e)}")
            return None
    
    def walk_forward_validation(self, initial_train_periods=1000, 
                                       test_periods=200, step_periods=100, verbose=True):
        """
        walk-forward validation with flexible time units
        
        Args:
            initial_train_periods (int): Initial training period in time_unit
            test_periods (int): Test period length in time_unit  
            step_periods (int): Step size for moving window in time_unit
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Validation results
        """
        print(f"\n{'='*60}")
        print(f"ATLAS Walk-Forward Validation")
        print(f"Initial train: {initial_train_periods} {self.time_unit}")
        print(f"Test: {test_periods} {self.time_unit}, Step: {step_periods} {self.time_unit}")
        print(f"{'='*60}")
        
        all_fold_results = []
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        for ticker in tqdm(self.ticker_list, desc="Processing tickers"):
            try:
                ticker_folds = self._walk_forward_single_ticker(
                    ticker, initial_train_periods, test_periods, step_periods
                )
                
                if ticker_folds:
                    all_fold_results.extend(ticker_folds)
                    for fold in ticker_folds:
                        all_predictions.extend(fold['predictions'])
                        all_true_labels.extend(fold['true_labels'])
                        all_probabilities.extend(fold['probabilities'])
                        
            except Exception as e:
                print(f"❌ Error in walk-forward for {ticker}: {str(e)}")
                continue
        
        if all_predictions:
            overall_metrics = self._calculate_metrics(
                all_true_labels, all_predictions, all_probabilities
            )
            
            results = {
                'method': 'walk_forward',
                'overall_metrics': overall_metrics,
                'fold_results': all_fold_results,
                'config': {
                    'initial_train_periods': initial_train_periods,
                    'test_periods': test_periods,
                    'step_periods': step_periods,
                    'time_unit': self.time_unit,
                    'tickers': self.ticker_list
                }
            }
            
            if verbose:
                self._print_validation_results(results)
                self._plot_walk_forward_results(all_fold_results)
            
            return results
        else:
            print("❌ No successful walk-forward validations")
            return None
    
    def _walk_forward_single_ticker(self, ticker, initial_train_periods, test_periods, step_periods):
        """walk-forward for single ticker"""
        try:
            file_path = self.data_files.get(ticker)
            if not file_path or not os.path.exists(file_path):
                return None
            
            # Load data
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                date_cols = ['Date', 'Datetime', 'date', 'datetime', 'timestamp', 'time']
                date_col = None
                for col in date_cols:
                    if col in data.columns:
                        date_col = col
                        break
                        
                if date_col:
                    data[date_col] = pd.to_datetime(data[date_col])
                    data.set_index(date_col, inplace=True)
            else:
                data = load_data_from_csv(file_path)
            
            fold_results = []
            
            # Start from initial training period
            start_idx = 0
            while start_idx + initial_train_periods + test_periods < len(data):
                train_end = start_idx + initial_train_periods
                test_start = train_end
                test_end = test_start + test_periods
                
                # Extract training and test periods
                train_data = data.iloc[start_idx:train_end]
                test_data = data.iloc[test_start:test_end]
                
                if len(train_data) < self.model_info.get('window_size', 50) * 2:
                    start_idx += step_periods
                    continue
                
                # Process fold
                fold_result = self._process_walk_forward_fold(
                    ticker, train_data, test_data, len(fold_results)
                )
                
                if fold_result:
                    fold_results.append(fold_result)
                
                start_idx += step_periods
                
                # Limit number of folds to avoid excessive computation
                if len(fold_results) >= 20:
                    break
            
            return fold_results
            
        except Exception as e:
            return None
    
    def _process_walk_forward_fold(self, ticker, train_data, test_data, fold_idx):
        """Process single walk-forward fold"""
        try:
            selected_features = self.model_info.get('selected_features', [
                'Close', 'Open', 'High', 'Low', 'Volume',
                'MA5', 'MA20', 'MACD', 'RSI', 'Upper', 'Lower',
                'CRSI', 'Kalman_Trend', 'FFT_21'
            ])
            
            # Filter available features
            available_features = [f for f in selected_features if f in test_data.columns]
            
            window_size = self.model_info.get('window_size', 50)
            stride = 3 if self.time_unit == 'minutes' else 5
            
            # Extract test windows
            test_windows, test_labels, test_indices = extract_windows_with_stride(
                test_data, window_size=window_size, stride=stride, features=available_features
            )
            
            if len(test_windows) == 0:
                return None
            
            # Convert to images
            test_transformed = transform_3d_to_images(test_windows, image_size=50)
            
            # Encode labels
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.fit(['up', 'down'])  # Ensure consistent encoding
            numeric_labels = label_encoder.transform(test_labels)
            
            # Prepare for model
            X_test = np.transpose(test_transformed, (0, 3, 1, 2))
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(X_test_tensor)
                probabilities = outputs.cpu().numpy().flatten()
                predictions = (probabilities > 0.5).astype(int)
            
            metrics = self._calculate_metrics(numeric_labels, predictions, probabilities)
            
            return {
                'ticker': ticker,
                'fold': fold_idx,
                'predictions': predictions.tolist(),
                'true_labels': numeric_labels.tolist(),
                'probabilities': probabilities.tolist(),
                'metrics': metrics,
                'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
                'test_size': len(numeric_labels)
            }
            
        except Exception as e:
            return None
    
    def _calculate_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive metrics"""
        try:
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
            
            # Only calculate AUC if we have both classes
            try:
                if len(np.unique(true_labels)) > 1:
                    auc_score = roc_auc_score(true_labels, probabilities)
                    avg_precision = average_precision_score(true_labels, probabilities)
                else:
                    auc_score = np.nan
                    avg_precision = np.nan
            except:
                auc_score = np.nan
                avg_precision = np.nan
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc_score,
                'avg_precision': avg_precision,
                'n_samples': len(true_labels),
                'class_distribution': dict(zip(*np.unique(true_labels, return_counts=True)))
            }
        except Exception as e:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc_score': np.nan,
                'avg_precision': np.nan,
                'n_samples': len(true_labels),
                'error': str(e)
            }
    
    def _print_validation_results(self, results):
        """Print validation results"""
        method = results['method']
        metrics = results['overall_metrics']
        config = results['config']
        
        print(f"\n📊 {method.upper().replace('_', ' ')} VALIDATION RESULTS")
        print("="*60)
        print(f"Configuration:")
        print(f"  Time Unit: {config['time_unit']}")
        print(f"  Tickers: {', '.join(config['tickers'])}")
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        if not np.isnan(metrics['auc_score']):
            print(f"  AUC:       {metrics['auc_score']:.3f}")
        print(f"  Samples:   {metrics['n_samples']}")
        
        # Print individual ticker results if available
        if 'ticker_results' in results:
            print(f"\nPer-Ticker Results:")
            for ticker, result in results['ticker_results'].items():
                acc = result['metrics']['accuracy']
                size = result['test_size']
                print(f"  {ticker}: {acc:.3f} accuracy ({size} samples)")
    
    def _plot_walk_forward_results(self, fold_results):
        """Plot walk-forward validation results"""
        if not fold_results:
            return
        
        # Group by ticker
        ticker_folds = {}
        for fold in fold_results:
            ticker = fold['ticker']
            if ticker not in ticker_folds:
                ticker_folds[ticker] = []
            ticker_folds[ticker].append(fold)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Walk-Forward Validation Results ({self.time_unit})', fontsize=16)
        
        # Plot 1: Accuracy over time
        ax1 = axes[0, 0]
        for ticker, folds in ticker_folds.items():
            accuracies = [f['metrics']['accuracy'] for f in folds]
            ax1.plot(range(len(accuracies)), accuracies, marker='o', label=ticker)
        ax1.set_title('Accuracy Over Time')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1-Score distribution
        ax2 = axes[0, 1]
        all_f1_scores = [f['metrics']['f1_score'] for f in fold_results]
        ax2.hist(all_f1_scores, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('F1-Score Distribution')
        ax2.set_xlabel('F1-Score')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sample sizes
        ax3 = axes[1, 0]
        sample_sizes = [f['test_size'] for f in fold_results]
        ax3.boxplot(sample_sizes)
        ax3.set_title('Test Sample Sizes')
        ax3.set_ylabel('Number of Samples')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance by ticker
        ax4 = axes[1, 1]
        ticker_avg_acc = {}
        for ticker, folds in ticker_folds.items():
            avg_acc = np.mean([f['metrics']['accuracy'] for f in folds])
            ticker_avg_acc[ticker] = avg_acc
        
        tickers = list(ticker_avg_acc.keys())
        accuracies = list(ticker_avg_acc.values())
        ax4.bar(range(len(tickers)), accuracies)
        ax4.set_title('Average Accuracy by Ticker')
        ax4.set_xlabel('Ticker')
        ax4.set_ylabel('Average Accuracy')
        ax4.set_xticks(range(len(tickers)))
        ax4.set_xticklabels(tickers, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"results/walk_forward_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 walk-forward results saved: {save_path}")
        
        plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="ATLAS Model Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  - Support for single file or directory input
  - Automatic ticker detection from filenames
  - Flexible time units (minutes, hours, days)
  - Chinese stock market data support

Usage Examples:
  # Single file validation
  python validation.py --data "data_minute/processed_000001_XSHE.csv" --model "models/atlas_model.pth"
  
  # Directory validation with auto-detection
  python validation.py --data "data_minute/" --model "models/atlas_model.pth" --time-unit minutes
  
  # Specific method
  python validation.py --data "data/" --model "models/atlas_model.pth" --method holdout
        """
    )
    
    parser.add_argument("--data", type=str, required=True,
                       help="Data file or directory containing CSV files")
    parser.add_argument("--model", type=str, default="models/atlas_binary_model_best.pth",
                       help="Trained model path")
    parser.add_argument("--model_info", type=str, default="models/atlas_binary_model_info.pkl",
                       help="Model info file path")
    parser.add_argument("--method", type=str, default="holdout",
                       choices=['holdout', 'walk_forward', 'both'],
                       help="Validation method")
    parser.add_argument("--time-unit", type=str, default="minutes",
                       choices=['minutes', 'hours', 'days'],
                       help="Time unit for validation periods")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="Test set ratio for holdout validation")
    parser.add_argument("--gap", type=int, default=100,
                       help="Gap between train/test in time units")
    parser.add_argument("--save", action="store_true", 
                       help="Save validation results")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔬 ATLAS Model Validation System")
    print("="*70)
    print(f"Data source: {args.data}")
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Time unit: {args.time_unit}")
    
    try:
        # Initialize validator
        validator = ATLASValidator(
            model_path=args.model,
            model_info_path=args.model_info,
            data_source=args.data,
            time_unit=args.time_unit
        )
        
        if not validator.ticker_list:
            print("❌ No valid data files found!")
            return 1
        
        print(f"📊 Processing {len(validator.ticker_list)} ticker(s): {', '.join(validator.ticker_list)}")
        
        # Run validation
        results = {}
        
        if args.method in ['holdout', 'both']:
            print(f"\n🔬 Running Holdout Validation...")
            holdout_results = validator.holdout_validation(
                test_ratio=args.test_ratio, 
                gap_periods=args.gap,
                verbose=True
            )
            if holdout_results:
                results['holdout'] = holdout_results
        
        if args.method in ['walk_forward', 'both']:
            print(f"\n🔬 Running Walk-Forward Validation...")
            # Adjust periods based on time unit
            if args.time_unit == 'minutes':
                initial_train = 2000  # ~33 hours
                test_periods = 400    # ~6.7 hours  
                step_periods = 200    # ~3.3 hours
            elif args.time_unit == 'hours':
                initial_train = 100   # ~4 days
                test_periods = 20     # ~20 hours
                step_periods = 10     # ~10 hours
            else:  # days
                initial_train = 30    # 30 days
                test_periods = 7      # 7 days
                step_periods = 3      # 3 days
                
            wf_results = validator.walk_forward_validation(
                initial_train_periods=initial_train,
                test_periods=test_periods,
                step_periods=step_periods,
                verbose=True
            )
            if wf_results:
                results['walk_forward'] = wf_results
        
        # Summary
        if results:
            print(f"\n🎯 VALIDATION SUMMARY")
            print("="*50)
            for method, result in results.items():
                metrics = result['overall_metrics']
                print(f"{method.title()}:")
                print(f"  Accuracy: {metrics['accuracy']:.3f}")
                print(f"  F1-Score: {metrics['f1_score']:.3f}")
                print(f"  Samples:  {metrics['n_samples']}")
                
            # Save results if requested
            if args.save:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_path = f"results/validation_{timestamp}.pkl"
                os.makedirs("results", exist_ok=True)
                joblib.dump(results, results_path)
                print(f"\n📁 Results saved: {results_path}")
        else:
            print("❌ No successful validation results")
            return 1
        
        print(f"\n✅ validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)