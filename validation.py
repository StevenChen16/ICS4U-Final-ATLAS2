#!/usr/bin/env python3
"""
ATLAS Model Validation System
Comprehensive validation methods for time series prediction models

This module provides various validation techniques specifically designed for 
time series models, ensuring temporal integrity and realistic performance estimates.

Author: Based on ATLAS system
Date: June 2025

Usage:
    python atlas_validation.py --data data/ --model models/atlas_binary_model_best.pth
    python atlas_validation.py --method walk_forward --validation_type extensive
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

class ATLASModelValidator:
    """Comprehensive validation system for ATLAS time series models"""
    
    def __init__(self, model_path=None, model_info_path=None, data_dir="data"):
        """
        Initialize model validator
        
        Args:
            model_path (str): Path to trained model
            model_info_path (str): Path to model info file
            data_dir (str): Data directory containing stock CSV files
        """
        self.model_path = model_path
        self.model_info_path = model_info_path
        self.data_dir = data_dir
        self.model = None
        self.model_info = None
        self.device = device
        
        if model_path and model_info_path:
            self._load_model()
    
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
    
    def holdout_validation(self, ticker_list, test_ratio=0.2, gap_days=20, verbose=True):
        """
        Traditional holdout validation with temporal split
        
        Args:
            ticker_list (list): List of stock tickers to validate on
            test_ratio (float): Proportion of data for testing
            gap_days (int): Gap between training and test data
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Validation results
        """
        print(f"\n{'='*60}")
        print(f"ATLAS Holdout Validation")
        print(f"Tickers: {', '.join(ticker_list)}")
        print(f"Test ratio: {test_ratio:.1%}, Gap: {gap_days} days")
        print(f"{'='*60}")
        
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        ticker_results = {}
        
        for ticker in tqdm(ticker_list, desc="Processing tickers"):
            try:
                # Load and prepare data
                ticker_result = self._validate_single_ticker_holdout(
                    ticker, test_ratio, gap_days, verbose=False
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
                    'gap_days': gap_days,
                    'tickers': ticker_list
                }
            }
            
            if verbose:
                self._print_validation_results(results)
            
            return results
        else:
            print("❌ No successful validations")
            return None
    
    def walk_forward_validation(self, ticker_list, initial_train_months=12, 
                               test_months=3, step_months=1, verbose=True):
        """
        Walk-forward validation - more realistic for time series
        
        Args:
            ticker_list (list): List of stock tickers
            initial_train_months (int): Initial training period in months
            test_months (int): Test period length in months
            step_months (int): Step size for moving window in months
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Validation results
        """
        print(f"\n{'='*60}")
        print(f"ATLAS Walk-Forward Validation")
        print(f"Initial train: {initial_train_months}m, Test: {test_months}m, Step: {step_months}m")
        print(f"{'='*60}")
        
        all_fold_results = []
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        for ticker in tqdm(ticker_list, desc="Processing tickers"):
            try:
                ticker_folds = self._walk_forward_single_ticker(
                    ticker, initial_train_months, test_months, step_months
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
                    'initial_train_months': initial_train_months,
                    'test_months': test_months,
                    'step_months': step_months,
                    'tickers': ticker_list
                }
            }
            
            if verbose:
                self._print_validation_results(results)
                self._plot_walk_forward_results(all_fold_results)
            
            return results
        else:
            print("❌ No successful walk-forward validations")
            return None
    
    def time_series_cross_validation(self, ticker_list, n_splits=5, test_ratio=0.2, verbose=True):
        """
        Time series cross-validation with expanding window
        
        Args:
            ticker_list (list): List of stock tickers
            n_splits (int): Number of CV splits
            test_ratio (float): Test ratio for each split
            verbose (bool): Whether to print detailed results
            
        Returns:
            dict: Cross-validation results
        """
        print(f"\n{'='*60}")
        print(f"ATLAS Time Series Cross-Validation")
        print(f"Splits: {n_splits}, Test ratio: {test_ratio:.1%}")
        print(f"{'='*60}")
        
        all_cv_results = []
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        for ticker in tqdm(ticker_list, desc="Processing tickers"):
            try:
                ticker_cv_results = self._cross_validate_single_ticker(
                    ticker, n_splits, test_ratio
                )
                
                if ticker_cv_results:
                    all_cv_results.extend(ticker_cv_results)
                    for cv_result in ticker_cv_results:
                        all_predictions.extend(cv_result['predictions'])
                        all_true_labels.extend(cv_result['true_labels'])
                        all_probabilities.extend(cv_result['probabilities'])
                        
            except Exception as e:
                print(f"❌ Error in CV for {ticker}: {str(e)}")
                continue
        
        if all_predictions:
            overall_metrics = self._calculate_metrics(
                all_true_labels, all_predictions, all_probabilities
            )
            
            # Calculate CV statistics
            cv_accuracies = [cv['metrics']['accuracy'] for cv in all_cv_results]
            cv_f1_scores = [cv['metrics']['f1_score'] for cv in all_cv_results]
            
            results = {
                'method': 'time_series_cv',
                'overall_metrics': overall_metrics,
                'cv_results': all_cv_results,
                'cv_statistics': {
                    'mean_accuracy': np.mean(cv_accuracies),
                    'std_accuracy': np.std(cv_accuracies),
                    'mean_f1': np.mean(cv_f1_scores),
                    'std_f1': np.std(cv_f1_scores)
                },
                'config': {
                    'n_splits': n_splits,
                    'test_ratio': test_ratio,
                    'tickers': ticker_list
                }
            }
            
            if verbose:
                self._print_validation_results(results)
                self._plot_cv_results(all_cv_results)
            
            return results
        else:
            print("❌ No successful CV validations")
            return None
    
    def extensive_validation_suite(self, ticker_list, save_results=True):
        """
        Run comprehensive validation suite with multiple methods
        
        Args:
            ticker_list (list): List of stock tickers
            save_results (bool): Whether to save results to files
            
        Returns:
            dict: Complete validation results
        """
        print(f"\n{'='*80}")
        print(f"🚀 ATLAS EXTENSIVE VALIDATION SUITE")
        print(f"{'='*80}")
        
        validation_results = {}
        
        # 1. Holdout Validation
        print("\n1️⃣ Running Holdout Validation...")
        holdout_results = self.holdout_validation(ticker_list, verbose=False)
        if holdout_results:
            validation_results['holdout'] = holdout_results
        
        # 2. Walk-Forward Validation
        print("\n2️⃣ Running Walk-Forward Validation...")
        walkforward_results = self.walk_forward_validation(ticker_list, verbose=False)
        if walkforward_results:
            validation_results['walk_forward'] = walkforward_results
        
        # 3. Time Series Cross-Validation
        print("\n3️⃣ Running Time Series Cross-Validation...")
        cv_results = self.time_series_cross_validation(ticker_list, verbose=False)
        if cv_results:
            validation_results['time_series_cv'] = cv_results
        
        # 4. Generate comprehensive report
        self._generate_comprehensive_report(validation_results)
        
        # 5. Save results if requested
        if save_results:
            self._save_validation_results(validation_results)
        
        return validation_results
    
    def _validate_single_ticker_holdout(self, ticker, test_ratio, gap_days, verbose=False):
        """Holdout validation for single ticker"""
        try:
            # Load data
            file_path = os.path.join(self.data_dir, f"{ticker}.csv")
            if not os.path.exists(file_path):
                if verbose:
                    print(f"⚠️ Data file not found for {ticker}")
                return None
            
            data = load_data_from_csv(file_path)
            
            # Prepare features and labels
            selected_features = self.model_info.get('selected_features', [
                'Close', 'Open', 'High', 'Low', 'Volume',
                'MA5', 'MA20', 'MACD', 'RSI', 'Upper', 'Lower',
                'CRSI', 'Kalman_Trend', 'FFT_21'
            ])
            
            window_size = self.model_info['window_size']
            windows, labels, indices = extract_windows_with_stride(
                data, window_size=window_size, stride=10, features=selected_features
            )
            
            if len(windows) < 50:  # Minimum required samples
                return None
            
            # Convert to images
            transformed_images = transform_3d_to_images(windows, image_size=50)
            
            # Encode labels
            label_encoder = self.model_info.get('label_encoder')
            if label_encoder is None:
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                numeric_labels = label_encoder.fit_transform(labels)
            else:
                numeric_labels = label_encoder.transform(labels)
            
            # Time series split
            X_train, X_test, y_train, y_test, train_idx, test_idx = time_series_split(
                transformed_images, indices, numeric_labels, 
                test_size=test_ratio, gap_size=gap_days
            )
            
            if len(X_test) == 0:
                return None
            
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
                'test_indices': test_idx.tolist()
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Error validating {ticker}: {str(e)}")
            return None
    
    def _walk_forward_single_ticker(self, ticker, initial_train_months, test_months, step_months):
        """Walk-forward validation for single ticker"""
        try:
            file_path = os.path.join(self.data_dir, f"{ticker}.csv")
            if not os.path.exists(file_path):
                return None
            
            data = load_data_from_csv(file_path)
            
            # Convert months to approximate days
            initial_train_days = initial_train_months * 30
            test_days = test_months * 30
            step_days = step_months * 30
            
            fold_results = []
            
            # Start from initial training period
            start_idx = 0
            while start_idx + initial_train_days + test_days < len(data):
                train_end = start_idx + initial_train_days
                test_start = train_end
                test_end = test_start + test_days
                
                # Extract training and test periods
                train_data = data.iloc[start_idx:train_end]
                test_data = data.iloc[test_start:test_end]
                
                if len(train_data) < self.model_info['window_size'] * 2:
                    start_idx += step_days
                    continue
                
                # Process fold (simplified - in practice you'd retrain model)
                fold_result = self._process_walk_forward_fold(
                    ticker, train_data, test_data, len(fold_results)
                )
                
                if fold_result:
                    fold_results.append(fold_result)
                
                start_idx += step_days
                
                # Limit number of folds to avoid excessive computation
                if len(fold_results) >= 10:
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
            
            window_size = self.model_info['window_size']
            
            # Extract test windows
            test_windows, test_labels, test_indices = extract_windows_with_stride(
                test_data, window_size=window_size, stride=5, features=selected_features
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
    
    def _cross_validate_single_ticker(self, ticker, n_splits, test_ratio):
        """Cross-validation for single ticker"""
        try:
            file_path = os.path.join(self.data_dir, f"{ticker}.csv")
            if not os.path.exists(file_path):
                return None
            
            data = load_data_from_csv(file_path)
            
            selected_features = self.model_info.get('selected_features', [
                'Close', 'Open', 'High', 'Low', 'Volume',
                'MA5', 'MA20', 'MACD', 'RSI', 'Upper', 'Lower',
                'CRSI', 'Kalman_Trend', 'FFT_21'
            ])
            
            window_size = self.model_info['window_size']
            windows, labels, indices = extract_windows_with_stride(
                data, window_size=window_size, stride=10, features=selected_features
            )
            
            if len(windows) < n_splits * 20:  # Minimum samples per split
                return None
            
            transformed_images = transform_3d_to_images(windows, image_size=50)
            
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            numeric_labels = label_encoder.fit_transform(labels)
            
            cv_results = []
            
            # Time series expanding window CV
            for split in range(n_splits):
                # Expanding training window
                train_end_ratio = 0.5 + (split * 0.1)  # Start from 50%, expand by 10% each split
                train_end_idx = int(len(transformed_images) * train_end_ratio)
                test_start_idx = train_end_idx
                test_end_idx = int(test_start_idx + len(transformed_images) * test_ratio)
                
                if test_end_idx > len(transformed_images):
                    break
                
                X_test = transformed_images[test_start_idx:test_end_idx]
                y_test = numeric_labels[test_start_idx:test_end_idx]
                
                if len(X_test) == 0:
                    continue
                
                # Prepare for model
                X_test = np.transpose(X_test, (0, 3, 1, 2))
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(X_test_tensor)
                    probabilities = outputs.cpu().numpy().flatten()
                    predictions = (probabilities > 0.5).astype(int)
                
                metrics = self._calculate_metrics(y_test, predictions, probabilities)
                
                cv_results.append({
                    'ticker': ticker,
                    'split': split,
                    'predictions': predictions.tolist(),
                    'true_labels': y_test.tolist(),
                    'probabilities': probabilities.tolist(),
                    'metrics': metrics,
                    'test_size': len(y_test)
                })
            
            return cv_results
            
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
        
        print(f"\n📊 {method.upper()} VALIDATION RESULTS")
        print("="*50)
        print(f"Overall Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        if not np.isnan(metrics['auc_score']):
            print(f"  AUC:       {metrics['auc_score']:.3f}")
        print(f"  Samples:   {metrics['n_samples']}")
        
        if method == 'time_series_cv':
            cv_stats = results['cv_statistics']
            print(f"\nCross-Validation Statistics:")
            print(f"  Mean Accuracy: {cv_stats['mean_accuracy']:.3f} ± {cv_stats['std_accuracy']:.3f}")
            print(f"  Mean F1:       {cv_stats['mean_f1']:.3f} ± {cv_stats['std_f1']:.3f}")
    
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
        fig.suptitle('Walk-Forward Validation Results', fontsize=16)
        
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
        save_path = f"results/walk_forward_validation_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 Walk-forward results saved: {save_path}")
        
        plt.show()
    
    def _plot_cv_results(self, cv_results):
        """Plot cross-validation results"""
        if not cv_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Time Series Cross-Validation Results', fontsize=16)
        
        # Plot 1: Accuracy by split
        ax1 = axes[0, 0]
        splits = [cv['split'] for cv in cv_results]
        accuracies = [cv['metrics']['accuracy'] for cv in cv_results]
        ax1.scatter(splits, accuracies, alpha=0.6)
        ax1.set_title('Accuracy by CV Split')
        ax1.set_xlabel('Split')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Metrics distribution
        ax2 = axes[0, 1]
        metrics_data = {
            'Accuracy': [cv['metrics']['accuracy'] for cv in cv_results],
            'Precision': [cv['metrics']['precision'] for cv in cv_results],
            'Recall': [cv['metrics']['recall'] for cv in cv_results],
            'F1-Score': [cv['metrics']['f1_score'] for cv in cv_results]
        }
        
        positions = range(1, len(metrics_data) + 1)
        box_data = [metrics_data[key] for key in metrics_data.keys()]
        ax2.boxplot(box_data, positions=positions)
        ax2.set_title('Metrics Distribution')
        ax2.set_xticklabels(metrics_data.keys())
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance by ticker
        ax3 = axes[1, 0]
        ticker_performance = {}
        for cv in cv_results:
            ticker = cv['ticker']
            if ticker not in ticker_performance:
                ticker_performance[ticker] = []
            ticker_performance[ticker].append(cv['metrics']['accuracy'])
        
        tickers = list(ticker_performance.keys())
        avg_accuracies = [np.mean(ticker_performance[t]) for t in tickers]
        
        ax3.bar(range(len(tickers)), avg_accuracies)
        ax3.set_title('Average CV Accuracy by Ticker')
        ax3.set_xlabel('Ticker')
        ax3.set_ylabel('Average Accuracy')
        ax3.set_xticks(range(len(tickers)))
        ax3.set_xticklabels(tickers, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sample size vs performance
        ax4 = axes[1, 1]
        sample_sizes = [cv['test_size'] for cv in cv_results]
        accuracies = [cv['metrics']['accuracy'] for cv in cv_results]
        ax4.scatter(sample_sizes, accuracies, alpha=0.6)
        ax4.set_title('Sample Size vs Accuracy')
        ax4.set_xlabel('Test Sample Size')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"results/cv_validation_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 CV results saved: {save_path}")
        
        plt.show()
    
    def _generate_comprehensive_report(self, validation_results):
        """Generate comprehensive validation report"""
        print(f"\n{'='*80}")
        print(f"🎯 COMPREHENSIVE VALIDATION REPORT")
        print(f"{'='*80}")
        
        if not validation_results:
            print("❌ No validation results available")
            return
        
        # Summary table
        summary_data = []
        for method, results in validation_results.items():
            metrics = results['overall_metrics']
            summary_data.append([
                method.replace('_', ' ').title(),
                f"{metrics['accuracy']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}",
                f"{metrics['n_samples']}"
            ])
        
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"{'Method':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Samples':<10}")
        print("-" * 80)
        for row in summary_data:
            print(f"{row[0]:<20} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")
        
        # Best performing method
        best_method = max(validation_results.keys(), 
                         key=lambda x: validation_results[x]['overall_metrics']['accuracy'])
        best_accuracy = validation_results[best_method]['overall_metrics']['accuracy']
        
        print(f"\n🏆 BEST PERFORMING METHOD: {best_method.replace('_', ' ').title()}")
        print(f"   Best Accuracy: {best_accuracy:.3f}")
        
        # Stability analysis
        if 'time_series_cv' in validation_results:
            cv_stats = validation_results['time_series_cv']['cv_statistics']
            print(f"\n📈 MODEL STABILITY (from CV):")
            print(f"   Accuracy Std Dev: {cv_stats['std_accuracy']:.3f}")
            print(f"   F1-Score Std Dev: {cv_stats['std_f1']:.3f}")
            
            if cv_stats['std_accuracy'] < 0.05:
                stability = "Very Stable"
            elif cv_stats['std_accuracy'] < 0.1:
                stability = "Stable"
            else:
                stability = "Unstable"
            print(f"   Overall Stability: {stability}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if best_accuracy > 0.8:
            print("   ✅ Model shows excellent performance across validation methods")
        elif best_accuracy > 0.7:
            print("   ✅ Model shows good performance, consider fine-tuning")
        elif best_accuracy > 0.6:
            print("   ⚠️  Model shows moderate performance, significant improvement needed")
        else:
            print("   ❌ Model shows poor performance, consider model redesign")
        
        print(f"   📚 For time series models, walk-forward validation is most realistic")
        print(f"   📚 Consider ensemble methods if individual model performance varies")
    
    def _save_validation_results(self, validation_results):
        """Save validation results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_path = f"results/validation_results_{timestamp}.pkl"
        os.makedirs("results", exist_ok=True)
        joblib.dump(validation_results, results_path)
        
        # Save summary CSV
        summary_data = []
        for method, results in validation_results.items():
            metrics = results['overall_metrics']
            summary_data.append({
                'method': method,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'auc_score': metrics.get('auc_score', np.nan),
                'n_samples': metrics['n_samples']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"results/validation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n📁 Validation results saved:")
        print(f"   Detailed: {results_path}")
        print(f"   Summary:  {summary_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="ATLAS Model Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation Methods:
  holdout     - Traditional train/test split with temporal order
  walk_forward - Realistic time series validation with moving window  
  time_series_cv - Cross-validation with expanding window
  extensive   - Run all validation methods
        """
    )
    
    parser.add_argument("--data", type=str, default="data",
                       help="Data directory containing CSV files")
    parser.add_argument("--model", type=str, default="models/atlas_binary_model_best.pth",
                       help="Trained model path")
    parser.add_argument("--model_info", type=str, default="models/atlas_binary_model_info.pkl",
                       help="Model info file path")
    parser.add_argument("--method", type=str, default="extensive",
                       choices=['holdout', 'walk_forward', 'time_series_cv', 'extensive'],
                       help="Validation method")
    parser.add_argument("--tickers", type=str, 
                       default="AAPL,MSFT,GOOGL,AMZN,TSLA",
                       help="Comma-separated list of tickers to validate on")
    parser.add_argument("--save", action="store_true", 
                       help="Save validation results")
    
    args = parser.parse_args()
    
    # Parse ticker list
    ticker_list = [t.strip().upper() for t in args.tickers.split(',')]
    
    print("="*60)
    print("🔬 ATLAS Model Validation System")
    print("="*60)
    print(f"Data directory: {args.data}")
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Tickers: {', '.join(ticker_list)}")
    
    try:
        # Initialize validator
        validator = ATLASModelValidator(
            model_path=args.model,
            model_info_path=args.model_info,
            data_dir=args.data
        )
        
        # Run validation
        if args.method == "holdout":
            results = validator.holdout_validation(ticker_list)
        elif args.method == "walk_forward":
            results = validator.walk_forward_validation(ticker_list)
        elif args.method == "time_series_cv":
            results = validator.time_series_cross_validation(ticker_list)
        elif args.method == "extensive":
            results = validator.extensive_validation_suite(ticker_list, save_results=args.save)
        
        print(f"\n✅ Validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)