#!/usr/bin/env python3
"""
ATLAS: Advanced Technical Learning Analysis System
Inference Module - Stock Market Prediction

This module loads trained ATLAS model and performs predictions on new stock data.
Supports real-time data fetching, historical data analysis and batch predictions.

Author: Based on Steven Chen's ATLAS system
Date: June 2025

Usage:
    python atlas_inference.py --ticker AAPL --days 50
    python atlas_inference.py --file path/to/data.csv
    python atlas_inference.py --batch ticker1,ticker2,ticker3
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

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import necessary modules
try:
    from src.atlas2 import (
        ATLASModel, create_specialized_kernels, 
        transform_3d_to_images, load_data_from_csv,
        visualize_sample
    )
    from src.data import download_and_prepare_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run this script from the ATLAS project root directory")
    sys.exit(1)

# Setup
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ATLASInferenceEngine:
    """ATLAS Inference Engine for Stock Prediction"""
    
    def __init__(self, model_path="models/atlas_binary_model_best.pth", 
                 model_info_path="models/atlas_binary_model_info.pkl"):
        """
        Initialize inference engine
        
        Args:
            model_path (str): Path to trained model
            model_info_path (str): Path to model info file
        """
        self.model_path = model_path
        self.model_info_path = model_info_path
        self.model = None
        self.model_info = None
        self.device = device
        
        # Check if model files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(model_info_path):
            # raise FileNotFoundError(f"Model info file not found: {model_info_path}")
            pass
            
        self._load_model()
        
    def _load_model(self):
        """Load trained model"""
        print("Loading ATLAS model...")
        
        # Load model info
        self.model_info = joblib.load(self.model_info_path)
        
        # Create model architecture
        kernels = create_specialized_kernels()
        self.model = ATLASModel(
            input_shape=self.model_info['input_shape'],
            kernels=kernels,
            dropout_rate=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=False))
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Validation accuracy: {self.model_info['test_accuracy']:.1%}")
        print(f"   Window size: {self.model_info['window_size']}")
        print(f"   Input shape: {self.model_info['input_shape']}")
        print(f"   Supported classes: {', '.join(self.model_info['class_names'])}")
        
    def fetch_realtime_data(self, ticker, start_date="2023-01-01", end_date=None):
        """
        Fetch real-time stock data
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data fetching
            end_date (str): End date for data fetching
            
        Returns:
            pd.DataFrame: Stock data with technical indicators
        """
        print(f"Fetching real-time data for {ticker}...")
        
        try:
            # Set end date to today if not specified
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            # Use the actual data preparation function from the project
            data = download_and_prepare_data(ticker, start_date, end_date)
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
                
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            print(f"✅ Successfully fetched {ticker} data, {len(data)} records")
            return data
            
        except Exception as e:
            print(f"❌ Failed to fetch data: {str(e)}")
            return None
    
    def prepare_data_for_prediction(self, data, window_size=None):
        """
        Prepare data for prediction
        
        Args:
            data (pd.DataFrame): Stock data
            window_size (int): Window size, uses model default if None
            
        Returns:
            tuple: (processed image data, last window raw data, corresponding dates)
        """
        if window_size is None:
            window_size = self.model_info['window_size']
            
        # Select features - use the actual features from model training
        selected_features = self.model_info.get('selected_features', [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'MA5', 'MA20', 'MACD', 'RSI', 'Upper', 'Lower',
            'CRSI', 'Kalman_Trend', 'FFT_21'
        ])
        
        # Ensure all required features exist
        available_features = [f for f in selected_features if f in data.columns]
        if len(available_features) < len(selected_features):
            missing = set(selected_features) - set(available_features)
            print(f"⚠️ Missing features: {missing}")
            print(f"   Using available features: {available_features}")
        
        # Prepare last window data
        if len(data) < window_size:
            raise ValueError(f"Data length ({len(data)}) is less than window size ({window_size})")
            
        # Get the last complete window
        window_data = data.iloc[-window_size:][available_features].values
        window_data = window_data.reshape(1, window_size, len(available_features))
        
        # Convert to image representations
        image_size = self.model_info['input_shape'][0]  # Usually 50
        transformed_images = transform_3d_to_images(window_data, image_size=image_size)
        
        # Adjust channel order (B, H, W, C) -> (B, C, H, W)
        transformed_images = np.transpose(transformed_images, (0, 3, 1, 2))
        
        # Convert to PyTorch tensor
        tensor_data = torch.FloatTensor(transformed_images).to(self.device)
        
        # Get last window raw data and dates
        last_window_raw = data.iloc[-window_size:][available_features].values
        last_dates = data.iloc[-window_size:]['Date'].values if 'Date' in data.columns else None
        
        return tensor_data, last_window_raw, last_dates
    
    def predict(self, data, return_probability=True):
        """
        Make predictions on data
        
        Args:
            data (torch.Tensor): Preprocessed image data
            return_probability (bool): Whether to return probabilities
            
        Returns:
            dict: Dictionary containing prediction results
        """
        with torch.no_grad():
            # Forward pass
            outputs = self.model(data)
            probabilities = outputs.cpu().numpy().flatten()
            
            # Convert to predicted classes
            predictions = (probabilities > 0.5).astype(int)
            
            # Map to class names
            class_names = self.model_info['class_names']
            predicted_classes = [class_names[pred] for pred in predictions]
            
            results = {
                'predictions': predicted_classes,
                'probabilities': probabilities,
                'raw_predictions': predictions,
                'confidence': np.abs(probabilities - 0.5) * 2  # Confidence (0-1)
            }
            
            return results
    
    def predict_single_ticker(self, ticker, start_date="2023-01-01", visualize=True, save_path=None):
        """
        Predict single stock
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date for data
            visualize (bool): Whether to generate visualization
            save_path (str): Save path for visualization
            
        Returns:
            dict: Prediction results
        """
        print(f"\n{'='*50}")
        print(f"ATLAS Stock Prediction: {ticker}")
        print(f"{'='*50}")
        
        # Fetch data
        data = self.fetch_realtime_data(ticker, start_date=start_date)
        if data is None:
            return None
            
        try:
            # Prepare data
            tensor_data, raw_window, dates = self.prepare_data_for_prediction(data)
            
            # Make prediction
            results = self.predict(tensor_data)
            
            # Prepare return results
            prediction_result = {
                'ticker': ticker,
                'prediction': results['predictions'][0],
                'probability': results['probabilities'][0],
                'confidence': results['confidence'][0],
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_end_date': data['Date'].iloc[-1] if 'Date' in data.columns else 'Unknown',
                'window_size': self.model_info['window_size']
            }
            
            # Print results
            self._print_prediction_result(prediction_result)
            
            # Visualization
            if visualize:
                self._visualize_prediction(
                    ticker, data, raw_window, dates, 
                    prediction_result, save_path
                )
                
            return prediction_result
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return None
    
    def predict_batch(self, ticker_list, start_date="2023-01-01", save_results=True):
        """
        Batch prediction for multiple stocks
        
        Args:
            ticker_list (list): List of stock ticker symbols
            start_date (str): Start date for data
            save_results (bool): Whether to save results
            
        Returns:
            pd.DataFrame: Summary of prediction results
        """
        print(f"\n{'='*60}")
        print(f"ATLAS Batch Stock Prediction")
        print(f"Target stocks: {', '.join(ticker_list)}")
        print(f"{'='*60}")
        
        results = []
        
        for ticker in tqdm(ticker_list, desc="Prediction progress"):
            result = self.predict_single_ticker(
                ticker, start_date=start_date, visualize=False
            )
            if result:
                results.append(result)
                
        # Convert to DataFrame
        if results:
            df_results = pd.DataFrame(results)
            
            # Sort by probability
            df_results = df_results.sort_values('probability', ascending=False)
            
            # Save results
            if save_results:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_path = f"results/batch_predictions_{timestamp}.csv"
                os.makedirs("results", exist_ok=True)
                df_results.to_csv(results_path, index=False)
                print(f"\n📁 Batch prediction results saved: {results_path}")
                
            # Print summary
            self._print_batch_summary(df_results)
            
            return df_results
        else:
            print("❌ No successful prediction results")
            return pd.DataFrame()
    
    def predict_from_file(self, file_path, visualize=True):
        """
        Predict from CSV file
        
        Args:
            file_path (str): CSV file path
            visualize (bool): Whether to visualize
            
        Returns:
            dict: Prediction results
        """
        print(f"\nPredicting from file: {file_path}")
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                data = load_data_from_csv(file_path)
                # Reset index to make Date a column if it was the index
                if 'Date' not in data.columns and data.index.name == 'Date':
                    data.reset_index(inplace=True)
            else:
                data = pd.read_csv(file_path)
                
            # Prepare data
            tensor_data, raw_window, dates = self.prepare_data_for_prediction(data)
            
            # Make prediction
            results = self.predict(tensor_data)
            
            # Prepare results
            prediction_result = {
                'file': file_path,
                'prediction': results['predictions'][0],
                'probability': results['probabilities'][0],
                'confidence': results['confidence'][0],
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'window_size': self.model_info['window_size']
            }
            
            # Print results
            self._print_prediction_result(prediction_result)
            
            # Visualization
            if visualize:
                self._visualize_prediction(
                    os.path.basename(file_path), data, raw_window, dates, 
                    prediction_result
                )
                
            return prediction_result
            
        except Exception as e:
            print(f"❌ File prediction failed: {str(e)}")
            return None
    
    def _print_prediction_result(self, result):
        """Print prediction results"""
        prob = result['probability']
        confidence = result['confidence']
        prediction = result['prediction']
        
        print(f"\n📊 Prediction Results:")
        print(f"   🎯 Predicted Direction: {prediction.upper()}")
        print(f"   📈 Up Probability: {prob:.1%}")
        print(f"   🎪 Confidence: {confidence:.1%}")
        
        # Give advice based on confidence
        if confidence > 0.7:
            confidence_level = "High"
            advice = "Worth attention"
        elif confidence > 0.4:
            confidence_level = "Medium"
            advice = "Consider carefully"
        else:
            confidence_level = "Low"
            advice = "Suggest wait and see"
            
        print(f"   💡 Confidence Level: {confidence_level}")
        print(f"   💰 Investment Advice: {advice}")
        
        # Risk warnings
        if prediction == "up" and prob > 0.8:
            print(f"   ⚠️  Note: High probability up prediction, but please consider other factors")
        elif prediction == "down" and prob < 0.2:
            print(f"   ⚠️  Note: High probability down prediction, suggest risk control")
    
    def _print_batch_summary(self, df_results):
        """Print batch prediction summary"""
        total = len(df_results)
        up_count = sum(df_results['prediction'] == 'up')
        down_count = total - up_count
        
        high_confidence = sum(df_results['confidence'] > 0.7)
        avg_confidence = df_results['confidence'].mean()
        
        print(f"\n📊 Batch Prediction Summary:")
        print(f"   Total: {total} stocks")
        print(f"   📈 Predicted Up: {up_count} stocks ({up_count/total:.1%})")
        print(f"   📉 Predicted Down: {down_count} stocks ({down_count/total:.1%})")
        print(f"   🎪 High Confidence (>70%): {high_confidence} stocks")
        print(f"   💯 Average Confidence: {avg_confidence:.1%}")
        
        print(f"\n🔥 High Confidence Up Stocks:")
        high_up = df_results[
            (df_results['prediction'] == 'up') & 
            (df_results['confidence'] > 0.7)
        ].head(5)
        
        if not high_up.empty:
            for _, row in high_up.iterrows():
                print(f"   {row['ticker']}: {row['probability']:.1%} (Confidence: {row['confidence']:.1%})")
        else:
            print("   No high confidence up predictions")
    
    def _visualize_prediction(self, name, data, raw_window, dates, result, save_path=None):
        """Visualize prediction results"""
        try:
            # Create charts
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'ATLAS Prediction Analysis: {name}', fontsize=16, fontweight='bold')
            
            # 1. Price trend chart
            ax1 = axes[0, 0]
            recent_data = data.tail(100)  # Show recent 100 days
            
            # Create continuous x-axis indices for proper alignment
            x_indices = range(len(recent_data))
            
            ax1.plot(x_indices, recent_data['Close'], 'b-', linewidth=2, label='Close Price')
            
            if 'MA5' in recent_data.columns:
                ax1.plot(x_indices, recent_data['MA5'], 'orange', alpha=0.7, label='MA5')
            if 'MA20' in recent_data.columns:
                ax1.plot(x_indices, recent_data['MA20'], 'red', alpha=0.7, label='MA20')
                
            # Mark prediction window - now aligned with the actual x-axis
            window_size = self.model_info['window_size']
            if len(recent_data) >= window_size:
                window_start = len(recent_data) - window_size
                window_end = len(recent_data)
                ax1.axvspan(window_start, window_end, alpha=0.3, 
                           color='green' if result['prediction'] == 'up' else 'red',
                           label=f'Prediction Window ({window_size} days)')
                
            # Set x-axis labels to show dates if available
            if 'Date' in recent_data.columns:
                # Show every 10th date to avoid overcrowding
                step = max(1, len(recent_data)//10)
                date_indices = range(0, len(recent_data), step)
                date_labels = []
                for i in date_indices:
                    if i < len(recent_data):
                        date_val = recent_data.iloc[i]['Date']
                        if isinstance(date_val, str):
                            # If it's already a string, truncate to show just date
                            date_labels.append(date_val[:10] if len(date_val) > 10 else date_val)
                        else:
                            # If it's a datetime object, format it
                            date_labels.append(date_val.strftime('%Y-%m-%d'))
                    else:
                        date_labels.append('')
                        
                ax1.set_xticks(date_indices)
                ax1.set_xticklabels(date_labels, rotation=45, ha='right')
            else:
                ax1.set_xlabel('Days (Recent)')
                
            ax1.set_title('Price Trend and Prediction Window')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. Technical indicators
            ax2 = axes[0, 1]
            if 'RSI' in data.columns:
                recent_rsi = recent_data['RSI']
                ax2.plot(x_indices, recent_rsi, 'purple', label='RSI')
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
                ax2.set_ylim(0, 100)
            ax2.set_title('RSI Technical Indicator')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Prediction probability table
            ax3 = axes[1, 0]
            ax3.axis('off')
            
            # Create prediction info table
            table_data = [
                ['Prediction Direction', result['prediction'].upper()],
                ['Up Probability', f"{result['probability']:.1%}"],
                ['Confidence', f"{result['confidence']:.1%}"],
                ['Prediction Time', result['prediction_date']],
                ['Data End Date', str(result.get('data_end_date', 'Unknown'))],
                ['Window Size', str(result['window_size'])]
            ]
            
            table = ax3.table(cellText=table_data, cellLoc='left', loc='center',
                             colWidths=[0.3, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            # Set table style
            for i in range(len(table_data)):
                table[(i, 0)].set_facecolor('#E6E6FA')
                table[(i, 1)].set_facecolor('#F0F8FF')
                
            ax3.set_title('Prediction Result Details', pad=20)
            
            # 4. Confidence visualization
            ax4 = axes[1, 1]
            
            # Draw confidence bar chart
            confidence = result['confidence']
            colors = ['red' if confidence < 0.4 else 'orange' if confidence < 0.7 else 'green']
            bars = ax4.bar(['Confidence'], [confidence], color=colors, alpha=0.7)
            
            # Add confidence interval lines
            ax4.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Medium Confidence')
            ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High Confidence')
            
            ax4.set_ylim(0, 1)
            ax4.set_ylabel('Confidence')
            ax4.set_title('Prediction Confidence')
            ax4.legend()
            
            # Add numerical values on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f"results/prediction_{name}_{timestamp}.png"
                
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📁 Prediction chart saved: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Visualization generation failed: {str(e)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="ATLAS Stock Prediction Inference System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Predict single stock
  python atlas_inference.py --ticker AAPL
  
  # Predict multiple stocks
  python atlas_inference.py --batch AAPL,MSFT,GOOGL,TSLA
  
  # Predict from file
  python atlas_inference.py --file data/AAPL.csv
  
  # Specify data period
  python atlas_inference.py --ticker AAPL --start 2022-01-01
        """
    )
    
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--batch", type=str, help="Batch stock tickers, comma separated (e.g., AAPL,MSFT,GOOGL)")
    parser.add_argument("--file", type=str, help="CSV file path")
    parser.add_argument("--start", type=str, default="2023-01-01",
                       help="Start date for data (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default="models/atlas_binary_model_best.pth",
                       help="Model file path")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    # Print header
    print("="*60)
    print("🚀 ATLAS Stock Prediction Inference System")
    print("Advanced Technical Learning Analysis System - Inference")
    print("="*60)
    
    try:
        # Initialize inference engine
        engine = ATLASInferenceEngine(model_path=args.model)
        
        # Execute different operations based on parameters
        if args.ticker:
            # Single stock prediction
            result = engine.predict_single_ticker(
                args.ticker, 
                start_date=args.start,
                visualize=not args.no_viz
            )
            
        elif args.batch:
            # Batch prediction
            ticker_list = [t.strip().upper() for t in args.batch.split(',')]
            result = engine.predict_batch(ticker_list, start_date=args.start)
            
        elif args.file:
            # File prediction
            result = engine.predict_from_file(args.file, visualize=not args.no_viz)
            
        else:
            # Interactive mode
            print("\n🎯 Interactive Prediction Mode")
            print("Please select prediction method:")
            print("1. Single stock prediction")
            print("2. Batch prediction")
            print("3. File prediction")
            
            choice = input("\nPlease enter choice (1-3): ").strip()
            
            if choice == "1":
                ticker = input("Please enter stock ticker: ").strip().upper()
                start_date = input("Please enter start date (default 2023-01-01): ").strip() or "2023-01-01"
                result = engine.predict_single_ticker(ticker, start_date=start_date)
                
            elif choice == "2":
                tickers_input = input("Please enter stock tickers (comma separated): ").strip()
                ticker_list = [t.strip().upper() for t in tickers_input.split(',')]
                start_date = input("Please enter start date (default 2023-01-01): ").strip() or "2023-01-01"
                result = engine.predict_batch(ticker_list, start_date=start_date)
                
            elif choice == "3":
                file_path = input("Please enter CSV file path: ").strip()
                result = engine.predict_from_file(file_path)
                
            else:
                print("❌ Invalid choice")
                return
                
        print(f"\n✅ Prediction completed!")
        print("\n💡 Disclaimer: This prediction is for reference only and does not constitute investment advice. Investment involves risks, please make decisions carefully.")
        
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)