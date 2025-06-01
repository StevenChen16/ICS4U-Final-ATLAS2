# auto_tuning.py
# Simple auto-tuning module for ATLAS system
# Inspired by nnUNet's data fingerprint approach but simplified for financial time series

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

# Import user's data module
from src.data import load_data_from_csv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataFingerprint:
    """Simple data fingerprint for ATLAS auto-tuning"""
    
    # Basic info
    n_tickers: int
    n_samples: int  
    n_features: int
    
    # Time series characteristics
    avg_volatility: float       # Average volatility across assets
    trend_strength: float       # Average trend strength (R²)
    noise_level: float          # Signal-to-noise ratio
    
    # Label characteristics  
    label_balance: float        # Up/down balance (0.5 = perfect balance)
    avg_price_change: float     # Average price change magnitude
    
    # Data quality
    missing_ratio: float        # Missing data ratio
    data_length_days: int       # Total time span in days

@dataclass  
class ATLASConfig:
    """ATLAS configuration parameters"""
    
    # Data processing
    window_size: int = 50
    stride: int = 10
    threshold: float = 0.5
    image_size: int = 50
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    patience: int = 10
    dropout_rate: float = 0.5
    validation_size: float = 0.2
    gap_size: int = 20
    
    # Advanced options
    use_specialized_kernels: bool = True

class SimpleDataAnalyzer:
    """Simple data analyzer for generating fingerprints"""
    
    def analyze_dataset(self, ticker_list: List[str], data_dir: str = "data_short") -> DataFingerprint:
        """Analyze dataset and create fingerprint"""
        
        logger.info(f"Analyzing {len(ticker_list)} tickers...")
        
        all_data = {}
        volatilities = []
        trend_strengths = []
        price_changes = []
        missing_ratios = []
        
        # Load and analyze each ticker
        for ticker in ticker_list:
            try:
                data_path = os.path.join(data_dir, f"{ticker}.csv")
                data = load_data_from_csv(data_path)
                
                if len(data) < 50:  # Skip if too little data
                    continue
                    
                all_data[ticker] = data
                
                # Calculate volatility
                returns = data['Close'].pct_change().dropna()
                vol = returns.std() * np.sqrt(252)  # Annualized
                volatilities.append(vol)
                
                # Calculate trend strength (R²)
                x = np.arange(len(data))
                slope, intercept, r_value, _, _ = stats.linregress(x, data['Close'])
                trend_strengths.append(r_value ** 2)
                
                # Calculate average price change  
                price_change = abs(data['Close'].pct_change(20).mean())  # 20-day average change
                price_changes.append(price_change)
                
                # Calculate missing data ratio
                missing_ratio = data.isnull().sum().sum() / data.size
                missing_ratios.append(missing_ratio)
                
            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No valid data files found")
        
        # Calculate aggregate statistics
        n_tickers = len(all_data)
        n_samples = int(np.mean([len(data) for data in all_data.values()]))
        n_features = list(all_data.values())[0].shape[1]
        
        avg_volatility = np.mean(volatilities)
        trend_strength = np.mean(trend_strengths)  
        avg_price_change = np.mean(price_changes)
        missing_ratio = np.mean(missing_ratios)
        
        # Calculate noise level (simplified)
        # Higher volatility relative to trend = more noise
        noise_level = avg_volatility / (trend_strength + 0.01)  # Avoid division by zero
        
        # Calculate label balance (simplified)
        # Assumes roughly balanced up/down movements for most assets
        label_balance = 0.5 - abs(avg_price_change - 0.01) * 5  # Closer to 0.5 = more balanced
        label_balance = max(0.1, min(0.9, label_balance))
        
        data_length_days = n_samples  # Approximation
        
        fingerprint = DataFingerprint(
            n_tickers=n_tickers,
            n_samples=n_samples,
            n_features=n_features,
            avg_volatility=avg_volatility,
            trend_strength=trend_strength,
            noise_level=noise_level,
            label_balance=label_balance,
            avg_price_change=avg_price_change,
            missing_ratio=missing_ratio,
            data_length_days=data_length_days
        )
        
        self._log_fingerprint(fingerprint)
        return fingerprint
    
    def _log_fingerprint(self, fp: DataFingerprint):
        """Log fingerprint summary"""
        logger.info("=== Data Fingerprint ===")
        logger.info(f"Tickers: {fp.n_tickers}, Samples: {fp.n_samples}, Features: {fp.n_features}")
        logger.info(f"Avg Volatility: {fp.avg_volatility:.3f}")
        logger.info(f"Trend Strength: {fp.trend_strength:.3f}")
        logger.info(f"Noise Level: {fp.noise_level:.3f}")
        logger.info(f"Label Balance: {fp.label_balance:.3f}")
        logger.info(f"Missing Data: {fp.missing_ratio:.3%}")

class SimpleRuleConfigurator:
    """Simple rule-based configurator"""
    
    def configure(self, fingerprint: DataFingerprint) -> ATLASConfig:
        """Generate configuration based on data fingerprint"""
        
        logger.info("Generating configuration based on rules...")
        
        config = ATLASConfig()  # Start with defaults
        
        # Rule 1: Window size based on data length and volatility
        if fingerprint.n_samples < 500:
            config.window_size = 30  # Smaller window for limited data
        elif fingerprint.avg_volatility > 0.4:
            config.window_size = 60  # Larger window for high volatility
        elif fingerprint.avg_volatility < 0.2:
            config.window_size = 40  # Smaller window for low volatility
        # else keep default 50
        
        # Rule 2: Stride based on data amount
        if fingerprint.n_samples < 1000:
            config.stride = 5   # Smaller stride for limited data
        elif fingerprint.n_samples > 5000:
            config.stride = 15  # Larger stride for abundant data
        # else keep default 10
        
        # Rule 3: Threshold based on volatility and price change
        if fingerprint.avg_volatility > 0.4:
            config.threshold = 0.02  # Higher threshold for volatile data
        elif fingerprint.avg_volatility < 0.2:
            config.threshold = 0.005 # Lower threshold for stable data
        elif fingerprint.avg_price_change > 0.02:
            config.threshold = 0.015 # Adjust for large price movements
        # else keep default 0.01
        
        # Rule 4: Batch size based on data amount
        total_samples = fingerprint.n_tickers * fingerprint.n_samples
        if total_samples < 1000:
            config.batch_size = 16
        elif total_samples > 10000:
            config.batch_size = 64
        # else keep default 32
        
        # Rule 5: Learning rate based on noise level
        if fingerprint.noise_level > 3.0:
            config.learning_rate = 0.0005  # Lower LR for noisy data
        elif fingerprint.noise_level < 1.0:
            config.learning_rate = 0.002   # Higher LR for clean data
        # else keep default 0.001
        
        # Rule 6: Dropout based on data amount and noise
        if fingerprint.n_samples < 1000:
            config.dropout_rate = 0.6   # More regularization for small data
        elif fingerprint.noise_level > 2.5:
            config.dropout_rate = 0.6   # More regularization for noisy data
        elif fingerprint.n_samples > 5000 and fingerprint.noise_level < 1.5:
            config.dropout_rate = 0.3   # Less regularization for large clean data
        # else keep default 0.5
        
        # Rule 7: Training epochs based on data characteristics
        if fingerprint.n_samples < 1000:
            config.epochs = 80          # More epochs for small data
        elif fingerprint.label_balance < 0.3:
            config.epochs = 70          # More epochs for imbalanced data
        elif fingerprint.n_samples > 5000:
            config.epochs = 40          # Fewer epochs for large data
        # else keep default 50
        
        # Rule 8: Early stopping patience
        config.patience = max(10, config.epochs // 5)
        
        # Rule 9: Gap size based on window size
        config.gap_size = max(10, config.window_size // 3)
        
        # Rule 10: Image size should not exceed window size
        config.image_size = min(config.window_size, 64)
        
        self._log_configuration_choices(fingerprint, config)
        return config
    
    def _log_configuration_choices(self, fp: DataFingerprint, config: ATLASConfig):
        """Log configuration choices and reasoning"""
        logger.info("=== Configuration Choices ===")
        logger.info(f"Window Size: {config.window_size} (based on samples={fp.n_samples}, vol={fp.avg_volatility:.3f})")
        logger.info(f"Stride: {config.stride} (based on data amount)")
        logger.info(f"Threshold: {config.threshold} (based on volatility and price change)")
        logger.info(f"Batch Size: {config.batch_size} (based on total samples)")
        logger.info(f"Learning Rate: {config.learning_rate} (based on noise level={fp.noise_level:.3f})")
        logger.info(f"Dropout: {config.dropout_rate} (based on data size and noise)")
        logger.info(f"Epochs: {config.epochs} (based on data characteristics)")

class ATLASAutoTuner:
    """Main auto-tuning controller for ATLAS"""
    
    def __init__(self, config_save_dir: str = "auto_configs"):
        self.config_save_dir = config_save_dir
        os.makedirs(config_save_dir, exist_ok=True)
        
        self.analyzer = SimpleDataAnalyzer()
        self.configurator = SimpleRuleConfigurator()
    
    def auto_tune(self, ticker_list: List[str], 
                  data_dir: str = "data_short",
                  save_config: bool = True) -> ATLASConfig:
        """Main auto-tuning process"""
        
        logger.info("🚀 ATLAS Auto-Tuning Started")
        logger.info("=" * 50)
        
        # Step 1: Analyze data and create fingerprint
        logger.info("📊 Step 1: Data Analysis")
        fingerprint = self.analyzer.analyze_dataset(ticker_list, data_dir)
        
        # Step 2: Generate configuration based on rules
        logger.info("\n⚙️ Step 2: Configuration Generation") 
        config = self.configurator.configure(fingerprint)
        
        # Step 3: Save configuration (optional)
        if save_config:
            self._save_config(fingerprint, config, ticker_list)
        
        logger.info("\n✅ Auto-Tuning Complete!")
        logger.info("=" * 50)
        
        return config
    
    def _save_config(self, fingerprint: DataFingerprint, 
                    config: ATLASConfig, ticker_list: List[str]):
        """Save configuration to file"""
        
        config_data = {
            'fingerprint': asdict(fingerprint),
            'configuration': asdict(config),
            'ticker_list': ticker_list,
            'version': '1.0',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        config_file = os.path.join(
            self.config_save_dir, 
            f"atlas_auto_config_{timestamp}.json"
        )
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Configuration saved to: {config_file}")

def load_config(config_file: str) -> Tuple[DataFingerprint, ATLASConfig]:
    """Load saved configuration"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    fingerprint = DataFingerprint(**config_data['fingerprint'])
    configuration = ATLASConfig(**config_data['configuration'])
    
    return fingerprint, configuration

# Simple integration function for atlas2.py
def get_auto_config(ticker_list: List[str], 
                   data_dir: str = "data_short") -> Dict:
    """
    Simple function to get auto-tuned configuration for ATLAS
    Returns dict that can be directly used in run_atlas_binary_pipeline
    """
    
    tuner = ATLASAutoTuner()
    config = tuner.auto_tune(ticker_list, data_dir)
    
    # Convert to dict format expected by atlas2.py
    return {
        'window_size': config.window_size,
        'stride': config.stride,
        'threshold': config.threshold,
        'image_size': config.image_size,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'epochs': config.epochs,
        'patience': config.patience,
        'dropout_rate': config.dropout_rate,
        'validation_size': config.validation_size,
        'gap_size': config.gap_size,
        'use_specialized_kernels': config.use_specialized_kernels
    }

if __name__ == "__main__":
    # Test the auto-tuning system
    ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    tuner = ATLASAutoTuner()
    config = tuner.auto_tune(ticker_list)
    
    print("\n🎯 Auto-tuned Configuration:")
    print(f"Window Size: {config.window_size}")
    print(f"Stride: {config.stride}")
    print(f"Threshold: {config.threshold}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")