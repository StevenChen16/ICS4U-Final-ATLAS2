"""
Data Fingerprint Extractor for ATLAS MoE System

This module implements the data fingerprinting system for routing time series
to appropriate experts based on their market characteristics.

Author: Steven Chen
Date: 2025-06-14
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from scipy import stats
from sklearn.linear_model import LinearRegression


class DataFingerprinter:
    """Extract market characteristics fingerprint from time series data"""
    
    def __init__(self):
        """Initialize the fingerprinter"""
        pass
    
    def extract_fingerprint(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract comprehensive data fingerprint from OHLCV data
        
        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            
        Returns:
            Dictionary containing market characteristics
        """
        if len(df) < 30:
            raise ValueError("Need at least 30 data points for fingerprint extraction")
        
        close_prices = df['Close'].values
        high_prices = df['High'].values  
        low_prices = df['Low'].values
        volume = df['Volume'].values if 'Volume' in df.columns else None
        
        fingerprint = {}
        
        # 1. Volatility metrics
        fingerprint['vol30'] = self._realized_volatility(close_prices, window=30)
        fingerprint['vol7'] = self._realized_volatility(close_prices, window=7)
        fingerprint['vol_ratio'] = fingerprint['vol7'] / fingerprint['vol30'] if fingerprint['vol30'] > 0 else 1.0
        
        # 2. Trend strength
        fingerprint['trend_r2'] = self._linear_trend_r2(close_prices, window=30)
        fingerprint['trend_slope'] = self._trend_slope(close_prices, window=30)
        
        # 3. Hurst exponent (long-term memory)
        fingerprint['hurst'] = self._hurst_exponent(close_prices)
        
        # 4. Price range characteristics
        fingerprint['atr'] = self._average_true_range(high_prices, low_prices, close_prices)
        fingerprint['atr_pct'] = fingerprint['atr'] / np.mean(close_prices) if np.mean(close_prices) > 0 else 0
        
        # 5. Volume characteristics (if available)
        if volume is not None:
            fingerprint['volume_z'] = self._zscore(volume[-30:]) if len(volume) >= 30 else 0
            fingerprint['volume_trend'] = self._linear_trend_r2(volume, window=min(len(volume), 30))
        else:
            fingerprint['volume_z'] = 0
            fingerprint['volume_trend'] = 0
        
        # 6. Market microstructure (spread estimation)
        fingerprint['spread_estimate'] = self._estimate_spread(high_prices, low_prices, close_prices)
        
        # 7. Distribution characteristics
        returns = np.diff(close_prices) / close_prices[:-1]
        fingerprint['skewness'] = stats.skew(returns) if len(returns) > 0 else 0
        fingerprint['kurtosis'] = stats.kurtosis(returns) if len(returns) > 0 else 0
        
        # 8. Momentum characteristics
        fingerprint['momentum_10'] = self._momentum(close_prices, 10)
        fingerprint['momentum_5'] = self._momentum(close_prices, 5)
        
        # 9. Regime detection
        fingerprint['regime_stability'] = self._regime_stability(close_prices)
        
        return fingerprint
    
    def _realized_volatility(self, prices: np.ndarray, window: int = 30) -> float:
        """Calculate realized volatility"""
        if len(prices) < window:
            window = len(prices)
        
        returns = np.diff(prices[-window:]) / prices[-window:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized
    
    def _linear_trend_r2(self, prices: np.ndarray, window: int = 30) -> float:
        """Calculate R² of linear trend"""
        if len(prices) < window:
            window = len(prices)
        
        y = prices[-window:]
        x = np.arange(len(y)).reshape(-1, 1)
        
        if len(y) < 2:
            return 0.0
        
        try:
            model = LinearRegression()
            model.fit(x, y)
            return model.score(x, y)
        except:
            return 0.0
    
    def _trend_slope(self, prices: np.ndarray, window: int = 30) -> float:
        """Calculate trend slope (normalized)"""
        if len(prices) < window:
            window = len(prices)
        
        y = prices[-window:]
        x = np.arange(len(y))
        
        if len(y) < 2:
            return 0.0
        
        try:
            slope, _ = np.polyfit(x, y, 1)
            return slope / np.mean(y) if np.mean(y) > 0 else 0  # Normalized slope
        except:
            return 0.0
    
    def _hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        if len(prices) < 10:
            return 0.5  # Random walk default
        
        try:
            # Use log prices for better stability
            log_prices = np.log(prices)
            n = len(log_prices)
            
            # Calculate increments
            increments = np.diff(log_prices)
            
            # Choose scales
            scales = np.unique(np.logspace(1, np.log10(n//4), 10).astype(int))
            scales = scales[scales <= len(increments)]
            
            if len(scales) < 3:
                return 0.5
            
            rs_values = []
            
            for scale in scales:
                if scale >= len(increments):
                    continue
                    
                # Split into non-overlapping windows
                num_windows = len(increments) // scale
                if num_windows == 0:
                    continue
                
                rs_window = []
                for i in range(num_windows):
                    window = increments[i*scale:(i+1)*scale]
                    if len(window) == 0:
                        continue
                    
                    # Mean-adjusted cumulative sum
                    mean_increment = np.mean(window)
                    deviations = np.cumsum(window - mean_increment)
                    
                    # Range
                    R = np.max(deviations) - np.min(deviations)
                    
                    # Standard deviation
                    S = np.std(window)
                    
                    if S > 0:
                        rs_window.append(R / S)
                
                if len(rs_window) > 0:
                    rs_values.append(np.mean(rs_window))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Linear regression on log-log plot
            log_scales = np.log(scales[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Remove infinite/nan values
            valid_mask = np.isfinite(log_scales) & np.isfinite(log_rs)
            if np.sum(valid_mask) < 3:
                return 0.5
                
            log_scales = log_scales[valid_mask]
            log_rs = log_rs[valid_mask]
            
            hurst, _ = np.polyfit(log_scales, log_rs, 1)
            
            # Clip to reasonable range
            return np.clip(hurst, 0.0, 1.0)
            
        except:
            return 0.5  # Default to random walk
    
    def _average_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> float:
        """Calculate Average True Range"""
        if len(high) < window or len(low) < window or len(close) < window:
            window = min(len(high), len(low), len(close))
        
        if window < 2:
            return 0.0
        
        # True Range calculation
        tr_list = []
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr_list.append(max(tr1, tr2, tr3))
        
        if len(tr_list) == 0:
            return 0.0
        
        # Average over window
        return np.mean(tr_list[-window:])
    
    def _zscore(self, values: np.ndarray) -> float:
        """Calculate Z-score of most recent value"""
        if len(values) < 2:
            return 0.0
        
        mean_val = np.mean(values[:-1])  # Exclude last value from mean
        std_val = np.std(values[:-1])
        
        if std_val == 0:
            return 0.0
        
        return (values[-1] - mean_val) / std_val
    
    def _estimate_spread(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Estimate bid-ask spread from OHLC data"""
        if len(high) < 5 or len(low) < 5 or len(close) < 5:
            return 0.0
        
        # Use recent data
        recent_high = high[-10:]
        recent_low = low[-10:]
        recent_close = close[-10:]
        
        # Estimate spread as percentage of price
        spreads = (recent_high - recent_low) / recent_close
        
        # Use median to avoid outliers
        return np.median(spreads)
    
    def _momentum(self, prices: np.ndarray, window: int) -> float:
        """Calculate momentum (rate of change)"""
        if len(prices) < window + 1:
            return 0.0
        
        return (prices[-1] - prices[-(window+1)]) / prices[-(window+1)]
    
    def _regime_stability(self, prices: np.ndarray, window: int = 20) -> float:
        """Measure regime stability using rolling correlation"""
        if len(prices) < window * 2:
            return 0.5  # Neutral stability
        
        try:
            # Split into two halves
            mid = len(prices) // 2
            first_half = prices[:mid]
            second_half = prices[mid:]
            
            # Ensure equal length
            min_len = min(len(first_half), len(second_half))
            first_half = first_half[-min_len:]
            second_half = second_half[-min_len:]
            
            if min_len < 5:
                return 0.5
            
            # Calculate correlation between regimes
            correlation = np.corrcoef(first_half, second_half)[0, 1]
            
            if np.isnan(correlation):
                return 0.5
            
            return abs(correlation)  # Higher means more stable
            
        except:
            return 0.5


def classify_market_archetype(fingerprint: Dict[str, float]) -> str:
    """
    Classify market archetype based on fingerprint
    
    Args:
        fingerprint: Market characteristics dictionary
        
    Returns:
        Expert name string
    """
    vol30 = fingerprint.get('vol30', 0)
    trend_r2 = fingerprint.get('trend_r2', 0)
    spread = fingerprint.get('spread_estimate', 0)
    volume_z = fingerprint.get('volume_z', 0)
    hurst = fingerprint.get('hurst', 0.5)
    atr_pct = fingerprint.get('atr_pct', 0)
    
    # Rule-based classification as per design document
    if vol30 >= 0.06 and spread <= 0.0005 and volume_z >= 3:
        return 'crypto_hft'
    elif 0.01 <= vol30 < 0.03 and trend_r2 >= 0.6:
        return 'equity_intraday' 
    elif 0.005 <= vol30 < 0.02 and atr_pct < 0.05:
        return 'equity_daily'
    elif trend_r2 >= 0.8 and hurst > 0.55:
        return 'futures_trend'
    elif vol30 < 0.005 and volume_z < -1:
        return 'low_vol_etf'
    else:
        return 'equity_daily'  # Default expert


def get_expert_weights(fingerprint: Dict[str, float]) -> Dict[str, float]:
    """
    Get soft weights for all experts based on fingerprint
    
    Args:
        fingerprint: Market characteristics dictionary
        
    Returns:
        Dictionary mapping expert names to weights (sum to 1.0)
    """
    experts = ['crypto_hft', 'equity_intraday', 'equity_daily', 'futures_trend', 'low_vol_etf']
    weights = {}
    
    vol30 = fingerprint.get('vol30', 0)
    trend_r2 = fingerprint.get('trend_r2', 0)
    spread = fingerprint.get('spread_estimate', 0)
    volume_z = fingerprint.get('volume_z', 0)
    hurst = fingerprint.get('hurst', 0.5)
    atr_pct = fingerprint.get('atr_pct', 0)
    
    # Calculate affinity scores
    weights['crypto_hft'] = _sigmoid(vol30 - 0.06) * _sigmoid(-spread + 0.0005) * _sigmoid(volume_z - 3)
    weights['equity_intraday'] = _sigmoid(vol30 - 0.01) * _sigmoid(-vol30 + 0.03) * _sigmoid(trend_r2 - 0.6)
    weights['equity_daily'] = _sigmoid(vol30 - 0.005) * _sigmoid(-vol30 + 0.02) * _sigmoid(-atr_pct + 0.05)
    weights['futures_trend'] = _sigmoid(trend_r2 - 0.8) * _sigmoid(hurst - 0.55)
    weights['low_vol_etf'] = _sigmoid(-vol30 + 0.005) * _sigmoid(-volume_z - 1)
    
    # Add baseline for equity_daily as default
    weights['equity_daily'] += 0.1
    
    # Normalize
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        # Fallback to equal weights
        weights = {expert: 1.0 / len(experts) for expert in experts}
    
    return weights


def _sigmoid(x: float, steepness: float = 10.0) -> float:
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-steepness * x))


# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Create test data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # High volatility crypto-like data
    close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.1, 100)))
    high_prices = close_prices * (1 + np.random.uniform(0, 0.05, 100))
    low_prices = close_prices * (1 - np.random.uniform(0, 0.05, 100))
    open_prices = close_prices + np.random.normal(0, 1, 100)
    volume = np.random.lognormal(10, 1, 100)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices, 
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    })
    
    # Test fingerprinting
    fingerprinter = DataFingerprinter()
    fingerprint = fingerprinter.extract_fingerprint(df)
    
    print("Market Fingerprint:")
    for key, value in fingerprint.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nClassified as: {classify_market_archetype(fingerprint)}")
    
    print("\nExpert weights:")
    weights = get_expert_weights(fingerprint)
    for expert, weight in weights.items():
        print(f"  {expert}: {weight:.3f}")