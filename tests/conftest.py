"""
Pytest configuration and fixtures for ATLAS tests
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from datetime import datetime, timedelta
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)
    
    # Generate realistic stock price data
    np.random.seed(42)
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # Daily returns
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data - ensure float64 type for talib compatibility
    closes = np.array(prices, dtype=np.float64)
    opens = closes * (1 + np.random.normal(0, 0.005, n_days))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    volumes = np.random.randint(1000000, 10000000, n_days).astype(np.float64)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    return df.set_index('Date')


@pytest.fixture
def processed_stock_data(sample_stock_data):
    """Create processed stock data with technical indicators"""
    from src.data import rolling_normalize
    
    df = sample_stock_data.copy()
    
    # Add some basic technical indicators
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = 50.0  # Simplified RSI
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['Middle'] = df['Close'].rolling(20).mean()
    df['Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['CRSI'] = 50.0  # Simplified Connors RSI
    df['Kalman_Price'] = df['Close']  # Simplified
    df['Kalman_Trend'] = 0.0
    df['FFT_21'] = df['Close']  # Simplified
    df['FFT_63'] = df['Close']  # Simplified
    
    # Apply normalization
    for col in df.columns:
        if col in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20']:
            df[col] = rolling_normalize(df[col])
    
    return df.dropna()


@pytest.fixture
def temp_csv_file(sample_stock_data):
    """Create a temporary CSV file with stock data"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_stock_data.to_csv(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_data_dir(sample_stock_data):
    """Create a temporary directory with multiple stock CSV files"""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    for ticker in tickers:
        file_path = os.path.join(temp_dir, f'{ticker}.csv')
        sample_stock_data.to_csv(file_path)
    
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_windows():
    """Create sample time series windows for testing"""
    np.random.seed(42)
    n_samples = 100
    window_size = 50
    n_features = 10
    
    windows = np.random.randn(n_samples, window_size, n_features)
    labels = np.random.choice(['up', 'down'], n_samples)
    indices = np.arange(n_samples) * 10  # Simulated time indices
    
    return windows, labels, indices


@pytest.fixture
def sample_images():
    """Create sample transformed images for testing"""
    np.random.seed(42)
    n_samples = 20
    image_size = 50
    n_channels = 4
    
    images = np.random.rand(n_samples, image_size, image_size, n_channels)
    return images


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance data for testing"""
    class MockTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            
        def history(self, start=None, end=None, progress=False):
            # Return sample data
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            np.random.seed(42)
            
            data = pd.DataFrame({
                'Open': 100 + np.random.randn(100).cumsum(),
                'High': 102 + np.random.randn(100).cumsum(),
                'Low': 98 + np.random.randn(100).cumsum(),
                'Close': 100 + np.random.randn(100).cumsum(),
                'Volume': np.random.randint(1000000, 10000000, 100)
            }, index=dates)
            
            return data
    
    return MockTicker


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'window_size': 50,
        'stride': 10,
        'threshold': 0.5,
        'image_size': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 5,  # Small for testing
        'patience': 3,
        'dropout_rate': 0.5,
        'validation_size': 0.2,
        'gap_size': 10,
        'use_specialized_kernels': True
    }


@pytest.fixture
def device():
    """Get device for testing (CPU)"""
    return torch.device('cpu')