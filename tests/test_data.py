"""
Tests for src/data.py module
"""

import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import patch, MagicMock

from src.data import (
    rolling_normalize,
    calculate_connors_rsi,
    apply_kalman_filter,
    apply_fft_filter_rolling,
    download_and_prepare_data,
    load_data_from_csv
)


class TestRollingNormalize:
    """Test rolling_normalize function"""
    
    def test_rolling_normalize_basic(self):
        """Test basic functionality"""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = rolling_normalize(series, window=3)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        assert not result.isna().any()
    
    def test_rolling_normalize_window_size(self):
        """Test different window sizes"""
        series = pd.Series(np.random.randn(100))
        
        for window in [5, 10, 21]:
            result = rolling_normalize(series, window=window)
            assert len(result) == len(series)
            assert not result.isna().any()
    
    def test_rolling_normalize_edge_cases(self):
        """Test edge cases"""
        # Very small series
        small_series = pd.Series([1, 2, 3])
        result = rolling_normalize(small_series, window=5)
        assert len(result) == 3
        
        # Constant series (zero std)
        constant_series = pd.Series([5] * 10)
        result = rolling_normalize(constant_series, window=3)
        assert not result.isna().any()
        assert np.allclose(result, 0, atol=1e-6)
    
    def test_rolling_normalize_with_nans(self):
        """Test with NaN values"""
        series = pd.Series([1, 2, np.nan, 4, 5, 6])
        result = rolling_normalize(series, window=3)
        
        assert len(result) == len(series)
        # Should handle NaNs gracefully


class TestCalculateConnorsRsi:
    """Test calculate_connors_rsi function"""
    
    def test_connors_rsi_basic(self, sample_stock_data):
        """Test basic Connors RSI calculation"""
        result = calculate_connors_rsi(sample_stock_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
        assert result.min() >= 0
        assert result.max() <= 100
    
    def test_connors_rsi_parameters(self, sample_stock_data):
        """Test with different parameters"""
        result1 = calculate_connors_rsi(
            sample_stock_data, 
            rsi_period=3, 
            streak_period=2, 
            rank_period=100
        )
        result2 = calculate_connors_rsi(
            sample_stock_data, 
            rsi_period=5, 
            streak_period=3, 
            rank_period=50
        )
        
        assert not result1.equals(result2)
        assert len(result1) == len(result2)
    
    def test_connors_rsi_insufficient_data(self):
        """Test with insufficient data"""
        small_data = pd.DataFrame({
            'Close': np.array([100, 101, 99, 102, 98], dtype=np.float64)
        })
        
        result = calculate_connors_rsi(small_data)
        assert isinstance(result, pd.Series)


class TestApplyKalmanFilter:
    """Test apply_kalman_filter function"""
    
    def test_kalman_filter_basic(self, sample_stock_data):
        """Test basic Kalman filter functionality"""
        filtered_prices, trends = apply_kalman_filter(sample_stock_data)
        
        assert isinstance(filtered_prices, pd.Series)
        assert isinstance(trends, pd.Series)
        assert len(filtered_prices) == len(sample_stock_data)
        assert len(trends) == len(sample_stock_data)
    
    def test_kalman_filter_parameters(self, sample_stock_data):
        """Test with different noise parameters"""
        filtered1, trends1 = apply_kalman_filter(
            sample_stock_data, 
            measurement_noise=0.1, 
            process_noise=0.01
        )
        filtered2, trends2 = apply_kalman_filter(
            sample_stock_data, 
            measurement_noise=0.5, 
            process_noise=0.05
        )
        
        assert not filtered1.equals(filtered2)
        assert not trends1.equals(trends2)
    
    def test_kalman_filter_smoothing(self, sample_stock_data):
        """Test that Kalman filter provides smoothing"""
        original_prices = sample_stock_data['Close']
        filtered_prices, _ = apply_kalman_filter(sample_stock_data)
        
        # Filtered prices should generally be smoother
        original_volatility = original_prices.diff().std()
        filtered_volatility = filtered_prices.diff().std()
        
        assert filtered_volatility <= original_volatility


class TestApplyFftFilterRolling:
    """Test apply_fft_filter_rolling function"""
    
    def test_fft_filter_basic(self, sample_stock_data):
        """Test basic FFT filtering"""
        result = apply_fft_filter_rolling(sample_stock_data, cutoff_period=21)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
    
    def test_fft_filter_series_input(self):
        """Test with Series input"""
        series = pd.Series(np.random.randn(200))
        result = apply_fft_filter_rolling(series, cutoff_period=21)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
    
    def test_fft_filter_different_periods(self, sample_stock_data):
        """Test with different cutoff periods"""
        result1 = apply_fft_filter_rolling(sample_stock_data, cutoff_period=10)
        result2 = apply_fft_filter_rolling(sample_stock_data, cutoff_period=50)
        
        assert not result1.equals(result2)
        assert len(result1) == len(result2)
    
    def test_fft_filter_window_size(self, sample_stock_data):
        """Test with different window sizes"""
        result1 = apply_fft_filter_rolling(
            sample_stock_data, 
            cutoff_period=21, 
            window_size=100
        )
        result2 = apply_fft_filter_rolling(
            sample_stock_data, 
            cutoff_period=21, 
            window_size=200
        )
        
        assert len(result1) == len(result2)
    
    def test_fft_filter_small_data(self):
        """Test with small dataset"""
        small_data = pd.DataFrame({
            'Close': np.random.randn(30)
        })
        
        result = apply_fft_filter_rolling(small_data, cutoff_period=21)
        assert len(result) == len(small_data)


class TestDownloadAndPrepareData:
    """Test download_and_prepare_data function"""
    
    @patch('src.data.yf.download')
    def test_download_and_prepare_data_success(self, mock_download, mock_yfinance_data):
        """Test successful data download and preparation"""
        # Mock yfinance download
        mock_ticker = mock_yfinance_data('AAPL')
        mock_data = mock_ticker.history()
        mock_download.return_value = mock_data
        
        result = download_and_prepare_data(
            'AAPL', 
            '2023-01-01', 
            '2023-12-31'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Close' in result.columns
        assert 'MA5' in result.columns
        assert 'RSI' in result.columns
        assert 'CRSI' in result.columns
    
    @patch('src.data.yf.download')
    def test_download_and_prepare_data_empty(self, mock_download):
        """Test handling of empty data"""
        mock_download.return_value = pd.DataFrame()
        
        with pytest.raises(ValueError, match="No data downloaded"):
            download_and_prepare_data('INVALID', '2023-01-01', '2023-12-31')
    
    @patch('src.data.yf.download')
    def test_download_and_prepare_data_multiindex(self, mock_download, mock_yfinance_data):
        """Test handling of MultiIndex columns"""
        mock_ticker = mock_yfinance_data('AAPL')
        mock_data = mock_ticker.history()
        
        # Create MultiIndex columns
        mock_data.columns = pd.MultiIndex.from_product([mock_data.columns, ['AAPL']])
        mock_download.return_value = mock_data
        
        result = download_and_prepare_data('AAPL', '2023-01-01', '2023-12-31')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestLoadDataFromCsv:
    """Test load_data_from_csv function"""
    
    def test_load_data_from_csv_success(self, temp_csv_file):
        """Test successful CSV loading"""
        result = load_data_from_csv(temp_csv_file)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_load_data_from_csv_nonexistent(self):
        """Test loading nonexistent file"""
        with pytest.raises(FileNotFoundError):
            load_data_from_csv('nonexistent_file.csv')
    
    def test_load_data_from_csv_index_type(self, temp_csv_file):
        """Test that Date column becomes datetime index"""
        result = load_data_from_csv(temp_csv_file)
        
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == 'Date'


class TestIntegration:
    """Integration tests for data module"""
    
    @patch('src.data.yf.download')
    def test_full_pipeline(self, mock_download, mock_yfinance_data):
        """Test the full data processing pipeline"""
        # Mock data
        mock_ticker = mock_yfinance_data('AAPL')
        mock_data = mock_ticker.history()
        mock_download.return_value = mock_data
        
        # Process data
        result = download_and_prepare_data('AAPL', '2023-01-01', '2023-12-31')
        
        # Check all expected columns are present
        expected_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI', 'Upper', 'Middle', 'Lower', 'Volume_MA5',
            'CRSI', 'Kalman_Price', 'Kalman_Trend',
            'FFT_21', 'FFT_63'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
        
        # Check data quality
        assert not result.isna().any().any(), "Data contains NaN values"
        assert len(result) > 0, "Result is empty"
    
    def test_technical_indicators_consistency(self, sample_stock_data):
        """Test that technical indicators are calculated consistently"""
        # Apply processing
        df = sample_stock_data.copy()
        
        # Calculate some indicators manually for comparison
        ma5_manual = df['Close'].rolling(5).mean()
        
        # Process with the function
        from src.data import rolling_normalize
        df['MA5'] = rolling_normalize(df['Close'].rolling(5).mean())
        
        # The shapes should match
        assert len(df['MA5']) == len(ma5_manual)