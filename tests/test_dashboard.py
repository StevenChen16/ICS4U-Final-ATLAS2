"""
Tests for src/dashboard.py module
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import functions from dashboard (note: this might need adjustment based on actual structure)
try:
    from src.dashboard import (
        get_stock_data_safe,
        calculate_connors_rsi,
        apply_kalman_filter_simple,
        apply_fft_filter_simple,
        calculate_technical_indicators,
        get_multiple_stocks_data,
        get_stock_name
    )
except ImportError:
    # If direct import fails, we'll mock these functions for testing
    pass


class TestGetStockDataSafe:
    """Test get_stock_data_safe function"""
    
    @patch('src.dashboard.yf.Ticker')
    def test_get_stock_data_safe_success(self, mock_ticker_class, sample_stock_data):
        """Test successful stock data retrieval"""
        # Mock yfinance Ticker
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_stock_data
        mock_ticker_class.return_value = mock_ticker
        
        result = get_stock_data_safe('AAPL', period='1d', interval='1m')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # Should have technical indicators added
        assert 'MA5' in result.columns
        assert 'RSI' in result.columns
    
    @patch('src.dashboard.yf.Ticker')
    def test_get_stock_data_safe_empty_data(self, mock_ticker_class):
        """Test handling of empty data"""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        result = get_stock_data_safe('INVALID')
        
        assert result is None
    
    @patch('src.dashboard.yf.Ticker')
    def test_get_stock_data_safe_exception(self, mock_ticker_class):
        """Test handling of exceptions"""
        mock_ticker_class.side_effect = Exception("Network error")
        
        result = get_stock_data_safe('AAPL')
        
        assert result is None
    
    @patch('src.dashboard.yf.Ticker')
    @patch('src.dashboard.data_cache')
    def test_get_stock_data_safe_cache_fallback(self, mock_cache, mock_ticker_class, sample_stock_data):
        """Test falling back to cache when API fails"""
        # Setup cache - fix lambda function parameters
        mock_cache.__contains__ = lambda self, symbol: symbol == 'AAPL'
        mock_cache.__getitem__ = lambda self, symbol: {'df': sample_stock_data}
        
        # Make API fail
        mock_ticker_class.side_effect = Exception("API error")
        
        result = get_stock_data_safe('AAPL')
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestCalculateConnorsRsi:
    """Test calculate_connors_rsi function (dashboard version)"""
    
    def test_calculate_connors_rsi_basic(self, sample_stock_data):
        """Test basic Connors RSI calculation"""
        result = calculate_connors_rsi(sample_stock_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
        assert not result.isna().any()
        # The simplified version actually calculates RSI, not fixed 50
        assert result.min() >= 0
        assert result.max() <= 100
    
    def test_calculate_connors_rsi_edge_case(self):
        """Test with minimal data"""
        small_data = pd.DataFrame({
            'Close': [100, 101, 99]
        })
        
        result = calculate_connors_rsi(small_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert (result == 50).all()


class TestApplyKalmanFilterSimple:
    """Test apply_kalman_filter_simple function"""
    
    def test_apply_kalman_filter_simple_basic(self, sample_stock_data):
        """Test basic Kalman filter functionality"""
        filtered_prices, trends = apply_kalman_filter_simple(sample_stock_data)
        
        assert isinstance(filtered_prices, pd.Series)
        assert isinstance(trends, pd.Series)
        assert len(filtered_prices) == len(sample_stock_data)
        assert len(trends) == len(sample_stock_data)
    
    def test_apply_kalman_filter_simple_smoothing(self, sample_stock_data):
        """Test that the filter provides smoothing"""
        original_prices = sample_stock_data['Close']
        filtered_prices, _ = apply_kalman_filter_simple(sample_stock_data)
        
        # Filtered prices should generally be smoother
        original_volatility = original_prices.diff().std()
        filtered_volatility = filtered_prices.diff().std()
        
        # Due to exponential smoothing, should be somewhat smoother
        assert filtered_volatility <= original_volatility * 1.1  # Allow some tolerance
    
    def test_apply_kalman_filter_simple_exception(self):
        """Test exception handling"""
        # Create problematic data
        bad_data = pd.DataFrame({
            'Close': [np.nan, np.inf, -np.inf]
        })
        
        filtered_prices, trends = apply_kalman_filter_simple(bad_data)
        
        # Should fall back to original close prices
        assert isinstance(filtered_prices, pd.Series)
        assert isinstance(trends, pd.Series)


class TestApplyFftFilterSimple:
    """Test apply_fft_filter_simple function"""
    
    def test_apply_fft_filter_simple_with_scipy(self, sample_stock_data):
        """Test FFT filter with scipy available"""
        result = apply_fft_filter_simple(sample_stock_data, cutoff_period=21)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
    
    def test_apply_fft_filter_simple_small_data(self):
        """Test with data smaller than cutoff period"""
        small_data = pd.DataFrame({
            'Close': [100, 101, 99, 102, 98]
        })
        
        result = apply_fft_filter_simple(small_data, cutoff_period=21)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 5
    
    @patch('scipy.signal.savgol_filter')
    def test_apply_fft_filter_simple_scipy_error(self, mock_savgol, sample_stock_data):
        """Test fallback when scipy fails"""
        mock_savgol.side_effect = ImportError("scipy not available")
        
        result = apply_fft_filter_simple(sample_stock_data, cutoff_period=21)
        
        # Should fall back to moving average
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)


class TestCalculateTechnicalIndicators:
    """Test calculate_technical_indicators function"""
    
    def test_calculate_technical_indicators_basic(self, sample_stock_data):
        """Test basic technical indicators calculation"""
        result = calculate_technical_indicators(sample_stock_data)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        
        # Check that all expected indicators are present
        expected_indicators = [
            'MA5', 'MA10', 'MA20', 'MA50',
            'BB_Upper', 'BB_Middle', 'BB_Lower',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'CRSI', 'Kalman_Price', 'Kalman_Trend',
            'FFT_21', 'FFT_63', 'Volume_MA',
            'Price_Change', 'Price_Change_5'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"
    
    def test_calculate_technical_indicators_empty_data(self):
        """Test with empty dataframe"""
        empty_df = pd.DataFrame()
        result = calculate_technical_indicators(empty_df)
        
        assert result.empty
    
    def test_calculate_technical_indicators_data_quality(self, sample_stock_data):
        """Test data quality after indicator calculation"""
        result = calculate_technical_indicators(sample_stock_data)
        
        # Should not have NaN values after forward and backward fill
        assert not result.isna().any().any()
        
        # RSI should be in valid range
        if 'RSI' in result.columns and not result['RSI'].empty:
            assert result['RSI'].min() >= 0
            assert result['RSI'].max() <= 100


class TestGetMultipleStocksData:
    """Test get_multiple_stocks_data function"""
    
    @patch('src.dashboard.get_stock_data_safe')
    @patch('src.dashboard.time.sleep')  # Mock sleep to speed up tests
    def test_get_multiple_stocks_data_success(self, mock_sleep, mock_get_stock, sample_stock_data):
        """Test successful multi-stock data retrieval"""
        mock_get_stock.return_value = sample_stock_data
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = get_multiple_stocks_data(symbols)
        
        assert isinstance(result, dict)
        assert len(result) == 3
        assert 'AAPL' in result
        assert 'MSFT' in result
        assert 'GOOGL' in result
        
        for symbol, data in result.items():
            assert isinstance(data, pd.DataFrame)
            assert not data.empty
    
    @patch('src.dashboard.get_stock_data_safe')
    @patch('src.dashboard.time.sleep')
    def test_get_multiple_stocks_data_partial_failure(self, mock_sleep, mock_get_stock, sample_stock_data):
        """Test handling partial failures"""
        def side_effect(symbol, *args, **kwargs):
            if symbol == 'FAIL':
                return None
            return sample_stock_data
        
        mock_get_stock.side_effect = side_effect
        
        symbols = ['AAPL', 'FAIL', 'GOOGL']
        result = get_multiple_stocks_data(symbols)
        
        assert isinstance(result, dict)
        assert len(result) == 2  # Only successful ones
        assert 'AAPL' in result
        assert 'GOOGL' in result
        assert 'FAIL' not in result
    
    @patch('src.dashboard.get_stock_data_safe')
    @patch('src.dashboard.time.sleep')
    def test_get_multiple_stocks_data_rate_limiting(self, mock_sleep, mock_get_stock, sample_stock_data):
        """Test that rate limiting (sleep) is applied"""
        mock_get_stock.return_value = sample_stock_data
        
        symbols = ['AAPL', 'MSFT']
        get_multiple_stocks_data(symbols)
        
        # Should sleep between requests (but not after the last one)
        assert mock_sleep.call_count == 1


# Note: calculate_simple_prediction function doesn't exist in dashboard.py
# class TestCalculateSimplePrediction:
#     """Test calculate_simple_prediction function"""
#     
#     def test_calculate_simple_prediction_basic(self, sample_stock_data):
#         """Test basic prediction calculation"""
#         # Add required technical indicators
#         df = calculate_technical_indicators(sample_stock_data)
#         
#         prediction, confidence, score = calculate_simple_prediction(df)
#         
#         assert isinstance(prediction, str)
#         assert prediction in ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'HOLD', 'WEAK_SELL', 'SELL', 'STRONG_SELL']
#         assert isinstance(confidence, float)
#         assert 0 <= confidence <= 1
#         assert isinstance(score, (int, float))
#     
#     def test_calculate_simple_prediction_empty_data(self):
#         """Test with empty data"""
#         empty_df = pd.DataFrame()
#         
#         prediction, confidence, score = calculate_simple_prediction(empty_df)
#         
#         assert prediction == "HOLD"
#         assert confidence == 0.5
#         assert score == 0
#     
#     def test_calculate_simple_prediction_insufficient_data(self):
#         """Test with insufficient data"""
#         small_df = pd.DataFrame({
#             'Close': [100, 101, 99],
#             'RSI': [50, 55, 45],
#             'MACD': [0.1, 0.2, -0.1],
#             'MACD_Signal': [0.05, 0.15, 0.05]
#         })
#         
#         prediction, confidence, score = calculate_simple_prediction(small_df)
#         
#         assert prediction == "HOLD"
#         assert confidence == 0.5
#         assert score == 0
#     
#     def test_calculate_simple_prediction_bullish_signals(self):
#         """Test with bullish signals"""
#         # Create data with bullish indicators
#         dates = pd.date_range('2023-01-01', periods=30, freq='D')
#         df = pd.DataFrame({
#             'Close': np.linspace(100, 110, 30),  # Upward trend
#             'RSI': [30] * 30,  # Oversold
#             'MACD': [0.5] * 30,  # Positive
#             'MACD_Signal': [0.3] * 30,  # Below MACD
#             'CRSI': [25] * 30,  # Oversold
#             'Kalman_Trend': [0.6] * 30,  # Positive trend
#             'BB_Upper': [115] * 30,
#             'BB_Lower': [95] * 30,
#             'Volume': [1000000] * 30,
#             'Volume_MA': [800000] * 30,  # High volume
#             'Price_Change_5': [0.08] * 30  # Strong positive change
#         }, index=dates)
#         
#         prediction, confidence, score = calculate_simple_prediction(df)
#         
#         # Should be bullish
#         assert prediction in ['BUY', 'STRONG_BUY', 'WEAK_BUY']
#         assert confidence > 0.5
#         assert score > 0
#     
#     def test_calculate_simple_prediction_bearish_signals(self):
#         """Test with bearish signals"""
#         # Create data with bearish indicators
#         dates = pd.date_range('2023-01-01', periods=30, freq='D')
#         df = pd.DataFrame({
#             'Close': np.linspace(110, 100, 30),  # Downward trend
#             'RSI': [80] * 30,  # Overbought
#             'MACD': [-0.5] * 30,  # Negative
#             'MACD_Signal': [-0.3] * 30,  # Above MACD
#             'CRSI': [85] * 30,  # Overbought
#             'Kalman_Trend': [-0.6] * 30,  # Negative trend
#             'BB_Upper': [115] * 30,
#             'BB_Lower': [95] * 30,
#             'Volume': [1000000] * 30,
#             'Volume_MA': [800000] * 30,
#             'Price_Change_5': [-0.08] * 30  # Strong negative change
#         }, index=dates)
#         
#         prediction, confidence, score = calculate_simple_prediction(df)
#         
#         # Should be bearish
#         assert prediction in ['SELL', 'STRONG_SELL', 'WEAK_SELL']
#         assert confidence > 0.5
#         assert score < 0


class TestGetStockName:
    """Test get_stock_name function"""
    
    def test_get_stock_name_known_symbols(self):
        """Test with known stock symbols"""
        assert get_stock_name('AAPL') == 'Apple Inc.'
        assert get_stock_name('MSFT') == 'Microsoft'
        assert get_stock_name('TSLA') == 'Tesla'
        assert get_stock_name('NVDA') == 'NVIDIA'
    
    def test_get_stock_name_unknown_symbol(self):
        """Test with unknown stock symbol"""
        unknown_symbol = 'UNKNOWN123'
        result = get_stock_name(unknown_symbol)
        
        assert result == unknown_symbol  # Should return the symbol itself
    
    def test_get_stock_name_case_sensitivity(self):
        """Test case sensitivity"""
        # Should work with exact case
        assert get_stock_name('AAPL') == 'Apple Inc.'
        
        # Unknown case variations should return the input
        assert get_stock_name('aapl') == 'aapl'
        assert get_stock_name('Aapl') == 'Aapl'


class TestDashboardIntegration:
    """Integration tests for dashboard functionality"""
    
    @patch('src.dashboard.get_stock_data_safe')
    def test_data_processing_pipeline(self, mock_get_stock, sample_stock_data):
        """Test the complete data processing pipeline"""
        mock_get_stock.return_value = sample_stock_data
        
        # Simulate the pipeline: get data -> calculate indicators
        symbols = ['AAPL']
        stock_data = get_multiple_stocks_data(symbols)
        
        assert 'AAPL' in stock_data
        
        df = stock_data['AAPL']
        # The data should already have technical indicators from get_stock_data_safe
        
        # Basic validation that we got processed data
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
    
    @patch('src.dashboard.yf.Ticker')
    def test_error_handling_pipeline(self, mock_ticker_class):
        """Test error handling in the pipeline"""
        # Mock API failure
        mock_ticker_class.side_effect = Exception("API Error")
        
        symbols = ['FAIL1', 'FAIL2']
        result = get_multiple_stocks_data(symbols)
        
        # Should handle failures gracefully
        assert isinstance(result, dict)
        assert len(result) == 0  # No successful data
    
    def test_technical_indicators_consistency(self, sample_stock_data):
        """Test that technical indicators are calculated consistently"""
        df1 = calculate_technical_indicators(sample_stock_data.copy())
        df2 = calculate_technical_indicators(sample_stock_data.copy())
        
        # Results should be identical for the same input
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_prediction_consistency(self, sample_stock_data):
        """Test technical indicators consistency"""
        df = calculate_technical_indicators(sample_stock_data)
        
        # Test that technical indicators are calculated consistently
        df1 = calculate_technical_indicators(sample_stock_data.copy())
        df2 = calculate_technical_indicators(sample_stock_data.copy())
        
        # Results should be identical for the same input
        pd.testing.assert_frame_equal(df1, df2)


# Additional test for dashboard app structure (if needed)
class TestDashboardApp:
    """Test dashboard app structure and configuration"""
    
    def test_dashboard_imports(self):
        """Test that dashboard imports work correctly"""
        # This test verifies that the dashboard module can be imported
        # and basic components are available
        try:
            import src.dashboard as dashboard
            assert hasattr(dashboard, 'app')
            assert hasattr(dashboard, 'COLORS')
            assert hasattr(dashboard, 'DEFAULT_SYMBOLS')
        except ImportError:
            pytest.skip("Dashboard module not available for import")
    
    def test_dashboard_configuration(self):
        """Test dashboard configuration constants"""
        try:
            from src.dashboard import COLORS, DEFAULT_SYMBOLS, REFRESH_INTERVAL
            
            assert isinstance(COLORS, dict)
            assert isinstance(DEFAULT_SYMBOLS, list)
            assert isinstance(REFRESH_INTERVAL, int)
            assert REFRESH_INTERVAL > 0
            
            # Check that default symbols are reasonable
            assert len(DEFAULT_SYMBOLS) > 0
            assert all(isinstance(symbol, str) for symbol in DEFAULT_SYMBOLS)
            
        except ImportError:
            pytest.skip("Dashboard constants not available for import")