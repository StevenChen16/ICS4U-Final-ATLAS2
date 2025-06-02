"""
Tests for src/auto_tuning.py module
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import os
import logging
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from src.auto_tuning import (
    DataFingerprint,
    ATLASConfig,
    SimpleDataAnalyzer,
    SimpleRuleConfigurator,
    ATLASAutoTuner,
    get_auto_config,
    load_config
)


class TestDataFingerprint:
    """Test DataFingerprint dataclass"""
    
    def test_data_fingerprint_creation(self):
        """Test DataFingerprint creation"""
        fingerprint = DataFingerprint(
            n_tickers=5,
            n_samples=1000,
            n_features=20,
            avg_volatility=0.25,
            trend_strength=0.6,
            noise_level=1.8,
            label_balance=0.52,
            avg_price_change=0.015,
            missing_ratio=0.01,
            data_length_days=1000
        )
        
        assert fingerprint.n_tickers == 5
        assert fingerprint.avg_volatility == 0.25
        assert fingerprint.label_balance == 0.52
    
    def test_data_fingerprint_to_dict(self):
        """Test converting DataFingerprint to dict"""
        fingerprint = DataFingerprint(
            n_tickers=3,
            n_samples=500,
            n_features=15,
            avg_volatility=0.3,
            trend_strength=0.5,
            noise_level=2.0,
            label_balance=0.48,
            avg_price_change=0.02,
            missing_ratio=0.05,
            data_length_days=500
        )
        
        fp_dict = asdict(fingerprint)
        assert isinstance(fp_dict, dict)
        assert fp_dict['n_tickers'] == 3
        assert fp_dict['avg_volatility'] == 0.3


class TestATLASConfig:
    """Test ATLASConfig dataclass"""
    
    def test_atlas_config_defaults(self):
        """Test ATLASConfig default values"""
        config = ATLASConfig()
        
        assert config.window_size == 50
        assert config.stride == 10
        assert config.threshold == 0.5
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.use_specialized_kernels == True
    
    def test_atlas_config_custom_values(self):
        """Test ATLASConfig with custom values"""
        config = ATLASConfig(
            window_size=40,
            batch_size=64,
            learning_rate=0.002,
            epochs=100
        )
        
        assert config.window_size == 40
        assert config.batch_size == 64
        assert config.learning_rate == 0.002
        assert config.epochs == 100
        # Default values should still be there
        assert config.stride == 10
        assert config.threshold == 0.5
    
    def test_atlas_config_to_dict(self):
        """Test converting ATLASConfig to dict"""
        config = ATLASConfig(window_size=60, epochs=75)
        config_dict = asdict(config)
        
        assert isinstance(config_dict, dict)
        assert config_dict['window_size'] == 60
        assert config_dict['epochs'] == 75


class TestSimpleDataAnalyzer:
    """Test SimpleDataAnalyzer class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.analyzer = SimpleDataAnalyzer()
    
    @patch('src.auto_tuning.load_data_from_csv')
    def test_analyze_dataset_success(self, mock_load_csv, sample_stock_data):
        """Test successful dataset analysis"""
        mock_load_csv.return_value = sample_stock_data
        
        ticker_list = ['AAPL', 'MSFT', 'GOOGL']
        fingerprint = self.analyzer.analyze_dataset(ticker_list, 'test_dir')
        
        assert isinstance(fingerprint, DataFingerprint)
        assert fingerprint.n_tickers == 3
        assert fingerprint.n_samples > 0
        assert fingerprint.n_features > 0
        assert 0 <= fingerprint.avg_volatility <= 5  # Reasonable range
        assert 0 <= fingerprint.trend_strength <= 1
        assert fingerprint.noise_level > 0
        assert 0 <= fingerprint.label_balance <= 1
        assert fingerprint.missing_ratio >= 0
    
    @patch('src.auto_tuning.load_data_from_csv')
    def test_analyze_dataset_with_errors(self, mock_load_csv, sample_stock_data):
        """Test dataset analysis with some failed loads"""
        def side_effect(path):
            if 'FAIL' in path:
                raise FileNotFoundError("File not found")
            return sample_stock_data
        
        mock_load_csv.side_effect = side_effect
        
        ticker_list = ['AAPL', 'FAIL', 'GOOGL']
        fingerprint = self.analyzer.analyze_dataset(ticker_list, 'test_dir')
        
        # Should succeed with 2 tickers instead of 3
        assert fingerprint.n_tickers == 2
    
    @patch('src.auto_tuning.load_data_from_csv')
    def test_analyze_dataset_no_valid_data(self, mock_load_csv):
        """Test analysis when no valid data is found"""
        mock_load_csv.side_effect = FileNotFoundError("No files found")
        
        ticker_list = ['INVALID1', 'INVALID2']
        
        with pytest.raises(ValueError, match="No valid data files found"):
            self.analyzer.analyze_dataset(ticker_list, 'test_dir')
    
    @patch('src.auto_tuning.load_data_from_csv')
    def test_analyze_dataset_small_data(self, mock_load_csv):
        """Test analysis with very small datasets"""
        # Create minimal data
        small_data = pd.DataFrame({
            'Close': [100, 101, 99, 102],
            'Volume': [1000, 1100, 900, 1050]
        })
        mock_load_csv.return_value = small_data
        
        ticker_list = ['SMALL']
        # Since small data gets filtered out (< 50 rows), expect ValueError
        with pytest.raises(ValueError, match="No valid data files found"):
            self.analyzer.analyze_dataset(ticker_list, 'test_dir')
    
    def test_log_fingerprint(self, caplog):
        """Test fingerprint logging"""
        fingerprint = DataFingerprint(
            n_tickers=3,
            n_samples=1000,
            n_features=20,
            avg_volatility=0.25,
            trend_strength=0.6,
            noise_level=1.8,
            label_balance=0.52,
            avg_price_change=0.015,
            missing_ratio=0.01,
            data_length_days=1000
        )
        
        with caplog.at_level(logging.INFO):
            self.analyzer._log_fingerprint(fingerprint)
        
        # Check log records
        log_messages = [record.message for record in caplog.records]
        assert any("Data Fingerprint" in msg for msg in log_messages)
        assert any("Tickers: 3" in msg for msg in log_messages)
        assert any("Avg Volatility: 0.250" in msg for msg in log_messages)


class TestSimpleRuleConfigurator:
    """Test SimpleRuleConfigurator class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.configurator = SimpleRuleConfigurator()
    
    def test_configure_basic(self):
        """Test basic configuration generation"""
        fingerprint = DataFingerprint(
            n_tickers=5,
            n_samples=1000,
            n_features=20,
            avg_volatility=0.25,
            trend_strength=0.6,
            noise_level=1.8,
            label_balance=0.52,
            avg_price_change=0.015,
            missing_ratio=0.01,
            data_length_days=1000
        )
        
        config = self.configurator.configure(fingerprint)
        
        assert isinstance(config, ATLASConfig)
        assert config.window_size > 0
        assert config.stride > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.epochs > 0
    
    def test_configure_small_data(self):
        """Test configuration for small datasets"""
        fingerprint = DataFingerprint(
            n_tickers=2,
            n_samples=300,  # Small dataset
            n_features=15,
            avg_volatility=0.2,
            trend_strength=0.4,
            noise_level=1.5,
            label_balance=0.5,
            avg_price_change=0.01,
            missing_ratio=0.02,
            data_length_days=300
        )
        
        config = self.configurator.configure(fingerprint)
        
        # Should adjust for small data
        assert config.window_size == 30  # Smaller window for limited data
        assert config.stride == 5       # Smaller stride for limited data
        assert config.batch_size == 16  # Smaller batch size
        assert config.epochs == 80      # More epochs for small data
    
    def test_configure_high_volatility(self):
        """Test configuration for high volatility data"""
        fingerprint = DataFingerprint(
            n_tickers=5,
            n_samples=2000,
            n_features=25,
            avg_volatility=0.5,  # High volatility
            trend_strength=0.3,
            noise_level=3.5,     # High noise
            label_balance=0.45,
            avg_price_change=0.025,
            missing_ratio=0.01,
            data_length_days=2000
        )
        
        config = self.configurator.configure(fingerprint)
        
        # Should adjust for high volatility
        assert config.window_size == 60     # Larger window for high volatility
        assert config.threshold == 0.02     # Higher threshold for volatile data
        assert config.learning_rate == 0.0005  # Lower LR for noisy data
        assert config.dropout_rate == 0.6   # More regularization for noisy data
    
    def test_configure_large_dataset(self):
        """Test configuration for large datasets"""
        fingerprint = DataFingerprint(
            n_tickers=10,
            n_samples=10000,  # Large dataset
            n_features=30,
            avg_volatility=0.15,  # Low volatility
            trend_strength=0.8,
            noise_level=0.8,      # Low noise
            label_balance=0.51,
            avg_price_change=0.008,
            missing_ratio=0.001,
            data_length_days=10000
        )
        
        config = self.configurator.configure(fingerprint)
        
        # Should adjust for large clean data
        assert config.stride == 15          # Larger stride for abundant data
        assert config.batch_size == 64      # Larger batch size
        assert config.learning_rate == 0.002  # Higher LR for clean data
        assert config.epochs == 40          # Fewer epochs for large data
        assert config.dropout_rate == 0.3   # Less regularization
    
    def test_configure_patience_calculation(self):
        """Test patience calculation logic"""
        fingerprint = DataFingerprint(
            n_tickers=3,
            n_samples=1000,
            n_features=20,
            avg_volatility=0.25,
            trend_strength=0.6,
            noise_level=1.8,
            label_balance=0.52,
            avg_price_change=0.015,
            missing_ratio=0.01,
            data_length_days=1000
        )
        
        config = self.configurator.configure(fingerprint)
        
        # Patience should be at least 10 or epochs//5
        expected_patience = max(10, config.epochs // 5)
        assert config.patience == expected_patience
    
    def test_log_configuration_choices(self, caplog):
        """Test configuration logging"""
        fingerprint = DataFingerprint(
            n_tickers=3,
            n_samples=1000,
            n_features=20,
            avg_volatility=0.25,
            trend_strength=0.6,
            noise_level=1.8,
            label_balance=0.52,
            avg_price_change=0.015,
            missing_ratio=0.01,
            data_length_days=1000
        )
        
        with caplog.at_level(logging.INFO):
            config = self.configurator.configure(fingerprint)
        
        # Check log records
        log_messages = [record.message for record in caplog.records]
        assert any("Configuration Choices" in msg for msg in log_messages)
        assert any("Window Size:" in msg for msg in log_messages)
        assert any("Batch Size:" in msg for msg in log_messages)


class TestATLASAutoTuner:
    """Test ATLASAutoTuner class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.tuner = ATLASAutoTuner(config_save_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.auto_tuning.SimpleDataAnalyzer.analyze_dataset')
    @patch('src.auto_tuning.SimpleRuleConfigurator.configure')
    def test_auto_tune_success(self, mock_configure, mock_analyze):
        """Test successful auto-tuning"""
        # Mock dependencies
        mock_fingerprint = DataFingerprint(
            n_tickers=3,
            n_samples=1000,
            n_features=20,
            avg_volatility=0.25,
            trend_strength=0.6,
            noise_level=1.8,
            label_balance=0.52,
            avg_price_change=0.015,
            missing_ratio=0.01,
            data_length_days=1000
        )
        mock_config = ATLASConfig(window_size=45, epochs=60)
        
        mock_analyze.return_value = mock_fingerprint
        mock_configure.return_value = mock_config
        
        ticker_list = ['AAPL', 'MSFT', 'GOOGL']
        result_config = self.tuner.auto_tune(ticker_list, 'test_dir')
        
        assert isinstance(result_config, ATLASConfig)
        assert result_config.window_size == 45
        assert result_config.epochs == 60
        
        # Check that methods were called
        mock_analyze.assert_called_once_with(ticker_list, 'test_dir')
        mock_configure.assert_called_once_with(mock_fingerprint)
    
    @patch('src.auto_tuning.SimpleDataAnalyzer.analyze_dataset')
    @patch('src.auto_tuning.SimpleRuleConfigurator.configure')
    def test_auto_tune_with_save(self, mock_configure, mock_analyze):
        """Test auto-tuning with config saving"""
        mock_fingerprint = DataFingerprint(
            n_tickers=2,
            n_samples=500,
            n_features=15,
            avg_volatility=0.3,
            trend_strength=0.5,
            noise_level=2.0,
            label_balance=0.48,
            avg_price_change=0.02,
            missing_ratio=0.05,
            data_length_days=500
        )
        mock_config = ATLASConfig(window_size=35)
        
        mock_analyze.return_value = mock_fingerprint
        mock_configure.return_value = mock_config
        
        ticker_list = ['AAPL', 'TSLA']
        self.tuner.auto_tune(ticker_list, 'test_dir', save_config=True)
        
        # Check that config file was created
        config_files = [f for f in os.listdir(self.temp_dir) if f.startswith('atlas_auto_config_')]
        assert len(config_files) == 1
        
        # Verify file content
        config_file = os.path.join(self.temp_dir, config_files[0])
        with open(config_file, 'r') as f:
            saved_data = json.load(f)
        
        assert 'fingerprint' in saved_data
        assert 'configuration' in saved_data
        assert 'ticker_list' in saved_data
        assert saved_data['ticker_list'] == ticker_list
    
    @patch('src.auto_tuning.SimpleDataAnalyzer.analyze_dataset')
    @patch('src.auto_tuning.SimpleRuleConfigurator.configure')
    def test_auto_tune_no_save(self, mock_configure, mock_analyze):
        """Test auto-tuning without saving"""
        mock_fingerprint = DataFingerprint(
            n_tickers=1,
            n_samples=200,
            n_features=10,
            avg_volatility=0.4,
            trend_strength=0.3,
            noise_level=2.5,
            label_balance=0.6,
            avg_price_change=0.03,
            missing_ratio=0.1,
            data_length_days=200
        )
        mock_config = ATLASConfig()
        
        mock_analyze.return_value = mock_fingerprint
        mock_configure.return_value = mock_config
        
        ticker_list = ['NVDA']
        self.tuner.auto_tune(ticker_list, 'test_dir', save_config=False)
        
        # No config files should be created
        config_files = [f for f in os.listdir(self.temp_dir) if f.startswith('atlas_auto_config_')]
        assert len(config_files) == 0


class TestLoadConfig:
    """Test load_config function"""
    
    def test_load_config_success(self):
        """Test successful config loading"""
        # Create temporary config file
        fingerprint = DataFingerprint(
            n_tickers=3,
            n_samples=1000,
            n_features=20,
            avg_volatility=0.25,
            trend_strength=0.6,
            noise_level=1.8,
            label_balance=0.52,
            avg_price_change=0.015,
            missing_ratio=0.01,
            data_length_days=1000
        )
        config = ATLASConfig(window_size=55, epochs=70)
        
        config_data = {
            'fingerprint': asdict(fingerprint),
            'configuration': asdict(config),
            'ticker_list': ['AAPL', 'MSFT'],
            'version': '1.0',
            'timestamp': '2024-01-01T12:00:00'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name
        
        try:
            loaded_fp, loaded_config = load_config(temp_file)
            
            assert isinstance(loaded_fp, DataFingerprint)
            assert isinstance(loaded_config, ATLASConfig)
            assert loaded_fp.n_tickers == 3
            assert loaded_config.window_size == 55
            assert loaded_config.epochs == 70
        finally:
            os.unlink(temp_file)
    
    def test_load_config_file_not_found(self):
        """Test loading nonexistent config file"""
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.json')


class TestGetAutoConfig:
    """Test get_auto_config function"""
    
    @patch('src.auto_tuning.ATLASAutoTuner.auto_tune')
    def test_get_auto_config_success(self, mock_auto_tune):
        """Test successful auto config generation"""
        mock_config = ATLASConfig(
            window_size=45,
            stride=8,
            threshold=0.8,
            batch_size=48,
            learning_rate=0.0015,
            epochs=65
        )
        mock_auto_tune.return_value = mock_config
        
        ticker_list = ['AAPL', 'GOOGL']
        result = get_auto_config(ticker_list, 'test_dir')
        
        assert isinstance(result, dict)
        assert result['window_size'] == 45
        assert result['stride'] == 8
        assert result['threshold'] == 0.8
        assert result['batch_size'] == 48
        assert result['learning_rate'] == 0.0015
        assert result['epochs'] == 65
        
        # Check all expected keys are present
        expected_keys = [
            'window_size', 'stride', 'threshold', 'image_size',
            'batch_size', 'learning_rate', 'epochs', 'patience',
            'dropout_rate', 'validation_size', 'gap_size', 'use_specialized_kernels'
        ]
        for key in expected_keys:
            assert key in result
    
    @patch('src.auto_tuning.ATLASAutoTuner.auto_tune')
    def test_get_auto_config_default_values(self, mock_auto_tune):
        """Test with default ATLASConfig values"""
        mock_auto_tune.return_value = ATLASConfig()  # Default values
        
        ticker_list = ['TSLA']
        result = get_auto_config(ticker_list)
        
        # Should return default values
        assert result['window_size'] == 50
        assert result['stride'] == 10
        assert result['batch_size'] == 32
        assert result['learning_rate'] == 0.001
        assert result['use_specialized_kernels'] == True


class TestIntegration:
    """Integration tests for auto_tuning module"""
    
    @patch('src.auto_tuning.load_data_from_csv')
    def test_full_auto_tuning_pipeline(self, mock_load_csv, sample_stock_data):
        """Test the complete auto-tuning pipeline"""
        mock_load_csv.return_value = sample_stock_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            tuner = ATLASAutoTuner(config_save_dir=temp_dir)
            
            ticker_list = ['AAPL', 'MSFT']
            config = tuner.auto_tune(ticker_list, 'test_dir', save_config=True)
            
            # Check that config is reasonable
            assert isinstance(config, ATLASConfig)
            assert config.window_size > 0
            assert config.epochs > 0
            assert 0 < config.learning_rate < 1
            
            # Check that file was saved
            config_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            assert len(config_files) == 1
            
            # Test loading the saved config
            config_file = os.path.join(temp_dir, config_files[0])
            loaded_fp, loaded_config = load_config(config_file)
            
            assert loaded_config.window_size == config.window_size
            assert loaded_config.epochs == config.epochs
    
    @patch('src.auto_tuning.load_data_from_csv')
    def test_get_auto_config_integration(self, mock_load_csv, sample_stock_data):
        """Test get_auto_config integration"""
        mock_load_csv.return_value = sample_stock_data
        
        ticker_list = ['AAPL', 'NVDA', 'TSLA']
        result = get_auto_config(ticker_list, 'test_dir')
        
        # Should return a complete config dict
        assert isinstance(result, dict)
        assert len(result) >= 12  # Should have all config parameters
        
        # All values should be reasonable
        assert result['window_size'] > 0
        assert result['stride'] > 0
        assert result['batch_size'] > 0
        assert 0 < result['learning_rate'] < 1
        assert result['epochs'] > 0
        assert 0 <= result['dropout_rate'] <= 1
        assert isinstance(result['use_specialized_kernels'], bool)