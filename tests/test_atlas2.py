"""
Tests for src/atlas2.py module
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.atlas2 import (
    load_ticker_data,
    create_binary_labels,
    extract_windows_with_stride,
    transform_3d_to_images,
    time_series_split,
    create_specialized_kernels,
    AttentionBlock,
    SingleBranchCNN,
    ATLASModel,
)


class TestLoadTickerData:
    """Test load_ticker_data function"""
    
    def test_load_ticker_data_success(self, temp_data_dir):
        """Test successful ticker data loading"""
        result = load_ticker_data('AAPL', data_dir=temp_data_dir)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_load_ticker_data_nonexistent(self, temp_data_dir):
        """Test loading nonexistent ticker"""
        with pytest.raises(FileNotFoundError):
            load_ticker_data('NONEXISTENT', data_dir=temp_data_dir)
    
    def test_load_ticker_data_with_range(self, temp_data_dir):
        """Test loading with index range"""
        result = load_ticker_data('AAPL', data_dir=temp_data_dir, start_idx=10, end_idx=50)
        
        assert len(result) == 40  # 50 - 10
        assert isinstance(result, pd.DataFrame)


class TestCreateBinaryLabels:
    """Test create_binary_labels function"""
    
    def test_create_binary_labels_basic(self, sample_stock_data):
        """Test basic label creation"""
        labels = create_binary_labels(sample_stock_data, window_size=20, threshold=0.01)
        
        assert isinstance(labels, list)
        assert len(labels) == len(sample_stock_data) - 20
        assert all(label in ['up', 'down'] for label in labels)
    
    def test_create_binary_labels_different_thresholds(self, sample_stock_data):
        """Test with different thresholds"""
        labels_low = create_binary_labels(sample_stock_data, window_size=20, threshold=0.005)
        labels_high = create_binary_labels(sample_stock_data, window_size=20, threshold=0.05)
        
        assert len(labels_low) == len(labels_high)
        # Different thresholds should potentially produce different labels
    
    def test_create_binary_labels_window_size(self, sample_stock_data):
        """Test with different window sizes"""
        labels_10 = create_binary_labels(sample_stock_data, window_size=10)
        labels_30 = create_binary_labels(sample_stock_data, window_size=30)
        
        assert len(labels_10) == len(sample_stock_data) - 10
        assert len(labels_30) == len(sample_stock_data) - 30
    
    def test_create_binary_labels_edge_case(self):
        """Test with minimal data"""
        small_data = pd.DataFrame({
            'Close': [100, 101, 99, 102, 98, 103, 97]
        })
        
        labels = create_binary_labels(small_data, window_size=3)
        assert len(labels) == 4  # 7 - 3


class TestExtractWindowsWithStride:
    """Test extract_windows_with_stride function"""
    
    def test_extract_windows_basic(self, processed_stock_data):
        """Test basic window extraction"""
        windows, labels, indices = extract_windows_with_stride(
            processed_stock_data, 
            window_size=20, 
            stride=5
        )
        
        assert isinstance(windows, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert windows.shape[1] == 20  # window_size
        assert windows.shape[2] == len(processed_stock_data.columns)  # features
        assert len(windows) == len(labels) == len(indices)
    
    def test_extract_windows_different_strides(self, processed_stock_data):
        """Test with different stride values"""
        windows_1, labels_1, indices_1 = extract_windows_with_stride(
            processed_stock_data, window_size=20, stride=1
        )
        windows_10, labels_10, indices_10 = extract_windows_with_stride(
            processed_stock_data, window_size=20, stride=10
        )
        
        assert len(windows_1) > len(windows_10)  # Smaller stride = more windows
        assert windows_1.shape[1] == windows_10.shape[1]  # Same window size
    
    def test_extract_windows_feature_selection(self, processed_stock_data):
        """Test with feature selection"""
        selected_features = ['Close', 'Volume', 'MA5']
        windows, labels, indices = extract_windows_with_stride(
            processed_stock_data,
            window_size=20,
            stride=5,
            features=selected_features
        )
        
        assert windows.shape[2] == len(selected_features)
    
    def test_extract_windows_insufficient_data(self):
        """Test with insufficient data"""
        small_data = pd.DataFrame({
            'Close': [100, 101, 99],
            'Volume': [1000, 1100, 900]
        })
        
        windows, labels, indices = extract_windows_with_stride(
            small_data, window_size=10, stride=1
        )
        
        assert len(windows) == 0
        assert len(labels) == 0
        assert len(indices) == 0


class TestTransform3dToImages:
    """Test transform_3d_to_images function"""
    
    def test_transform_3d_to_images_basic(self, sample_windows):
        """Test basic 3D to image transformation"""
        windows, _, _ = sample_windows
        # Use smaller windows for faster testing
        small_windows = windows[:5, :30, :5]  # 5 samples, 30 timesteps, 5 features
        
        images = transform_3d_to_images(small_windows, image_size=25)
        
        assert isinstance(images, np.ndarray)
        assert images.shape[0] == 5  # n_samples
        assert images.shape[1] == 25  # image_size
        assert images.shape[2] == 25  # image_size
        assert images.shape[3] == 4  # 4 image types (GASF, GADF, RP, MTF)
    
    def test_transform_3d_to_images_different_sizes(self, sample_windows):
        """Test with different image sizes"""
        windows, _, _ = sample_windows
        small_windows = windows[:3, :20, :3]
        
        for size in [15, 20]:
            images = transform_3d_to_images(small_windows, image_size=size)
            expected_size = min(size, 20)  # Limited by window length
            assert images.shape[1] == expected_size
            assert images.shape[2] == expected_size
    
    def test_transform_3d_to_images_edge_case(self):
        """Test with very small windows"""
        tiny_windows = np.random.randn(2, 10, 2)
        images = transform_3d_to_images(tiny_windows, image_size=15)
        
        assert images.shape[0] == 2
        assert images.shape[3] == 4


class TestTimeSeriesSplit:
    """Test time_series_split function"""
    
    def test_time_series_split_basic(self, sample_windows):
        """Test basic time series splitting"""
        windows, labels, indices = sample_windows
        label_encoder = {'up': 1, 'down': 0}
        numeric_labels = np.array([label_encoder[label] for label in labels])
        
        X_train, X_test, y_train, y_test, train_idx, test_idx = time_series_split(
            windows, indices, numeric_labels, test_size=0.2, gap_size=5
        )
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) == len(y_train) == len(train_idx)
        assert len(X_test) == len(y_test) == len(test_idx)
        
        # Check temporal ordering
        assert max(train_idx) < min(test_idx) - 5  # gap_size = 5
    
    def test_time_series_split_ratios(self, sample_windows):
        """Test different split ratios"""
        windows, labels, indices = sample_windows
        numeric_labels = np.array([1 if label == 'up' else 0 for label in labels])
        
        X_train, X_test, y_train, y_test, _, _ = time_series_split(
            windows, indices, numeric_labels, test_size=0.3
        )
        
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        
        assert 0.2 <= test_ratio <= 0.4  # Allow some deviation due to temporal constraints
    
    def test_time_series_split_gap_size(self, sample_windows):
        """Test gap size enforcement"""
        windows, labels, indices = sample_windows
        numeric_labels = np.array([1 if label == 'up' else 0 for label in labels])
        
        gap_size = 10
        X_train, X_test, y_train, y_test, train_idx, test_idx = time_series_split(
            windows, indices, numeric_labels, test_size=0.2, gap_size=gap_size
        )
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            actual_gap = min(test_idx) - max(train_idx)
            assert actual_gap >= gap_size


class TestCreateSpecializedKernels:
    """Test create_specialized_kernels function"""
    
    def test_create_specialized_kernels_structure(self):
        """Test kernel structure"""
        kernels = create_specialized_kernels()
        
        assert isinstance(kernels, dict)
        assert 'gasf' in kernels
        assert 'gadf' in kernels
        assert 'rp' in kernels
        assert 'mtf' in kernels
        
        for image_type, kernel_list in kernels.items():
            assert isinstance(kernel_list, list)
            assert len(kernel_list) > 0
            
            for kernel in kernel_list:
                assert isinstance(kernel, np.ndarray)
                assert kernel.shape == (5, 5)  # All kernels should be 5x5
    
    def test_create_specialized_kernels_values(self):
        """Test kernel values are reasonable"""
        kernels = create_specialized_kernels()
        
        for image_type, kernel_list in kernels.items():
            for kernel in kernel_list:
                # Check that kernels have reasonable value ranges
                assert not np.isnan(kernel).any()
                assert not np.isinf(kernel).any()
                assert kernel.min() >= -10.0  # Reasonable lower bound
                assert kernel.max() <= 10.0   # Reasonable upper bound


class TestAttentionBlock:
    """Test AttentionBlock module"""
    
    def test_attention_block_forward(self, device):
        """Test AttentionBlock forward pass"""
        batch_size, channels, height, width = 2, 32, 10, 10
        attention = AttentionBlock(channels).to(device)
        
        x = torch.randn(batch_size, channels, height, width).to(device)
        output = attention(x)
        
        assert output.shape == x.shape
        assert torch.is_tensor(output)
    
    def test_attention_block_different_channels(self, device):
        """Test with different channel numbers"""
        for channels in [16, 32, 64]:
            attention = AttentionBlock(channels).to(device)
            x = torch.randn(1, channels, 8, 8).to(device)
            output = attention(x)
            assert output.shape == (1, channels, 8, 8)


class TestSingleBranchCNN:
    """Test SingleBranchCNN module"""
    
    def test_single_branch_cnn_no_kernels(self, device):
        """Test SingleBranchCNN without specialized kernels"""
        model = SingleBranchCNN(in_channels=1).to(device)
        
        x = torch.randn(2, 1, 32, 32).to(device)
        output = model(x)
        
        assert torch.is_tensor(output)
        assert output.dim() == 4  # (batch, channels, height, width)
    
    def test_single_branch_cnn_with_kernels(self, device):
        """Test SingleBranchCNN with specialized kernels"""
        kernels = create_specialized_kernels()
        gasf_kernels = kernels['gasf'][:3]  # Use first 3 kernels
        
        model = SingleBranchCNN(in_channels=1, kernel_weights=gasf_kernels).to(device)
        
        x = torch.randn(2, 1, 32, 32).to(device)
        output = model(x)
        
        assert torch.is_tensor(output)
        assert output.shape[1] == 32  # Output channels after conv2
    
    def test_single_branch_cnn_different_input_sizes(self, device):
        """Test with different input sizes"""
        model = SingleBranchCNN(in_channels=1).to(device)
        
        for size in [16, 32, 64]:
            x = torch.randn(1, 1, size, size).to(device)
            output = model(x)
            assert output.shape[2] == size // 4  # Two pooling layers
            assert output.shape[3] == size // 4


class TestATLASModel:
    """Test ATLASModel module"""
    
    def test_atlas_model_creation(self, device):
        """Test ATLAS model creation"""
        input_shape = (50, 50, 4)
        model = ATLASModel(input_shape=input_shape).to(device)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'gasf_branch')
        assert hasattr(model, 'gadf_branch')
        assert hasattr(model, 'rp_branch')
        assert hasattr(model, 'mtf_branch')
        assert hasattr(model, 'classifier')
    
    def test_atlas_model_forward(self, device):
        """Test ATLAS model forward pass"""
        input_shape = (32, 32, 4)
        model = ATLASModel(input_shape=input_shape).to(device)
        
        batch_size = 2
        x = torch.randn(batch_size, 4, 32, 32).to(device)  # (B, C, H, W)
        output = model(x)
        
        assert output.shape == (batch_size, 1)  # Binary classification
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_atlas_model_with_kernels(self, device):
        """Test ATLAS model with specialized kernels"""
        kernels = create_specialized_kernels()
        input_shape = (32, 32, 4)
        
        model = ATLASModel(input_shape=input_shape, kernels=kernels).to(device)
        
        x = torch.randn(2, 4, 32, 32).to(device)
        output = model(x)
        
        assert output.shape == (2, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_atlas_model_different_shapes(self, device):
        """Test with different input shapes"""
        for size in [24, 48, 64]:
            input_shape = (size, size, 4)
            model = ATLASModel(input_shape=input_shape).to(device)
            
            # Use batch size > 1 to avoid BatchNorm issues, or set model to eval mode
            model.eval()  # Set to eval mode to avoid BatchNorm issues with batch_size=1
            x = torch.randn(1, 4, size, size).to(device)
            output = model(x)
            
            assert output.shape == (1, 1)
            

class TestIntegration:
    """Integration tests for atlas2 module"""
    
    def test_data_pipeline_integration(self, processed_stock_data):
        """Test the complete data processing pipeline"""
        # Extract windows
        windows, labels, indices = extract_windows_with_stride(
            processed_stock_data, 
            window_size=30, 
            stride=5
        )
        
        # Transform to images (using smaller subset for speed)
        if len(windows) > 0:
            sample_windows = windows[:3]  # Just test first 3
            images = transform_3d_to_images(sample_windows, image_size=25)
            
            # Check pipeline worked
            assert images.shape[0] == 3
            assert images.shape[3] == 4
            assert not np.isnan(images).any()
    
    def test_model_pipeline_integration(self, device):
        """Test model creation and inference pipeline"""
        # Create model
        kernels = create_specialized_kernels()
        model = ATLASModel(input_shape=(32, 32, 4), kernels=kernels).to(device)
        
        # Create dummy data
        batch_size = 2
        x = torch.randn(batch_size, 4, 32, 32).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        # Check output
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        
        # Test training mode
        model.train()
        output_train = model(x)
        assert output_train.shape == (batch_size, 1)
    
    @patch('src.atlas2.load_ticker_data')
    def test_load_and_process_integration(self, mock_load, processed_stock_data):
        """Test loading and processing integration"""
        mock_load.return_value = processed_stock_data
        
        # This simulates the beginning of run_atlas_binary_pipeline
        stock_data = mock_load('AAPL', data_dir='test_dir')
        
        windows, labels, indices = extract_windows_with_stride(
            stock_data, window_size=20, stride=5
        )
        
        assert len(windows) > 0
        assert len(labels) == len(windows)
        assert len(indices) == len(windows)