# ----------------------------------------
# ATLAS: Advanced Technical Learning Analysis System
# ----------------------------------------
# Binary Classification Version - Only distinguishes up and down
# ----------------------------------------

import os
import warnings

import joblib
# matplotlib font configuration
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from pyts.image import (GramianAngularField, MarkovTransitionField,
                        RecurrencePlot)
from skimage.transform import resize
from sklearn.metrics import (auc, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

USING_NPU = False
try:
    import torch_npu
    print("Successfully imported torch_npu")
    USING_NPU = True
except ImportError:
    pass

# Import user-provided data processing module
from src.data import load_data_from_csv  # User-provided data processing module
from src.auto_tuning import get_auto_config  # Auto-tuning module

mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
elif USING_NPU():
    torch.npu.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if USING_NPU:
    device = torch.device("npu" if torch.npu.is_available() else "cpu")
print(f"Using device: {device}")

# Ignore warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
# Part 1: Data Loading and Preprocessing
# ----------------------------------------


def load_ticker_data(ticker, data_dir="data_short", start_idx=None, end_idx=None):
    """
    Load preprocessed data for a single stock

    Parameters:
    ticker (str): Stock ticker symbol
    data_dir (str): Data directory
    start_idx, end_idx: Optional index range

    Returns:
    pd.DataFrame: Processed stock data
    """
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find data file for {ticker}: {file_path}")

    df = load_data_from_csv(file_path)

    # Select specific range of data (if specified)
    if start_idx is not None and end_idx is not None:
        df = df.iloc[start_idx:end_idx]

    return df


def create_binary_labels(df, window_size=50, threshold=0.5):
    """
    Create binary classification labels: up or down

    Parameters:
    df (pd.DataFrame): DataFrame containing price data
    window_size (int): Window size
    threshold (float): Price change percentage threshold for filtering minor changes

    Returns:
    list: Binary labels ('up' or 'down')
    """
    # Calculate price change percentage for future window_size days
    future_returns = df["Close"].pct_change(window_size).shift(-window_size)

    # Generate binary labels
    labels = []

    for i in range(len(df) - window_size):
        pct_change = future_returns.iloc[i]

        if pct_change > threshold:
            labels.append("up")
        elif pct_change < -threshold:
            labels.append("down")
        else:
            # If change is not large enough, we still classify by direction
            # But could also choose to ignore these samples
            if pct_change >= 0:
                labels.append("up")
            else:
                labels.append("down")

    return labels


def extract_windows_with_stride(df, window_size=50, stride=10, features=None):
    """
    Extract windows with fixed stride to reduce overlap between samples

    Parameters:
    df (pd.DataFrame): Input dataframe
    window_size (int): Window size
    stride (int): Window extraction stride
    features (list): List of features to use, if None, use all columns

    Returns:
    tuple: (window list, label list, index list)
    """
    if features is None:
        # Exclude date index and other non-feature columns
        features = df.columns

    # Create sufficiently long same index for labels
    labels = create_binary_labels(df, window_size)

    windows = []
    window_labels = []
    indices = []

    # Extract windows using stride
    for i in range(0, len(df) - window_size, stride):
        if i + window_size <= len(df) and i < len(labels):
            window = df.iloc[i : i + window_size][features].values
            windows.append(window)
            window_labels.append(labels[i])
            indices.append(i)  # Record window start index

    return np.array(windows), np.array(window_labels), np.array(indices)


def transform_3d_to_images(windows, image_size=50):
    """
    Convert 3D windows (samples, timesteps, features) into 4 types of image representations

    Parameters:
    windows (numpy.ndarray): 3D window array (n_samples, n_timesteps, n_features)
    image_size (int): Output image size

    Returns:
    numpy.ndarray: Transformed image array (n_samples, image_size, image_size, 4)
    """
    n_samples, n_timesteps, n_features = windows.shape

    # Ensure image_size doesn't exceed time series length
    actual_image_size = min(image_size, n_timesteps)
    print(
        f"Actual image size used: {actual_image_size} (original window length: {n_timesteps})"
    )

    # Initialize output array - 4 types of image representations
    transformed_images = np.zeros((n_samples, actual_image_size, actual_image_size, 4))

    # Use the first feature of time series (usually closing price) as main conversion basis
    # Transform each sample
    for i in tqdm(range(n_samples), desc="Converting images"):
        ts = windows[i, :, 0]  # Using the first feature (closing price)
        ts_reshaped = ts.reshape(1, -1)

        # 1. Gramian Angular Summation Field (GASF)
        gasf = GramianAngularField(image_size=actual_image_size, method="summation")
        transformed_images[i, :, :, 0] = gasf.fit_transform(ts_reshaped)[0]

        # 2. Gramian Angular Difference Field (GADF)
        gadf = GramianAngularField(image_size=actual_image_size, method="difference")
        transformed_images[i, :, :, 1] = gadf.fit_transform(ts_reshaped)[0]

        # 3. Recurrence Plot - use combination of all features
        # Create a synthetic feature using weighted combination of multiple features
        feature_weights = np.ones(n_features) / n_features  # simple average
        weighted_ts = np.sum(windows[i] * feature_weights.reshape(1, -1), axis=1)
        weighted_ts_reshaped = weighted_ts.reshape(1, -1)

        rp = RecurrencePlot(threshold="point", percentage=20)
        rp_image = rp.fit_transform(weighted_ts_reshaped)[0]

        if rp_image.shape[0] != actual_image_size:
            rp_image = resize(
                rp_image, (actual_image_size, actual_image_size), anti_aliasing=True
            )
        transformed_images[i, :, :, 2] = rp_image

        # 4. Markov Transition Field
        mtf = MarkovTransitionField(image_size=actual_image_size, n_bins=8)
        transformed_images[i, :, :, 3] = mtf.fit_transform(ts_reshaped)[0]

    return transformed_images


# ----------------------------------------
# Part 2: Time Series Splitting
# ----------------------------------------


def time_series_split(data, window_indices, labels, test_size=0.2, gap_size=20):
    """
    Split data chronologically, ensuring reasonable time range ratio between training and test sets, with gap preserved

    Parameters:
    data (numpy.ndarray): Input data
    window_indices (numpy.ndarray): Starting index of each window
    labels (numpy.ndarray): Labels
    test_size (float): Test set proportion (default 0.2, i.e., 20%)
    gap_size (int): Index gap between training and test sets (default 20)

    Returns:
    tuple: (X_train, X_test, y_train, y_test, train_indices, test_indices)
    """
    # Check input data consistency
    assert (
        len(data) == len(window_indices) == len(labels)
    ), "Input data, indices and labels must have consistent length"

    # Get time index range
    min_idx = min(window_indices)
    max_idx = max(window_indices)
    total_range = max_idx - min_idx
    print(f"Time index range: {min_idx} to {max_idx} (total range: {total_range})")

    # Calculate split point by time range
    split_time = min_idx + int((1 - test_size) * total_range)
    print(f"Calculated split point (split_time): {split_time}")

    # Consider gap_size to ensure gap between training and test sets
    train_end = split_time - gap_size
    test_start = split_time

    # Split data
    train_mask = window_indices < train_end
    test_mask = window_indices >= test_start

    X_train = data[train_mask]
    X_test = data[test_mask]
    y_train = labels[train_mask]
    y_test = labels[test_mask]
    train_indices = window_indices[train_mask]
    test_indices = window_indices[test_mask]

    # Print split results
    print(
        f"Training set size: {len(X_train)} samples, Test set size: {len(X_test)} samples"
    )
    print(
        f"Training set time range: {min(train_indices) if len(train_indices) > 0 else 'N/A'} to "
        f"{max(train_indices) if len(train_indices) > 0 else 'N/A'}"
    )
    print(
        f"Test set time range: {min(test_indices) if len(test_indices) > 0 else 'N/A'} to "
        f"{max(test_indices) if len(test_indices) > 0 else 'N/A'}"
    )
    print(
        f"Train/test gap size: {min(test_indices) - max(train_indices) if len(train_indices) > 0 and len(test_indices) > 0 else 'N/A'} index positions"
    )

    # Verify split ratio
    total_samples = len(data)
    train_ratio = len(X_train) / total_samples if total_samples > 0 else 0
    test_ratio = len(X_test) / total_samples if total_samples > 0 else 0
    print(f"\nSplit ratio check:")
    print(f"Training ratio: {train_ratio:.2%} (expected: {(1 - test_size):.2%})")
    print(f"Test ratio: {test_ratio:.2%} (expected: {test_size:.2%})")

    # Print class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    print("\nTraining set class distribution:")
    for label, count in zip(unique_train, counts_train):
        print(f"  {label}: {count} samples ({count/len(y_train)*100:.1f}%)")

    print("\nTest set class distribution:")
    for label, count in zip(unique_test, counts_test):
        print(f"  {label}: {count} samples ({count/len(y_test)*100:.1f}%)")

    # Warning: if split ratio deviation is too large
    expected_train_ratio = 1 - test_size
    if abs(train_ratio - expected_train_ratio) > 0.05:  # allow 5% deviation
        print(
            "\nWarning: Training set ratio deviates significantly from expected, may need to check data distribution or adjust split logic!"
        )

    return X_train, X_test, y_train, y_test, train_indices, test_indices


# ----------------------------------------
# Part 3: Specialized Financial Convolution Kernels
# ----------------------------------------


def create_specialized_kernels():
    """
    Create specialized financial convolution kernels optimized for financial chart pattern recognition

    Returns:
    dict: Specialized convolution kernels categorized by image type
    """
    # 1. Trend convolution kernels - enhanced detection capability
    uptrend_kernel = np.array(
        [
            [-2.0, -1.5, -1.0, 0.0, 2.0],
            [-1.5, -1.0, 0.0, 1.5, 3.0],
            [-1.0, 0.0, 2.0, 3.5, 4.0],
            [0.0, 1.5, 3.5, 4.5, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )

    downtrend_kernel = np.array(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0],
            [5.0, 4.5, 3.5, 1.5, 0.0],
            [4.0, 3.5, 2.0, 0.0, -1.0],
            [3.0, 1.5, 0.0, -1.0, -1.5],
            [2.0, 0.0, -1.0, -1.5, -2.0],
        ]
    )

    # 2. Support/resistance level detection kernel - enhanced weight contrast
    level_kernel = np.array(
        [
            [-1.5, -1.5, -1.5, -1.5, -1.5],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.5, -1.5, -1.5, -1.5, -1.5],
        ]
    )

    # 3. Reversal pattern convolution kernels - more precise feature extraction
    head_shoulders_top_kernel = np.array(
        [
            [0.8, 1.0, 0.2, 1.0, 0.8],
            [0.5, 0.7, -0.2, 0.7, 0.5],
            [0.0, 0.0, -1.5, 0.0, 0.0],
            [-0.7, -0.7, -1.0, -0.7, -0.7],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
        ]
    )

    head_shoulders_bottom_kernel = np.array(
        [
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [-0.7, -0.7, -1.0, -0.7, -0.7],
            [0.0, 0.0, -1.5, 0.0, 0.0],
            [0.5, 0.7, -0.2, 0.7, 0.5],
            [0.8, 1.0, 0.2, 1.0, 0.8],
        ]
    )

    double_top_kernel = np.array(
        [
            [0.0, 1.0, -0.2, 1.0, 0.0],
            [0.0, 0.7, -0.5, 0.7, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [-0.5, -0.5, -0.7, -0.5, -0.5],
            [-0.8, -0.8, -0.8, -0.8, -0.8],
        ]
    )

    double_bottom_kernel = np.array(
        [
            [-0.8, -0.8, -0.8, -0.8, -0.8],
            [-0.5, -0.5, -0.7, -0.5, -0.5],
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.7, -0.5, 0.7, 0.0],
            [0.0, 1.0, -0.2, 1.0, 0.0],
        ]
    )

    # 4. Enhanced uptrend pattern convolution kernels
    v_bottom_kernel = np.array(
        [
            [-1.5, -1.5, -1.5, -1.5, -1.5],
            [-1.0, -1.2, -1.4, -1.2, -1.0],
            [-0.2, -0.5, -1.0, -0.5, -0.2],
            [0.7, 0.5, 0.0, 0.5, 0.7],
            [1.5, 1.0, 0.5, 1.0, 1.5],
        ]
    )

    # Emphasize price bounce
    bounce_kernel = np.array(
        [
            [-2.0, -2.0, -2.0, -2.0, -2.0],
            [-1.5, -1.5, -1.5, -1.5, -1.5],
            [-0.5, -0.5, -0.5, -0.5, -0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ]
    )

    # 5. Trend change detection kernel - enhanced turning point detection
    trend_change_kernel = np.array(
        [
            [2.5, 1.5, 0.0, -1.5, -2.5],
            [1.5, 2.5, 1.0, -0.5, -1.5],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [-1.5, -0.5, 1.0, 2.5, 1.5],
            [-2.5, -1.5, 0.0, 1.5, 2.5],
        ]
    )

    # 6. Continuation pattern kernels - enhanced uptrend patterns
    symmetric_triangle_kernel = np.array(
        [
            [1.2, 0.6, 0.0, -0.6, -1.2],
            [0.6, 0.6, 0.0, -0.6, -0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.6, -0.6, 0.0, 0.6, 0.6],
            [-1.2, -0.6, 0.0, 0.6, 1.2],
        ]
    )

    # Enhanced ascending triangle detection capability
    ascending_triangle_kernel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.5, 0.7, 0.9, 1.1],
            [-0.5, -0.4, -0.3, -0.2, -0.1],
            [-1.2, -1.0, -0.8, -0.6, -0.4],
        ]
    )

    # 7. Upward breakout signals
    breakout_up_kernel = np.array(
        [
            [-1.5, -1.5, -1.5, -1.5, -1.5],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [-0.5, -0.5, -0.5, -0.5, -0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ]
    )

    # 8. Special pattern detection - cup and handle pattern (bullish signal)
    cup_handle_kernel = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, -0.5],
            [0.3, 0.0, -0.3, -0.3, -0.2],
            [0.7, 0.0, -0.7, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.5, 1.0],
            [1.2, 1.0, 0.7, 1.0, 1.2],
        ]
    )

    # Accumulation breakout pattern (bullish)
    accumulation_breakout_kernel = np.array(
        [
            [-0.5, -0.5, -0.5, -0.5, -0.5],
            [-0.2, -0.2, -0.2, -0.2, -0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.7, 0.9, 1.1, 1.3],
            [1.0, 1.2, 1.4, 1.6, 1.8],
        ]
    )

    # 9. Recurrence plot specific - stronger repetitive pattern detection
    diagonal_kernel = np.array(
        [
            [1.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.2],
        ]
    )

    # Periodicity detection kernel - enhanced periodicity recognition
    periodicity_kernel = np.array(
        [
            [1.2, 0.0, -1.2, 0.0, 1.2],
            [0.0, -1.2, 0.0, -1.2, 0.0],
            [-1.2, 0.0, 1.2, 0.0, -1.2],
            [0.0, -1.2, 0.0, -1.2, 0.0],
            [1.2, 0.0, -1.2, 0.0, 1.2],
        ]
    )

    # 10. MTF specific - enhanced market state transition recognition
    jump_kernel = np.array(
        [
            [2.5, 1.2, 0.0, -1.2, -2.5],
            [1.2, 0.0, -1.2, -2.5, -1.2],
            [0.0, -1.2, -2.5, -1.2, 0.0],
            [-1.2, -2.5, -1.2, 0.0, 1.2],
            [-2.5, -1.2, 0.0, 1.2, 2.5],
        ]
    )

    # Price breakout detection kernel - enhanced upward breakout signals
    breakout_kernel = np.array(
        [
            [-1.0, -1.0, 3.0, -1.0, -1.0],
            [-1.0, -1.0, 3.0, -1.0, -1.0],
            [-1.0, -1.0, 3.0, -1.0, -1.0],
            [-1.0, -1.0, 3.0, -1.0, -1.0],
            [-1.0, -1.0, 3.0, -1.0, -1.0],
        ]
    )

    # Group convolution kernels by image representation type, focusing on enhancing uptrend pattern detection
    return {
        # GASF is best for capturing overall patterns and trends
        "gasf": [
            uptrend_kernel,  # uptrend
            downtrend_kernel,  # downtrend
            level_kernel,  # support/resistance levels
            # cup and handle pattern (bullish signal)
            cup_handle_kernel,
            # accumulation breakout (bullish signal)
            accumulation_breakout_kernel,
            head_shoulders_top_kernel,  # head and shoulders top
            head_shoulders_bottom_kernel,  # head and shoulders bottom
            double_top_kernel,  # double top
            double_bottom_kernel,  # double bottom
        ],
        # GADF is best for capturing directional changes and turning points
        "gadf": [
            trend_change_kernel,  # trend change
            v_bottom_kernel,  # V-shaped bottom
            bounce_kernel,  # price bounce pattern
            breakout_up_kernel,  # upward breakout signal
        ],
        # RP is best for capturing repetitive structures and periodicity
        "rp": [
            diagonal_kernel,  # diagonal (repetitive patterns)
            periodicity_kernel,  # periodicity patterns
            symmetric_triangle_kernel,  # symmetric triangle
        ],
        # MTF is best for capturing state transitions and market structure changes
        "mtf": [
            jump_kernel,  # price jump
            breakout_kernel,  # price breakout
            ascending_triangle_kernel,  # ascending triangle (bullish signal)
        ],
    }


# ----------------------------------------
# Part 4: PyTorch Model Architecture
# ----------------------------------------


class AttentionBlock(nn.Module):
    """Attention mechanism module for highlighting important features"""

    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        out = x * attention_weights
        return out


class SingleBranchCNN(nn.Module):
    """Single-branch CNN module for processing one image type"""

    def __init__(self, in_channels=1, kernel_weights=None):
        super(SingleBranchCNN, self).__init__()

        if kernel_weights is not None:
            # Use specialized convolution kernels
            num_kernels = len(kernel_weights)
            # Create convolution layer and initialize weights
            self.conv1 = nn.Conv2d(in_channels, num_kernels, kernel_size=5, padding=2)
            # Initialize weights
            kernel_tensor = torch.FloatTensor(np.array(kernel_weights)).unsqueeze(1)
            self.conv1.weight.data = kernel_tensor
            self.conv1.bias.data.fill_(0.0)
        else:
            # Use standard convolution
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)

        # Shared components
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.attention = AttentionBlock(32)  # Add attention mechanism

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.attention(x)  # Apply attention
        x = self.pool(x)
        return x


class ATLASModel(nn.Module):
    """ATLAS模型: 多分支CNN用于金融二元分类"""

    def __init__(self, input_shape=(50, 50, 4), kernels=None, dropout_rate=0.5):
        super(ATLASModel, self).__init__()

        height, width, channels = input_shape
        assert channels == 4, "Input should have 4 channels"

        # 分支
        self.use_specialized_kernels = kernels is not None

        # 各通道分支
        if self.use_specialized_kernels:
            self.gasf_branch = SingleBranchCNN(
                in_channels=1, kernel_weights=kernels["gasf"]
            )
            self.gadf_branch = SingleBranchCNN(
                in_channels=1, kernel_weights=kernels["gadf"]
            )
            self.rp_branch = SingleBranchCNN(
                in_channels=1, kernel_weights=kernels["rp"]
            )
            self.mtf_branch = SingleBranchCNN(
                in_channels=1, kernel_weights=kernels["mtf"]
            )
        else:
            self.gasf_branch = SingleBranchCNN(in_channels=1)
            self.gadf_branch = SingleBranchCNN(in_channels=1)
            self.rp_branch = SingleBranchCNN(in_channels=1)
            self.mtf_branch = SingleBranchCNN(in_channels=1)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 计算融合特征维度
        fusion_input_size = 32 * 4  # 4个分支，每个32个特征

        # 分类器 - 二元分类只需要1个输出单元
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, 1),  # 二元分类
            nn.Sigmoid(),  # Sigmoid激活用于二元分类
        )

    def forward(self, x):
        # 分离每个通道
        # 输入形状: [batch_size, channels, height, width]
        batch_size = x.size(0)

        gasf_input = x[:, 0:1, :, :]
        gadf_input = x[:, 1:2, :, :]
        rp_input = x[:, 2:3, :, :]
        mtf_input = x[:, 3:4, :, :]

        # 处理每个分支
        gasf_features = self.gasf_branch(gasf_input)
        gadf_features = self.gadf_branch(gadf_input)
        rp_features = self.rp_branch(rp_input)
        mtf_features = self.mtf_branch(mtf_input)

        # 全局池化每个分支的特征
        gasf_features = self.global_pool(gasf_features).view(batch_size, -1)
        gadf_features = self.global_pool(gadf_features).view(batch_size, -1)
        rp_features = self.global_pool(rp_features).view(batch_size, -1)
        mtf_features = self.global_pool(mtf_features).view(batch_size, -1)

        # 特征融合
        combined = torch.cat(
            [gasf_features, gadf_features, rp_features, mtf_features], dim=1
        )

        # 分类
        output = self.classifier(combined)
        return output


# ----------------------------------------
# 第5部分: 训练和评估函数
# ----------------------------------------


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=30,
    device=device,
    patience=10,
    best_model_save_path="models/atlas_binary_model_best.pth",
    last_model_save_path="models/atlas_binary_model_last.pth",
):
    """
    训练二元分类模型

    参数:
    model (nn.Module): PyTorch模型
    train_loader (DataLoader): 训练数据加载器
    val_loader (DataLoader): 验证数据加载器
    criterion (nn.Module): 损失函数
    optimizer (optim.Optimizer): 优化器
    scheduler: 学习率调度器
    num_epochs (int): 训练轮数
    device (torch.device): 使用的设备
    patience (int): 早停耐心值
    best_model_save_path (str): 最佳模型保存路径
    last_model_save_path (str): 最后一个模型保存路径

    返回:
    dict: 训练历史记录
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Dropout
    model.dropout = nn.Dropout(0.5)

    # 最佳模型保存
    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted.view(-1) == labels).sum().item()

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
            ):
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().view(-1, 1))

                # 统计
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted.view(-1) == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        # 学习率调度
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 输出进度
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_save_path)
            print(f"保存最佳模型, 验证损失: {val_loss:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # 保存最后一个模型
        torch.save(model.state_dict(), last_model_save_path)

        # 早停
        if no_improve_epochs >= patience:
            print(f"早停: {patience}轮未改善验证损失")
            break

    return history


def evaluate_binary_model(model, test_loader, criterion, device=device):
    """
    评估二元分类模型并生成详细指标

    参数:
    model (nn.Module): PyTorch模型
    test_loader (DataLoader): 测试数据加载器
    criterion (nn.Module): 损失函数
    device (torch.device): 使用的设备

    返回:
    tuple: (测试损失, 测试准确率, 预测概率, 真实标签)
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估模型"):
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))

            # 统计
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted.view(-1) == labels).sum().item()

            # 收集预测概率和标签
            all_probs.extend(outputs.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / test_total
    test_acc = test_correct / test_total

    # 打印评估结果
    print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

    # 将概率和标签转换为numpy数组
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 计算混淆矩阵
    predictions = (all_probs > 0.5).astype(int)
    cm = confusion_matrix(all_labels, predictions)

    # 计算精确率、召回率等
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # 打印详细指标
    print("\n二元分类指标:")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"特异度 (Specificity): {specificity:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["下跌", "上涨"],
        yticklabels=["下跌", "上涨"],
    )
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.title("混淆矩阵")
    plt.savefig("results/binary_confusion_matrix.png")
    plt.show()

    # 绘制ROC曲线和PR曲线
    plot_binary_metrics(all_labels, all_probs)

    return test_loss, test_acc, all_probs, all_labels


def plot_binary_metrics(true_labels, predicted_probs):
    """
    绘制二元分类模型的ROC曲线和PR曲线

    参数:
    true_labels (numpy.ndarray): 真实标签
    predicted_probs (numpy.ndarray): 预测概率
    """
    # ROC曲线
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC曲线 (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("假正率 (False Positive Rate)")
    plt.ylabel("真正率 (True Positive Rate)")
    plt.title("ROC曲线")
    plt.legend(loc="lower right")

    # PR曲线
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs)
    avg_precision = average_precision_score(true_labels, predicted_probs)

    plt.subplot(1, 2, 2)
    plt.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"PR曲线 (AP = {avg_precision:.2f})",
    )
    plt.axhline(
        y=sum(true_labels) / len(true_labels),
        color="red",
        linestyle="--",
        label=f"基准 (正例比例 = {sum(true_labels)/len(true_labels):.2f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("召回率 (Recall)")
    plt.ylabel("精确率 (Precision)")
    plt.title("精确率-召回率曲线")
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("results/binary_metrics.png")
    plt.show()


# ----------------------------------------
# 第6部分: 可视化和解释工具
# ----------------------------------------


def visualize_sample(
    time_series,
    transformed_image,
    prediction=None,
    true_label=None,
    prob=None,
    all_features=None,
    index=None,
    save_path=None,
):
    """
    可视化时间序列样本及其图像转换

    参数:
    time_series (numpy.ndarray): 原始时间序列数据
    transformed_image (numpy.ndarray): 转换后的图像
    prediction (str, optional): 模型预测
    true_label (str, optional): 真实标签
    prob (float, optional): 预测概率
    all_features (numpy.ndarray, optional): 所有特征
    index (int, optional): 样本索引
    save_path (str, optional): 保存路径
    """
    # 创建大型图表
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])

    # 时间序列 - 只显示收盘价
    ax0 = plt.subplot(gs[0, :])
    ax0.plot(time_series, label="Close Price")
    ax0.set_title("Price Time Series", fontsize=15)
    ax0.legend()
    ax0.grid(True)

    # 如果是二元分类，添加颜色条以指示预测
    if prediction is not None and true_label is not None:
        if prediction == "up":
            color = "green" if prediction == true_label else "darkred"
        else:  # prediction == 'down'
            color = "red" if prediction == true_label else "darkgreen"

        ax0.axhspan(min(time_series), max(time_series), alpha=0.2, color=color)

    # 添加多特征可视化（如果提供）
    if all_features is not None:
        n_timesteps, n_features = all_features.shape
        if n_features > 1:
            ax1 = plt.subplot(gs[1, 0])
            # 显示技术指标
            for i in range(1, min(5, n_features)):  # 最多显示4个其他特征
                ax1.plot(all_features[:, i], label=f"Feature {i}")
            ax1.set_title("Technical Indicators", fontsize=12)
            ax1.legend(loc="upper right")
            ax1.grid(True)

            # 图像表示从第二个位置开始
            start_col = 1
        else:
            start_col = 0
    else:
        start_col = 0

    # 四种图像表示
    titles = ["GASF", "GADF", "Recurrence Plot", "MTF"]
    cmaps = ["viridis", "plasma", "binary", "hot"]

    for i in range(4):
        row = 1 + (i // (3 - start_col))
        col = (i % (3 - start_col)) + start_col
        ax = plt.subplot(gs[row, col])
        im = ax.imshow(transformed_image[:, :, i], cmap=cmaps[i])
        ax.set_title(titles[i], fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 添加预测和真实标签信息
    if prediction is not None or true_label is not None or index is not None:
        info_text = ""
        if index is not None:
            info_text += f"样本索引: {index}\n"
        if true_label is not None:
            info_text += f"真实标签: {true_label}\n"
        if prediction is not None:
            prob_str = f" (概率: {prob:.2f})" if prob is not None else ""
            info_text += f"预测结果: {prediction}{prob_str}"

        fig.text(
            0.5,
            0.01,
            info_text,
            ha="center",
            fontsize=12,
            bbox=dict(facecolor="yellow", alpha=0.2),
        )

    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def visualize_training_history(history):
    """
    可视化训练历史

    参数:
    history (dict): 训练历史字典
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 准确率
    ax1.plot(history["train_acc"], label="Train")
    ax1.plot(history["val_acc"], label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(loc="lower right")
    ax1.grid(True)

    # 损失
    ax2.plot(history["train_loss"], label="Train")
    ax2.plot(history["val_loss"], label="Validation")
    ax2.set_title("Model Loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("results/binary_training_history.png", bbox_inches="tight")
    plt.show()


def plot_predictions_over_time(predictions, true_labels, probs, indices):
    """
    绘制预测随时间变化的图表

    参数:
    predictions (numpy.ndarray): 预测类别 (0/1)
    true_labels (numpy.ndarray): 真实类别 (0/1)
    probs (numpy.ndarray): 预测概率
    indices (numpy.ndarray): 时间索引
    """
    plt.figure(figsize=(15, 8))

    # 将二元标签转换为1和-1，用于绘图
    plot_true = np.where(true_labels == 1, 1, -1)

    # 绘制预测概率
    plt.subplot(2, 1, 1)
    plt.plot(indices, probs, "b-", alpha=0.7, label="预测概率")
    plt.axhline(y=0.5, color="gray", linestyle="--", label="决策阈值")

    # 标记错误预测
    incorrect = predictions != true_labels
    plt.scatter(
        indices[incorrect],
        probs[incorrect],
        color="red",
        marker="x",
        s=50,
        label="错误预测",
    )

    plt.title("预测概率随时间变化")
    plt.ylabel("上涨概率")
    plt.legend()
    plt.grid(True)

    # 绘制真实标签和预测
    plt.subplot(2, 1, 2)
    plt.scatter(
        indices, plot_true, color="blue", marker="o", alpha=0.7, label="真实标签"
    )

    # 使用不同颜色标记正确和错误预测
    correct = ~incorrect
    plt.scatter(
        indices[correct],
        plot_true[correct],
        color="green",
        marker="o",
        s=100,
        alpha=0.5,
        label="正确预测",
    )
    plt.scatter(
        indices[incorrect],
        plot_true[incorrect],
        color="red",
        marker="x",
        s=100,
        label="错误预测",
    )

    plt.title("真实标签和预测随时间变化")
    plt.ylabel("标签 (1=上涨, -1=下跌)")
    plt.yticks([-1, 1], ["下跌", "上涨"])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/predictions_over_time.png", bbox_inches="tight")
    plt.show()


# ----------------------------------------
# 第7部分: 执行流程
# ----------------------------------------


def run_atlas_binary_pipeline(
    ticker_list=["AAPL", "MSFT", "GOOGL"],
    data_dir="data_short",
    window_size=50,
    stride=10,
    image_size=50,
    use_specialized_kernels=True,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    patience=10,
    validation_size=0.2,
    gap_size=10,
    threshold=0.5,
    enable_auto_tuning=False,
):
    """
    运行ATLAS二元分类流程

    参数:
    ticker_list (list): 要处理的股票代码列表
    data_dir (str): 数据目录
    window_size (int): 时间窗口大小
    stride (int): 窗口步长
    image_size (int): 生成图像大小
    use_specialized_kernels (bool): 是否使用专业卷积核
    epochs (int): 训练轮数
    batch_size (int): 批次大小
    learning_rate (float): 学习率
    patience (int): 早停耐心值
    validation_size (float): 验证集比例
    gap_size (int): 训练和测试集间隔
    threshold (float): 价格变动阈值，用于标签生成
    enable_auto_tuning (bool): 是否启用自动调参（类似nnUNet）

    返回:
    tuple: (训练好的模型, 测试数据)
    """
    # 确保输出目录存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # === Auto-Tuning Integration ===
    if enable_auto_tuning:
        print("\n🚀 ATLAS Auto-Tuning Enabled (nnUNet-like)")
        print("=" * 50)
        
        try:
            # Get auto-tuned configuration
            auto_config = get_auto_config(ticker_list, data_dir)
            
            # Override parameters with auto-tuned values
            print("📊 Auto-tuned parameters:")
            for key, value in auto_config.items():
                if key in locals():
                    old_value = locals()[key]
                    if old_value != value:
                        print(f"  {key}: {old_value} → {value}")
                    else:
                        print(f"  {key}: {value} (unchanged)")
                    locals()[key] = value
            
            # Update variables in current scope
            window_size = auto_config['window_size']
            stride = auto_config['stride']
            threshold = auto_config['threshold']
            image_size = auto_config['image_size']
            batch_size = auto_config['batch_size']
            learning_rate = auto_config['learning_rate']
            epochs = auto_config['epochs']
            patience = auto_config['patience']
            validation_size = auto_config['validation_size']
            gap_size = auto_config['gap_size']
            use_specialized_kernels = auto_config['use_specialized_kernels']
            
            print("✅ Auto-tuning completed!")
            print("=" * 50)
            
        except Exception as e:
            print(f"⚠️ Auto-tuning failed: {str(e)}")
            print("📋 Using manual parameters...")
    else:
        print("📋 Using manual parameters (auto-tuning disabled)")

    # 步骤1: 加载和处理数据
    all_windows = []
    all_labels = []
    all_indices = []

    selected_features = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "MA5",
        "MA20",
        "MACD",
        "RSI",
        "Upper",
        "Lower",
        "CRSI",
        "Kalman_Trend",
        "FFT_21",
    ]

    for ticker in ticker_list:
        print(f"\n处理股票: {ticker}")
        try:
            # 加载预处理的数据
            stock_data = load_ticker_data(ticker, data_dir=data_dir)

            # 提取窗口和标签
            windows, labels, indices = extract_windows_with_stride(
                stock_data,
                window_size=window_size,
                stride=stride,
                features=selected_features,
            )

            all_windows.extend(windows)
            all_labels.extend(labels)
            all_indices.extend(indices)

            print(f"从 {ticker} 提取的窗口数: {len(windows)}")

        except Exception as e:
            print(f"处理 {ticker} 时出错: {str(e)}")

    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)

    print(f"\n总窗口数: {len(all_windows)}")
    print(f"窗口形状: {all_windows.shape}")

    # 创建标签编码器
    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(all_labels)
    class_names = label_encoder.classes_
    print(f"类别映射: {dict(zip(class_names, range(len(class_names))))}")

    # 步骤2: 将时间序列转换为图像表示
    print("\n将时间序列转换为图像...")
    transformed_images = transform_3d_to_images(all_windows, image_size=image_size)

    # 步骤3: 按时间顺序分割训练集和测试集
    print("\n分割训练集和测试集...")
    X_train, X_test, y_train, y_test, train_indices, test_indices = time_series_split(
        transformed_images,
        all_indices,
        numeric_labels,
        test_size=0.2,
        gap_size=gap_size,
    )

    # 进一步将训练集分为训练和验证
    train_size = int((1 - validation_size) * len(X_train))
    val_size = len(X_train) - train_size

    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]

    print(f"最终训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")

    # 步骤4: 准备PyTorch数据集和加载器
    # PyTorch需要调整通道维度顺序 (B, H, W, C) -> (B, C, H, W)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_val = np.transpose(X_val, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # 创建数据集和加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 步骤5: 创建和训练模型
    kernels = create_specialized_kernels() if use_specialized_kernels else None

    # 获取输入形状
    input_shape = (
        transformed_images.shape[1],
        transformed_images.shape[2],
        transformed_images.shape[3],
    )
    print(f"模型输入形状: {input_shape}")

    model = ATLASModel(input_shape=input_shape, kernels=kernels, dropout_rate=0.5).to(
        device
    )

    # 打印模型结构
    print("\n模型结构:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")

    # 计算类别权重以处理不平衡问题
    counts = np.bincount(y_train)
    class_weight = 1.0 / counts
    class_weight = class_weight / np.sum(class_weight)

    print(f"类别权重: {class_weight}")

    # 对于二元分类，使用BCE损失
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # 训练模型
    print("\n开始训练模型...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=epochs,
        device=device,
        patience=patience,
    )

    # 可视化训练历史
    visualize_training_history(history)

    # 步骤6: 加载最佳模型并评估
    print("\n加载最佳模型进行评估...")
    model.load_state_dict(torch.load("models/atlas_binary_model_best.pth"))

    test_loss, test_acc, test_probs, test_labels = evaluate_binary_model(
        model, test_loader, criterion, device
    )

    # 将概率转换为预测类别
    test_preds = (test_probs > 0.5).astype(int)

    # 绘制预测随时间变化的图表
    plot_predictions_over_time(test_preds, test_labels, test_probs, test_indices)

    # 步骤7: 可视化一些预测样本
    print("\n可视化一些预测样本...")
    # 选择一些典型样本 (正确和错误的预测)
    correct_mask = test_preds == test_labels
    incorrect_mask = ~correct_mask

    # 取一些正确预测的样本
    n_correct = min(3, np.sum(correct_mask))
    if n_correct > 0:
        correct_indices = np.where(correct_mask)[0][:n_correct]

        for idx_in_test in correct_indices:
            # 转回原始通道顺序 (C, H, W) -> (H, W, C)
            sample_image = X_test[idx_in_test].transpose(1, 2, 0)
            sample_window_idx = test_indices[idx_in_test]

            # 获取原始窗口数据
            original_idx = np.where(all_indices == sample_window_idx)[0][0]
            sample_window = all_windows[original_idx]

            # 获取标签和预测
            pred = "up" if test_preds[idx_in_test] == 1 else "down"
            true = "up" if test_labels[idx_in_test] == 1 else "down"
            prob = test_probs[idx_in_test]

            # 可视化
            visualize_sample(
                sample_window[:, 0],  # 收盘价
                sample_image,
                prediction=pred,
                true_label=true,
                prob=prob,
                all_features=sample_window,
                index=sample_window_idx,
                save_path=f"results/binary_correct_prediction_{idx_in_test}.png",
            )

    # 取一些错误预测的样本
    n_incorrect = min(3, np.sum(incorrect_mask))
    if n_incorrect > 0:
        incorrect_indices = np.where(incorrect_mask)[0][:n_incorrect]

        for idx_in_test in incorrect_indices:
            # 转回原始通道顺序
            sample_image = X_test[idx_in_test].transpose(1, 2, 0)
            sample_window_idx = test_indices[idx_in_test]

            # 获取原始窗口数据
            original_idx = np.where(all_indices == sample_window_idx)[0][0]
            sample_window = all_windows[original_idx]

            # 获取标签和预测
            pred = "up" if test_preds[idx_in_test] == 1 else "down"
            true = "up" if test_labels[idx_in_test] == 1 else "down"
            prob = test_probs[idx_in_test]

            # 可视化
            visualize_sample(
                sample_window[:, 0],  # 收盘价
                sample_image,
                prediction=pred,
                true_label=true,
                prob=prob,
                all_features=sample_window,
                index=sample_window_idx,
                save_path=f"results/binary_incorrect_prediction_{idx_in_test}.png",
            )

    # 保存模型及相关信息
    model_info = {
        "class_names": class_names,
        "input_shape": input_shape,
        "window_size": window_size,
        "image_size": transformed_images.shape[1],
        "test_accuracy": test_acc,
        "selected_features": selected_features,
    }
    joblib.dump(model_info, "models/atlas_binary_model_info.pkl")

    return model, (X_test, y_test, test_indices, test_probs)


def demo_auto_tuning():
    """
    Demo function showing auto-tuning vs manual configuration
    """
    print("=" * 60)
    print("ATLAS Auto-Tuning Demo (nnUNet-inspired)")
    print("=" * 60)
    
    # Small test with a few tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    print("\n🔍 Demo 1: Manual Configuration")
    print("-" * 30)
    model1, _ = run_atlas_binary_pipeline(
        ticker_list=test_tickers,
        data_dir="data_short",
        window_size=50,
        stride=10,
        batch_size=32,
        learning_rate=0.001,
        epochs=10,  # Short for demo
        enable_auto_tuning=False,
    )
    
    print("\n🤖 Demo 2: Auto-Tuned Configuration (nnUNet-style)")
    print("-" * 30)
    model2, _ = run_atlas_binary_pipeline(
        ticker_list=test_tickers,
        data_dir="data_short",
        epochs=10,  # Short for demo
        enable_auto_tuning=True,  # 🎆 Magic happens here!
    )
    
    print("\n✅ Demo completed! Check the differences in configuration.")
    print("\n📝 Note: Auto-tuning analyzes your data characteristics and")
    print("automatically adjusts parameters like window_size, batch_size, learning_rate, etc.")
    print("Just like nnUNet does for medical images, but adapted for financial time series!")


def main():
    """
    主函数
    """
    print("=" * 50)
    print("ATLAS: Advanced Technical Learning Analysis System")
    print("股票市场技术二元分类系统")
    print("=" * 50)

    # 股票列表
    ticker_list = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
        "QQQ",
    ]
    ticker_list = [
        "AAPL",
        "ABBV",
        "ABT",
        "ADBE",
        "AMD",
        "AMGN",
        "AMT",
        "AMZN",
        "AXP",
        "BAC",
        "BLK",
        "BMY",
        "C",
        "CAT",
        "CCI",
        "CMCSA",
        "COP",
        "COST",
        "CRM",
        "CSCO",
        "CVS",
        "CVX",
        "DHR",
        "DIS",
        "DOW",
        "EMR",
        "EOG",
        "EQIX",
        "ETN",
        "FCX",
        "FDX",
        "GD",
        "GE",
        "GILD",
        "GOOGL",
        "GS",
        "HAL",
        "HD",
        "HON",
        "HUM",
        "IBM",
        "INTC",
        "ISRG",
        "JNJ",
        "JPM",
        "KMB",
        "KO",
        "LIN",
        "LLY",
        "LMT",
        "LOW",
        "MA",
        "MCD",
        "MET",
        "META",
        "MMM",
        "MPC",
        "MRK",
        "MS",
        "MSFT",
        "MU",
        "NFLX",
        "NKE",
        "NOW",
        "NSC",
        "NVDA",
        "ORCL",
        "PEP",
        "PFE",
        "PG",
        "PLD",
        "PNC",
        "PPG",
        "PSA",
        "QCOM",
        "REGN",
        "ROK",
        "RTX",
        "SBUX",
        "SCHW",
        "SHW",
        "SLB",
        "SPG",
        "T",
        "TGT",
        "TMO",
        "TSLA",
        "UNH",
        "UNP",
        "UPS",
        "USB",
        "V",
        "VLO",
        "VRTX",
        "VZ",
        "WFC",
        "WM",
        "WMT",
        "XOM",
        "ZTS",
    ]
    # ticker_list = ['000001.SS', 'AAPL', 'ABBV', 'ABT', 'ADBE', 'AIG', 'ALB', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMT', 'AMZN', 'APA', 'APD', 'ARE', 'ARKK', 'AVB', 'AXP', 'BA', 'BABA', 'BAC', 'BIDU', 'BIIB', 'BILI', 'BK', 'BKR', 'BLK', 'BMY', 'BXP', 'C', 'CAT', 'CCI', 'CE', 'CF', 'CI', 'CL', 'CMCSA', 'CMI', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DD', 'DE', 'DHR', 'DIA', 'DIS', 'DLR', 'DOW', 'DVN', 'ECL', 'EEM', 'EL', 'EMN', 'EMR', 'EOG', 'EP', 'EQIX', 'EQR', 'ESS', 'ETN', 'FANG', 'FCX', 'FDX', 'FMC', 'GD', 'GDX', 'GE', 'GILD', 'GIS', 'GOOGL', 'GOTU', 'GS', 'HAL', 'HD', 'HES', 'HON', 'HST', 'HUM', 'HYG', 'IBM', 'IEMG', 'IFF', 'INTC', 'IQ', 'IR', 'ISRG', 'IVV', 'IWM', 'JD', 'JNJ', 'JPM', 'K', 'KIM', 'KMB', 'KMI', 'KO', 'LI', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MAA', 'MCD', 'MET', 'META', 'MLM', 'MMM', 'MOS', 'MPC', 'MRK', 'MRO', 'MS', 'MSFT', 'MU', 'NEM', 'NFLX', 'NIO', 'NKE', 'NOW', 'NSC', 'NUE', 'NVDA', 'O', 'ORCL', 'OXY', 'PDD', 'PEP', 'PFE', 'PG', 'PH', 'PLD', 'PNC', 'PPG', 'PRU', 'PSA', 'PSX', 'QCOM', 'QQQ', 'REG', 'REGN', 'ROK', 'RTX', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SLB', 'SNOW', 'SPG', 'SPY', 'T', 'TFC', 'TGT', 'TLT', 'TME', 'TMO', 'TSLA', 'UDR', 'UNH', 'UNP', 'UPS', 'USB', 'USO', 'V', 'VLO', 'VMC', 'VNQ', 'VOO', 'VRTX', 'VTI', 'VTR', 'VZ', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'XLE', 'XLF', 'XLK', 'XLP', 'XLY', 'XOM', 'XPEV', 'ZTS', '^DJI', '^FTSE', '^GDAXI', '^GSPC', '^HSI', '^IXIC', '^NDX', '^RUT', '^VIX']

    # 运行完整流程
    print("\n🎆 Demo: ATLAS with Auto-Tuning (nnUNet-style)")
    print("📝 Set enable_auto_tuning=True to use data-driven parameter optimization")
    
    model, test_data = run_atlas_binary_pipeline(
        ticker_list=ticker_list,
        data_dir="data",  # 数据目录
        window_size=50,  # 时间窗口大小
        stride=10,  # 步长，减少窗口重叠
        image_size=50,  # 图像尺寸
        use_specialized_kernels=True,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        patience=15,  # 早停耐心值
        validation_size=0.2,
        gap_size=20,  # 训练和测试集间隔
        threshold=0.5,  # 标签生成阈值
        enable_auto_tuning=True,  # 🎆 Enable nnUNet-like auto-tuning!
    )

    print("\n完成!")


if __name__ == "__main__":
    # Uncomment the line below to run auto-tuning demo
    # demo_auto_tuning()
    
    # Run full pipeline with auto-tuning enabled
    main()
