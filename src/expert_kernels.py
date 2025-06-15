"""
Expert Kernel Bank for ATLAS MoE System

This module implements specialized convolution kernels for different market archetypes.
Each expert has kernels optimized for specific trading patterns and market dynamics.

Author: Steven Chen
Date: 2025-06-14
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod


class BaseExpert(ABC):
    """Base class for all expert kernel banks"""
    
    def __init__(self, expert_name: str):
        self.expert_name = expert_name
        self.kernels = {}
        self._initialize_kernels()
    
    @abstractmethod
    def _initialize_kernels(self):
        """Initialize expert-specific kernels"""
        pass
    
    def get_kernels(self) -> Dict[str, List[np.ndarray]]:
        """Get all kernels for this expert"""
        return self.kernels
    
    def get_kernel_counts(self) -> Dict[str, int]:
        """Get number of kernels for each image type"""
        return {img_type: len(kernels) for img_type, kernels in self.kernels.items()}


class SharedKernelBank:
    """Shared building blocks used across experts"""
    
    @staticmethod
    def sma_detector(size: int = 5) -> np.ndarray:
        """Simple Moving Average detector kernel"""
        kernel = np.ones((size, size)) / (size * size)
        return kernel
    
    @staticmethod
    def ema_detector(size: int = 5, alpha: float = 0.3) -> np.ndarray:
        """Exponential Moving Average detector kernel"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Create exponential decay pattern
        for i in range(size):
            for j in range(size):
                dist = abs(i - center) + abs(j - center)
                kernel[i, j] = alpha * (1 - alpha) ** dist
        
        return kernel / np.sum(kernel)
    
    @staticmethod
    def sobel_edge_detector() -> List[np.ndarray]:
        """Sobel-like edge detection kernels (5x5 version)"""
        # Horizontal edge (5x5)
        sobel_h = np.array([
            [-1, -2, -2, -2, -1],
            [-1, -2, -2, -2, -1],
            [0, 0, 0, 0, 0],
            [1, 2, 2, 2, 1],
            [1, 2, 2, 2, 1]
        ], dtype=np.float32)
        
        # Vertical edge (5x5)
        sobel_v = np.array([
            [-1, -1, 0, 1, 1],
            [-2, -2, 0, 2, 2],
            [-2, -2, 0, 2, 2],
            [-2, -2, 0, 2, 2],
            [-1, -1, 0, 1, 1]
        ], dtype=np.float32)
        
        return [sobel_h, sobel_v]
    
    @staticmethod
    def gap_detector() -> np.ndarray:
        """High-pass differentiator for gap detection"""
        kernel = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, -1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)
        return kernel
    
    @staticmethod
    def gaussian_denoiser(size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Gaussian denoising kernel"""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
        
        return kernel / np.sum(kernel)


class CryptoHFTExpert(BaseExpert):
    """Expert for crypto high-frequency trading patterns"""
    
    def __init__(self):
        super().__init__("crypto_hft")
    
    def _initialize_kernels(self):
        """Initialize crypto HFT specific kernels"""
        shared = SharedKernelBank()
        
        # Start with shared kernels
        self.kernels = {
            'gasf': [
                shared.sma_detector(5),
                shared.ema_detector(5, 0.3),
            ],
            'gadf': [
                shared.sobel_edge_detector()[0],
                shared.sobel_edge_detector()[1],
            ],
            'rp': [
                shared.gap_detector(),
            ],
            'mtf': [
                shared.gaussian_denoiser(5, 1.0),
            ]
        }
        
        # Add crypto-specific kernels
        
        # Multi-scale Difference-of-Gaussian for spike detection
        dog_kernel = self._difference_of_gaussian(sigma1=1.0, sigma2=3.0)
        self.kernels['gasf'].append(dog_kernel)
        
        # Long-tail mean-reversion filter (1x11)
        mean_reversion = self._mean_reversion_filter()
        self.kernels['gasf'].append(mean_reversion)
        
        # High-order temporal gradient
        temporal_gradient = self._temporal_gradient_filter()
        self.kernels['gadf'].append(temporal_gradient)
        
        # Extreme spike detector
        spike_detector = self._spike_detector()
        self.kernels['mtf'].append(spike_detector)
    
    def _difference_of_gaussian(self, sigma1: float = 1.0, sigma2: float = 3.0, size: int = 5) -> np.ndarray:
        """Difference of Gaussian for multi-scale spike detection"""
        g1 = SharedKernelBank.gaussian_denoiser(size, sigma1)
        g2 = SharedKernelBank.gaussian_denoiser(size, sigma2)
        return g1 - g2
    
    def _mean_reversion_filter(self) -> np.ndarray:
        """Long-tail mean reversion filter"""
        # Extended horizontal filter for mean reversion
        kernel = np.zeros((5, 5))
        kernel[2, :] = np.array([-0.2, -0.1, 0.6, -0.1, -0.2])  # Strong center, negative surroundings
        return kernel
    
    def _temporal_gradient_filter(self) -> np.ndarray:
        """High-order temporal gradient (4th order difference)"""
        kernel = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, -4, 6, -4, 1],  # 4th order difference
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32)
        return kernel
    
    def _spike_detector(self) -> np.ndarray:
        """Extreme spike detection kernel"""
        kernel = np.array([
            [-0.1, -0.1, -0.1, -0.1, -0.1],
            [-0.1, -0.2, -0.2, -0.2, -0.1],
            [-0.1, -0.2, 2.0, -0.2, -0.1],
            [-0.1, -0.2, -0.2, -0.2, -0.1],
            [-0.1, -0.1, -0.1, -0.1, -0.1]
        ], dtype=np.float32)
        return kernel


class EquityIntradayExpert(BaseExpert):
    """Expert for equity intraday trading patterns"""
    
    def __init__(self):
        super().__init__("equity_intraday")
    
    def _initialize_kernels(self):
        """Initialize equity intraday specific kernels"""
        shared = SharedKernelBank()
        
        # Start with shared kernels
        self.kernels = {
            'gasf': [
                shared.sma_detector(5),
                shared.ema_detector(5, 0.1),  # Fixed to 5x5
            ],
            'gadf': [
                shared.sobel_edge_detector()[0],
                shared.sobel_edge_detector()[1],
            ],
            'rp': [
                shared.gap_detector(),
            ],
            'mtf': [
                shared.gaussian_denoiser(5, 2.0),
            ]
        }
        
        # Add intraday-specific kernels
        
        # Momentum comb filter
        momentum_comb = self._momentum_comb_filter()
        self.kernels['gadf'].append(momentum_comb)
        
        # Triangular ramp detectors
        ascending_ramp, descending_ramp = self._triangular_ramp_detectors()
        self.kernels['gasf'].extend([ascending_ramp, descending_ramp])
        
        # VWAP drift detector
        vwap_drift = self._vwap_drift_detector()
        self.kernels['mtf'].append(vwap_drift)
    
    def _momentum_comb_filter(self) -> np.ndarray:
        """5-tap momentum comb filter"""
        kernel = np.zeros((5, 5))
        kernel[2, :] = np.array([-1, -2, 0, 2, 1])  # Momentum detection
        return kernel
    
    def _triangular_ramp_detectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ascending and descending triangular ramp detectors"""
        # Ascending ramp
        ascending = np.array([
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6, 0.7, 0.8]
        ], dtype=np.float32)
        
        # Descending ramp
        descending = np.flip(ascending)
        
        return ascending, descending
    
    def _vwap_drift_detector(self) -> np.ndarray:
        """VWAP drift detection kernel"""
        kernel = np.array([
            [-0.2, -0.1, 0.0, 0.1, 0.2],
            [-0.1, -0.05, 0.0, 0.05, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.05, 0.0, -0.05, -0.1],
            [0.2, 0.1, 0.0, -0.1, -0.2]
        ], dtype=np.float32)
        return kernel


class EquityDailyExpert(BaseExpert):
    """Expert for regular daily equity patterns (default)"""
    
    def __init__(self):
        super().__init__("equity_daily")
    
    def _initialize_kernels(self):
        """Initialize equity daily specific kernels (classic technical patterns)"""
        shared = SharedKernelBank()
        
        # Start with shared kernels
        self.kernels = {
            'gasf': [
                shared.sma_detector(5),
                shared.sma_detector(5),  # Fixed to 5x5
                shared.ema_detector(5, 0.4),
            ],
            'gadf': [
                shared.sobel_edge_detector()[0],
                shared.sobel_edge_detector()[1],
            ],
            'rp': [
                shared.gap_detector(),
            ],
            'mtf': [
                shared.gaussian_denoiser(5, 1.0),
            ]
        }
        
        # Add classic technical pattern kernels
        
        # Head and shoulders pattern
        head_shoulders = self._head_shoulders_pattern()
        self.kernels['gasf'].append(head_shoulders)
        
        # Double bottom pattern
        double_bottom = self._double_bottom_pattern()
        self.kernels['gasf'].append(double_bottom)
        
        # Support/resistance levels
        support_resistance = self._support_resistance_detector()
        self.kernels['gasf'].append(support_resistance)
        
        # Triangle pattern
        triangle = self._triangle_pattern()
        self.kernels['rp'].append(triangle)
    
    def _head_shoulders_pattern(self) -> np.ndarray:
        """Head and shoulders pattern detector"""
        kernel = np.array([
            [0.5, 1.0, 0.2, 1.0, 0.5],
            [0.3, 0.7, -0.2, 0.7, 0.3],
            [0.0, 0.0, -1.5, 0.0, 0.0],
            [-0.7, -0.7, -1.0, -0.7, -0.7],
            [-1.0, -1.0, -1.0, -1.0, -1.0]
        ], dtype=np.float32)
        return kernel
    
    def _double_bottom_pattern(self) -> np.ndarray:
        """Double bottom pattern detector"""
        kernel = np.array([
            [-0.8, -0.8, -0.8, -0.8, -0.8],
            [-0.5, -0.5, -0.7, -0.5, -0.5],
            [0.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, 0.7, -0.5, 0.7, 0.0],
            [0.0, 1.0, -0.2, 1.0, 0.0]
        ], dtype=np.float32)
        return kernel
    
    def _support_resistance_detector(self) -> np.ndarray:
        """Support/resistance level detector"""
        kernel = np.array([
            [-1.5, -1.5, -1.5, -1.5, -1.5],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.5, -1.5, -1.5, -1.5, -1.5]
        ], dtype=np.float32)
        return kernel
    
    def _triangle_pattern(self) -> np.ndarray:
        """Triangle pattern detector"""
        kernel = np.array([
            [1.2, 0.6, 0.0, -0.6, -1.2],
            [0.6, 0.6, 0.0, -0.6, -0.6],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.6, -0.6, 0.0, 0.6, 0.6],
            [-1.2, -0.6, 0.0, 0.6, 1.2]
        ], dtype=np.float32)
        return kernel


class FuturesTrendExpert(BaseExpert):
    """Expert for commodity/FX futures trend patterns"""
    
    def __init__(self):
        super().__init__("futures_trend")
    
    def _initialize_kernels(self):
        """Initialize futures trend specific kernels"""
        shared = SharedKernelBank()
        
        # Start with shared kernels
        self.kernels = {
            'gasf': [
                shared.sma_detector(5),  # Fixed to 5x5
                shared.ema_detector(5, 0.05),  # Fixed to 5x5
            ],
            'gadf': [
                shared.sobel_edge_detector()[0],
            ],
            'rp': [
                shared.gap_detector(),
            ],
            'mtf': [
                shared.gaussian_denoiser(5, 2.0),  # Fixed to 5x5
            ]
        }
        
        # Add trend-specific kernels
        
        # Long-range trend detector
        long_trend = self._long_range_trend_detector()
        self.kernels['gasf'].append(long_trend)
        
        # Diagonal trend kernels
        up_diagonal, down_diagonal = self._diagonal_trend_detectors()
        self.kernels['gasf'].extend([up_diagonal, down_diagonal])
        
        # Channel boundary detector
        channel_boundary = self._channel_boundary_detector()
        self.kernels['rp'].append(channel_boundary)
        
        # Momentum persistence
        momentum_persistence = self._momentum_persistence_detector()
        self.kernels['mtf'].append(momentum_persistence)
    
    def _long_range_trend_detector(self) -> np.ndarray:
        """Long-range trend detection kernel"""
        # 7x1 horizontal trend detector
        kernel = np.zeros((5, 5))
        kernel[2, :] = np.array([-2, -1, 0, 1, 2])  # Linear trend
        return kernel
    
    def _diagonal_trend_detectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonal trend detection kernels"""
        # Upward diagonal
        up_diagonal = np.array([
            [2.0, 1.0, 0.0, -1.0, -2.0],
            [1.0, 1.0, 0.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0, 1.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0]
        ], dtype=np.float32)
        
        # Downward diagonal
        down_diagonal = np.array([
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-1.0, -1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, -1.0, -1.0],
            [2.0, 1.0, 0.0, -1.0, -2.0]
        ], dtype=np.float32)
        
        return up_diagonal, down_diagonal
    
    def _channel_boundary_detector(self) -> np.ndarray:
        """Channel boundary detection kernel"""
        kernel = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, -2.0, -2.0, -2.0, -2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ], dtype=np.float32)
        return kernel
    
    def _momentum_persistence_detector(self) -> np.ndarray:
        """Momentum persistence detection"""
        kernel = np.array([
            [0.0, 0.5, 1.0, 1.5, 2.0],
            [0.0, 0.5, 1.0, 1.5, 2.0],
            [0.0, 0.5, 1.0, 1.5, 2.0],
            [0.0, 0.5, 1.0, 1.5, 2.0],
            [0.0, 0.5, 1.0, 1.5, 2.0]
        ], dtype=np.float32)
        return kernel


class LowVolETFExpert(BaseExpert):
    """Expert for low-volatility ETF patterns"""
    
    def __init__(self):
        super().__init__("low_vol_etf")
    
    def _initialize_kernels(self):
        """Initialize low-volatility ETF specific kernels"""
        shared = SharedKernelBank()
        
        # Start with shared kernels (more smoothing)
        self.kernels = {
            'gasf': [
                shared.sma_detector(5),  # Fixed to 5x5
                shared.ema_detector(5, 0.02),  # Fixed to 5x5
            ],
            'gadf': [
                shared.gaussian_denoiser(5, 3.0),  # Fixed to 5x5
            ],
            'rp': [
                shared.gaussian_denoiser(5, 4.0),  # Fixed to 5x5
            ],
            'mtf': [
                shared.gaussian_denoiser(5, 2.5),  # Fixed to 5x5
            ]
        }
        
        # Add low-volatility specific kernels
        
        # Subtle regime shift detector
        regime_shift = self._subtle_regime_shift_detector()
        self.kernels['gasf'].append(regime_shift)
        
        # Mean crossover detector
        mean_crossover = self._mean_crossover_detector()
        self.kernels['gadf'].append(mean_crossover)
        
        # Gentle trend detector
        gentle_trend = self._gentle_trend_detector()
        self.kernels['mtf'].append(gentle_trend)
    
    def _subtle_regime_shift_detector(self) -> np.ndarray:
        """Detect subtle regime shifts in quiet markets"""
        kernel = np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.1, -0.1, -0.1, -0.1, -0.1],
            [-0.2, -0.2, -0.2, -0.2, -0.2]
        ], dtype=np.float32)
        return kernel
    
    def _mean_crossover_detector(self) -> np.ndarray:
        """Mean crossover residual detector"""
        kernel = np.array([
            [-0.1, -0.1, 0.4, -0.1, -0.1],
            [-0.1, -0.1, 0.4, -0.1, -0.1],
            [-0.1, -0.1, 0.4, -0.1, -0.1],
            [-0.1, -0.1, 0.4, -0.1, -0.1],
            [-0.1, -0.1, 0.4, -0.1, -0.1]
        ], dtype=np.float32)
        return kernel
    
    def _gentle_trend_detector(self) -> np.ndarray:
        """Gentle trend detection for low-volatility assets"""
        kernel = np.array([
            [0.0, 0.05, 0.1, 0.15, 0.2],
            [0.0, 0.05, 0.1, 0.15, 0.2],
            [0.0, 0.05, 0.1, 0.15, 0.2],
            [0.0, 0.05, 0.1, 0.15, 0.2],
            [0.0, 0.05, 0.1, 0.15, 0.2]
        ], dtype=np.float32)
        return kernel


class ExpertKernelManager:
    """Manager for all expert kernel banks"""
    
    def __init__(self):
        self.experts = {
            'crypto_hft': CryptoHFTExpert(),
            'equity_intraday': EquityIntradayExpert(),
            'equity_daily': EquityDailyExpert(),
            'futures_trend': FuturesTrendExpert(),
            'low_vol_etf': LowVolETFExpert()
        }
    
    def get_expert_kernels(self, expert_name: str) -> Dict[str, List[np.ndarray]]:
        """Get kernels for specific expert"""
        if expert_name not in self.experts:
            expert_name = 'equity_daily'  # Default fallback
        return self.experts[expert_name].get_kernels()
    
    def get_all_experts(self) -> List[str]:
        """Get list of all available experts"""
        return list(self.experts.keys())
    
    def get_expert_kernel_counts(self, expert_name: str) -> Dict[str, int]:
        """Get kernel counts for specific expert"""
        if expert_name not in self.experts:
            expert_name = 'equity_daily'
        return self.experts[expert_name].get_kernel_counts()
    
    def create_mixed_kernels(self, expert_weights: Dict[str, float]) -> Dict[str, List[np.ndarray]]:
        """
        Create weighted mixture of kernels from multiple experts
        
        Args:
            expert_weights: Dictionary mapping expert names to weights
            
        Returns:
            Mixed kernel bank
        """
        # Normalize weights
        total_weight = sum(expert_weights.values())
        if total_weight == 0:
            expert_weights = {'equity_daily': 1.0}
            total_weight = 1.0
        
        normalized_weights = {k: v/total_weight for k, v in expert_weights.items()}
        
        # Get all image types
        image_types = ['gasf', 'gadf', 'rp', 'mtf']
        mixed_kernels = {img_type: [] for img_type in image_types}
        
        # For each image type, collect weighted kernels
        for img_type in image_types:
            for expert_name, weight in normalized_weights.items():
                if weight > 0.01:  # Skip very small weights
                    expert_kernels = self.get_expert_kernels(expert_name)[img_type]
                    
                    # Weight the kernels
                    weighted_kernels = [kernel * weight for kernel in expert_kernels]
                    mixed_kernels[img_type].extend(weighted_kernels)
        
        return mixed_kernels


# Example usage and testing
if __name__ == "__main__":
    # Test expert kernel creation
    manager = ExpertKernelManager()
    
    print("Available experts:", manager.get_all_experts())
    
    for expert_name in manager.get_all_experts():
        kernels = manager.get_expert_kernels(expert_name)
        counts = manager.get_expert_kernel_counts(expert_name)
        
        print(f"\n{expert_name.upper()} Expert:")
        print(f"Kernel counts: {counts}")
        print(f"Total kernels: {sum(counts.values())}")
        
        # Show first kernel for each image type
        for img_type, kernel_list in kernels.items():
            if kernel_list:
                print(f"{img_type} first kernel shape: {kernel_list[0].shape}")
    
    # Test mixed kernels
    print("\nTesting mixed kernels:")
    expert_weights = {
        'crypto_hft': 0.3,
        'equity_daily': 0.7
    }
    
    mixed_kernels = manager.create_mixed_kernels(expert_weights)
    mixed_counts = {img_type: len(kernels) for img_type, kernels in mixed_kernels.items()}
    print(f"Mixed kernel counts: {mixed_counts}")