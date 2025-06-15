"""
ATLAS Mixture-of-Experts (MoE) Model

This module implements the MoE version of ATLAS that routes inputs to specialized
experts based on market characteristics.

Author: Steven Chen  
Date: 2025-06-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from .expert_kernels import ExpertKernelManager
from .moe_gating import MoEGatingSystem, GatingConfig, create_moe_gating_system


@dataclass
class AtlasMoEConfig:
    """Configuration for ATLAS MoE model"""
    # Model architecture
    input_shape: Tuple[int, int, int] = (50, 50, 4)  # (height, width, channels)
    dropout_rate: float = 0.5
    
    # MoE configuration
    enable_moe: bool = True
    experts: List[str] = None
    top_k: int = 2
    load_balance_weight: float = 0.01
    use_learnable_gate: bool = True
    
    # Expert sharing
    share_first_layer: bool = True
    share_classifier: bool = True
    
    # Training
    expert_dropout_rate: float = 0.1
    
    def __post_init__(self):
        if self.experts is None:
            self.experts = ['crypto_hft', 'equity_intraday', 'equity_daily', 'futures_trend', 'low_vol_etf']


class AttentionBlock(nn.Module):
    """Attention mechanism module for highlighting important features"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class ExpertBranch(nn.Module):
    """Single expert branch for processing one image type with expert-specific kernels"""
    
    def __init__(self, in_channels: int = 1, expert_kernels: Optional[List[np.ndarray]] = None, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        if expert_kernels is not None:
            # Use specialized convolution kernels
            num_kernels = len(expert_kernels)
            self.conv1 = nn.Conv2d(in_channels, num_kernels, kernel_size=5, padding=2)
            
            # Validate and standardize kernel shapes
            standardized_kernels = []
            for kernel in expert_kernels:
                kernel = np.array(kernel, dtype=np.float32)
                if kernel.shape != (5, 5):
                    # Resize kernel to 5x5 if needed
                    if kernel.shape == (3, 3):
                        # Pad 3x3 to 5x5
                        padded = np.zeros((5, 5), dtype=np.float32)
                        padded[1:4, 1:4] = kernel
                        kernel = padded
                    else:
                        # Create a 5x5 kernel and put the original in the center
                        new_kernel = np.zeros((5, 5), dtype=np.float32)
                        h, w = kernel.shape
                        start_h, start_w = (5 - h) // 2, (5 - w) // 2
                        new_kernel[start_h:start_h+h, start_w:start_w+w] = kernel
                        kernel = new_kernel
                standardized_kernels.append(kernel)
            
            # Initialize weights with standardized kernels
            kernel_tensor = torch.FloatTensor(np.stack(standardized_kernels)).unsqueeze(1)
            self.conv1.weight.data = kernel_tensor
            self.conv1.bias.data.fill_(0.0)
        else:
            # Use standard convolution
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        
        # Shared components (remove BatchNorm to avoid single-batch issues)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 32, kernel_size=3, padding=1)
        self.attention = AttentionBlock(32)
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = self.pool(x)
        
        return x


class ATLASExpert(nn.Module):
    """Complete ATLAS expert with 4 branches for different image types"""
    
    def __init__(self, expert_name: str, input_shape: Tuple[int, int, int], 
                 kernels: Optional[Dict[str, List[np.ndarray]]] = None,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.expert_name = expert_name
        height, width, channels = input_shape
        assert channels == 4, "Input should have 4 channels (GASF, GADF, RP, MTF)"
        
        # Create branches for each image type
        if kernels is not None:
            self.gasf_branch = ExpertBranch(1, kernels.get('gasf', None), dropout_rate)
            self.gadf_branch = ExpertBranch(1, kernels.get('gadf', None), dropout_rate)
            self.rp_branch = ExpertBranch(1, kernels.get('rp', None), dropout_rate)
            self.mtf_branch = ExpertBranch(1, kernels.get('mtf', None), dropout_rate)
        else:
            self.gasf_branch = ExpertBranch(1, None, dropout_rate)
            self.gadf_branch = ExpertBranch(1, None, dropout_rate)
            self.rp_branch = ExpertBranch(1, None, dropout_rate)
            self.mtf_branch = ExpertBranch(1, None, dropout_rate)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature fusion dimension
        self.feature_dim = 32 * 4  # 4 branches, each with 32 features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for expert
        
        Args:
            x: Input tensor [batch_size, 4, height, width]
            
        Returns:
            features: Expert features [batch_size, feature_dim]
        """
        batch_size = x.size(0)
        
        # Separate each channel
        gasf_input = x[:, 0:1, :, :]  # [batch_size, 1, height, width]
        gadf_input = x[:, 1:2, :, :]
        rp_input = x[:, 2:3, :, :]
        mtf_input = x[:, 3:4, :, :]
        
        # Process each branch
        gasf_features = self.gasf_branch(gasf_input)
        gadf_features = self.gadf_branch(gadf_input)
        rp_features = self.rp_branch(rp_input)
        mtf_features = self.mtf_branch(mtf_input)
        
        # Global pooling each branch's features
        gasf_features = self.global_pool(gasf_features).view(batch_size, -1)
        gadf_features = self.global_pool(gadf_features).view(batch_size, -1)
        rp_features = self.global_pool(rp_features).view(batch_size, -1)
        mtf_features = self.global_pool(mtf_features).view(batch_size, -1)
        
        # Feature fusion
        combined = torch.cat([gasf_features, gadf_features, rp_features, mtf_features], dim=1)
        
        return combined


class AtlasMoEModel(nn.Module):
    """ATLAS Mixture-of-Experts Model"""
    
    def __init__(self, config: AtlasMoEConfig):
        super().__init__()
        
        self.config = config
        self.enable_moe = config.enable_moe
        
        # Initialize expert kernel manager
        self.kernel_manager = ExpertKernelManager()
        
        if self.enable_moe:
            # MoE setup
            self.experts = nn.ModuleDict()
            
            # Create experts
            for expert_name in config.experts:
                expert_kernels = self.kernel_manager.get_expert_kernels(expert_name)
                expert = ATLASExpert(
                    expert_name=expert_name,
                    input_shape=config.input_shape,
                    kernels=expert_kernels,
                    dropout_rate=config.expert_dropout_rate
                )
                self.experts[expert_name] = expert
            
            # Gating system
            gating_config = GatingConfig(
                num_experts=len(config.experts),
                top_k=config.top_k,
                load_balance_weight=config.load_balance_weight
            )
            self.gating_system = MoEGatingSystem(gating_config, config.use_learnable_gate)
            
            # Feature dimension (from expert)
            sample_expert = next(iter(self.experts.values()))
            self.feature_dim = sample_expert.feature_dim
            
        else:
            # Single expert fallback (use equity_daily as default)
            default_kernels = self.kernel_manager.get_expert_kernels('equity_daily')
            self.single_expert = ATLASExpert(
                expert_name='equity_daily',
                input_shape=config.input_shape,
                kernels=default_kernels,
                dropout_rate=config.dropout_rate
            )
            self.feature_dim = self.single_expert.feature_dim
        
        # Shared classifier  
        if config.share_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate / 2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            # Each expert has its own classifier (not implemented in this version)
            raise NotImplementedError("Individual expert classifiers not implemented yet")
    
    def forward(self, x: torch.Tensor, price_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MoE model
        
        Args:
            x: Image input [batch_size, 4, height, width]
            price_data: Raw price data for fingerprinting [batch_size, seq_len, price_features]
                       If None, will extract from x (not recommended)
        
        Returns:
            Dictionary containing predictions and routing information
        """
        if not self.enable_moe:
            # Single expert mode
            features = self.single_expert(x)
            output = self.classifier(features)
            
            return {
                'logits': output,
                'predictions': output,
                'expert_outputs': {'equity_daily': features},
                'routing_weights': torch.ones(x.size(0), 1, device=x.device),
                'load_balance_loss': torch.tensor(0.0, device=x.device),
                'gating_type': 'single_expert'
            }
        
        # MoE mode
        batch_size = x.size(0)
        device = x.device
        
        # Get routing information
        if price_data is not None:
            routing_info = self.gating_system(price_data)
        else:
            # Fallback: create dummy price data from image features
            # This is not ideal and should be avoided in practice
            dummy_price_data = torch.randn(batch_size, 50, 5, device=device)
            routing_info = self.gating_system(dummy_price_data)
        
        routing_weights = routing_info['routing_weights']  # [batch_size, num_experts]
        expert_indices = routing_info['expert_indices']    # [batch_size, top_k]
        
        # Process through experts
        expert_outputs = {}
        expert_features = torch.zeros(batch_size, self.feature_dim, device=device)
        
        for i, expert_name in enumerate(self.config.experts):
            expert = self.experts[expert_name]
            
            # Get samples that should use this expert
            expert_mask = routing_weights[:, i] > 0
            
            if expert_mask.any():
                # Process samples assigned to this expert
                expert_input = x[expert_mask]
                expert_feature = expert(expert_input)
                expert_outputs[expert_name] = expert_feature
                
                # Weighted combination
                expert_weight = routing_weights[expert_mask, i:i+1]  # [num_samples, 1]
                weighted_features = expert_feature * expert_weight
                
                # Accumulate features
                expert_features[expert_mask] += weighted_features
            else:
                expert_outputs[expert_name] = torch.zeros(0, self.feature_dim, device=device)
        
        # Final prediction through shared classifier
        output = self.classifier(expert_features)
        
        # Prepare return dictionary
        result = {
            'logits': output,
            'predictions': output,
            'expert_outputs': expert_outputs,
            'routing_weights': routing_weights,
            'expert_indices': expert_indices,
            'load_balance_loss': routing_info['load_balance_loss'],
            'gating_type': routing_info['gating_type'],
            'expert_utilization': routing_info['expert_utilization'],
            'fingerprint_features': routing_info['fingerprint_features']
        }
        
        return result
    
    def get_expert_distribution(self, routing_weights: torch.Tensor) -> Dict[str, float]:
        """Get distribution of samples across experts"""
        return self.gating_system.get_expert_distribution(routing_weights)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information for saving/loading"""
        info = {
            'config': self.config,
            'expert_names': self.config.experts if self.enable_moe else ['equity_daily'],
            'feature_dim': self.feature_dim,
            'enable_moe': self.enable_moe,
        }
        
        if self.enable_moe:
            info['gating_config'] = self.gating_system.config
        
        return info


def create_atlas_moe_model(
    input_shape: Tuple[int, int, int] = (50, 50, 4),
    enable_moe: bool = True,
    top_k: int = 2,
    use_learnable_gate: bool = True,
    dropout_rate: float = 0.5
) -> AtlasMoEModel:
    """Create ATLAS MoE model with standard configuration"""
    
    config = AtlasMoEConfig(
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        enable_moe=enable_moe,
        top_k=top_k,
        use_learnable_gate=use_learnable_gate
    )
    
    return AtlasMoEModel(config)


# Backward compatibility wrapper
class ATLASModel(AtlasMoEModel):
    """Backward compatibility wrapper for original ATLAS model"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (50, 50, 4), 
                 kernels: Optional[Dict[str, List[np.ndarray]]] = None,
                 dropout_rate: float = 0.5):
        
        # Create config that disables MoE
        config = AtlasMoEConfig(
            input_shape=input_shape,
            dropout_rate=dropout_rate,
            enable_moe=False
        )
        
        super().__init__(config)
        
        # If specific kernels provided, update the single expert
        if kernels is not None:
            self.single_expert = ATLASExpert(
                expert_name='custom',
                input_shape=input_shape,
                kernels=kernels,
                dropout_rate=dropout_rate
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified forward pass for backward compatibility"""
        result = super().forward(x)
        return result['predictions']


# Testing and example usage
if __name__ == "__main__":
    # Test MoE model
    torch.manual_seed(42)
    
    # Test data
    batch_size, height, width, channels = 4, 50, 50, 4
    x = torch.randn(batch_size, channels, height, width)
    
    # Price data for fingerprinting
    price_data = torch.randn(batch_size, 50, 5)  # [batch, seq_len, OHLCV]
    
    print("Testing ATLAS MoE Model:")
    
    # Test MoE enabled
    model_moe = create_atlas_moe_model(
        input_shape=(height, width, channels),
        enable_moe=True,
        top_k=2,
        use_learnable_gate=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model_moe.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = model_moe(x, price_data)
    
    print(f"Output shape: {output['predictions'].shape}")
    print(f"Routing weights shape: {output['routing_weights'].shape}")
    print(f"Load balance loss: {output['load_balance_loss'].item():.4f}")
    print(f"Expert utilization: {output['expert_utilization']}")
    
    distribution = model_moe.get_expert_distribution(output['routing_weights'])
    print(f"Expert distribution: {distribution}")
    
    # Test backward compatibility
    print("\nTesting Backward Compatibility:")
    model_single = ATLASModel(input_shape=(height, width, channels))
    
    with torch.no_grad():
        output_single = model_single(x)
    
    print(f"Single model output shape: {output_single.shape}")
    print(f"Single model parameters: {sum(p.numel() for p in model_single.parameters()):,}")
    
    # Test training mode
    print("\nTesting Training Mode:")
    model_moe.train()
    
    output_train = model_moe(x, price_data)
    loss = torch.mean(output_train['predictions']) + output_train['load_balance_loss']
    
    print(f"Training loss: {loss.item():.4f}")
    print(f"Load balance component: {output_train['load_balance_loss'].item():.4f}")
    
    # Test gradient flow
    loss.backward()
    grad_norms = {}
    for name, param in model_moe.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    print(f"Gradient norms (first 3): {dict(list(grad_norms.items())[:3])}")