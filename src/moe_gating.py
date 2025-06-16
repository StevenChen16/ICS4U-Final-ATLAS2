"""
Mixture-of-Experts Gating Network and Routing for ATLAS

This module implements the gating mechanism that routes inputs to appropriate
experts based on data fingerprints and learned patterns.

Author: Steven Chen
Date: 2025-06-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# --------- Sparsemax 实现 ---------
def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    PyTorch-native sparsemax (Martins & Astudillo, 2016).
    Returns a **sparse** probability simplex like Softmax → but with hard zeros.
    """
    original_size = logits.size()
    logits = logits.transpose(dim, -1)          # [..., d] → [..., d] with dim=-1
    logits = logits.reshape(-1, logits.size(-1))

    z_sorted, _ = torch.sort(logits, descending=True, dim=-1)
    k = torch.arange(1, z_sorted.size(1)+1, device=logits.device).float()
    z_cumsum = torch.cumsum(z_sorted, dim=-1)
    k_mask = (1 + k * z_sorted) > z_cumsum
    k_max = k_mask.float().argmax(dim=-1) + 1
    # τ = (Σ z_i − 1) / k
    tau = (z_cumsum[torch.arange(z_cumsum.size(0)), k_max-1] - 1) / k_max
    out = torch.clamp(logits - tau.unsqueeze(-1), min=0.0)
    out = out.reshape(original_size).transpose(dim, -1)
    return out

from .data_fingerprint import DataFingerprinter, classify_market_archetype, get_expert_weights


@dataclass
class GatingConfig:
    """Configuration for MoE gating"""
    num_experts: int = 5
    top_k: int = 2
    hidden_dim: int = 64
    intermediate_dim: int = 32
    load_balance_weight: float = 0.01
    noise_std: float = 0.1
    expert_capacity_factor: float = 1.25
    use_rule_based_warmstart: bool = True
    # ---- 新增超参 ----
    temperature_start: float = 2.0        # 退火起始温度
    temperature_end:   float = 0.5        # 退火终止温度
    temperature_decay: float = 5e-4       # 每 step 乘以 (1-decay)
    use_sparsemax:     bool  = False      # True → sparsemax；False → softmax / gumbel
    use_gumbel_hard:   bool  = False      # True → Hard Gumbel-Top-1 (训练期)


class RuleBasedGate(nn.Module):
    """Rule-based gating mechanism as warm-start (v0)"""
    
    def __init__(self, config: GatingConfig):
        super().__init__()
        self.config = config
        self.expert_names = ['crypto_hft', 'equity_intraday', 'equity_daily', 'futures_trend', 'low_vol_etf']
        self.fingerprinter = DataFingerprinter()
    
    def forward(self, fingerprint_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rule-based routing
        
        Args:
            fingerprint_features: Tensor of shape [batch_size, num_features]
            
        Returns:
            gates: Gating weights [batch_size, num_experts]
            load_balance_loss: Load balancing loss (zero for rule-based)
        """
        batch_size = fingerprint_features.shape[0]
        device = fingerprint_features.device
        
        gates = torch.zeros(batch_size, self.config.num_experts, device=device)
        
        # Convert tensor to numpy for fingerprint processing
        features_np = fingerprint_features.detach().cpu().numpy()
        
        for i in range(batch_size):
            # Create fingerprint dict from features
            # Assuming features are ordered: [vol30, trend_r2, hurst, spread, volume_z, atr_pct, ...]
            fingerprint = {
                'vol30': float(features_np[i, 0]) if features_np.shape[1] > 0 else 0.0,
                'trend_r2': float(features_np[i, 1]) if features_np.shape[1] > 1 else 0.0,
                'hurst': float(features_np[i, 2]) if features_np.shape[1] > 2 else 0.5,
                'spread_estimate': float(features_np[i, 3]) if features_np.shape[1] > 3 else 0.0,
                'volume_z': float(features_np[i, 4]) if features_np.shape[1] > 4 else 0.0,
                'atr_pct': float(features_np[i, 5]) if features_np.shape[1] > 5 else 0.0,
            }
            
            # Get expert weights using rule-based classification
            expert_weights = get_expert_weights(fingerprint)
            
            # Map to tensor
            for j, expert_name in enumerate(self.expert_names):
                gates[i, j] = expert_weights.get(expert_name, 0.0)
        
        # No load balancing loss for rule-based gating
        load_balance_loss = torch.tensor(0.0, device=device)
        
        return gates, load_balance_loss


class LearnableGate(nn.Module):
    """Learnable MLP-based gating network (v1)"""
    
    def __init__(self, input_dim: int, config: GatingConfig):
        super().__init__()
        self.config = config
        
        # MLP layers
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_dim, config.num_experts)
        )
        
        # Exploration & 退火温度
        self.noise_std = config.noise_std
        self.register_buffer("temperature", torch.tensor(config.temperature_start))
        
    def forward(self, fingerprint_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learnable gating
        
        Args:
            fingerprint_features: Tensor of shape [batch_size, input_dim]
            
        Returns:
            gates: Gating weights [batch_size, num_experts]
            load_balance_loss: Load balancing loss
        """
        batch_size = fingerprint_features.shape[0]
        
        # Forward through gate network
        logits = self.gate_network(fingerprint_features)
        
        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # ------ 温度退火 ------
        # 训练阶段按 decay 指数退火；推理保持最低温度
        if self.training:
            self.temperature = torch.clamp(
                self.temperature * (1.0 - self.config.temperature_decay),
                min=self.config.temperature_end
            )
        logits = logits / self.temperature
        
        # ------ 概率映射：Softmax / Sparsemax / Gumbel ------
        if self.config.use_gumbel_hard and self.training:
            # 可微 Gumbel-Softmax；hard=False 留连续概率给 Top-k
            gates = F.gumbel_softmax(logits, tau=self.temperature.item(),
                                     hard=False, dim=-1)
        elif self.config.use_sparsemax:
            gates = _sparsemax(logits, dim=-1)
        else:
            gates = F.softmax(logits, dim=-1)
        
        # Calculate load balancing loss
        load_balance_loss = self._calculate_load_balance_loss(gates)
        
        return gates, load_balance_loss
    
    def _calculate_load_balance_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """Calculate load balancing loss to prevent expert collapse"""
        # Average gate values across batch
        gate_means = gates.mean(dim=0)  # [num_experts]
        
        # Calculate entropy-based load balancing loss
        # Higher entropy = more balanced load
        entropy = -torch.sum(gate_means * torch.log(gate_means + 1e-8))
        max_entropy = np.log(self.config.num_experts)
        
        # Loss is negative normalized entropy (we want to maximize entropy)
        load_balance_loss = -(entropy / max_entropy)
        
        return load_balance_loss * self.config.load_balance_weight


class TopKRouter(nn.Module):
    """Top-K sparse routing mechanism"""
    
    def __init__(self, config: GatingConfig):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.expert_capacity_factor = config.expert_capacity_factor
    
    def forward(self, gates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply top-k routing
        
        Args:
            gates: Gating weights [batch_size, num_experts]
            
        Returns:
            routing_weights: Sparse routing weights [batch_size, num_experts]
            expert_indices: Selected expert indices [batch_size, top_k]
            routing_info: Additional routing information
        """
        batch_size, num_experts = gates.shape
        device = gates.device
        
        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        
        # Renormalize top-k weights
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Create sparse routing tensor
        routing_weights = torch.zeros_like(gates)
        
        # Fill in top-k weights
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        routing_weights[batch_indices, top_k_indices] = top_k_weights
        
        # Calculate expert utilization for monitoring
        expert_utilization = torch.sum(routing_weights > 0, dim=0).float() / batch_size
        
        # Calculate capacity constraints (optional)
        capacity_per_expert = int(batch_size * self.expert_capacity_factor / num_experts)
        expert_loads = torch.sum(routing_weights > 0, dim=0)
        capacity_exceeded = expert_loads > capacity_per_expert
        
        routing_info = {
            'expert_utilization': expert_utilization,
            'expert_loads': expert_loads,
            'capacity_exceeded': capacity_exceeded,
            'top_k_indices': top_k_indices,
            'top_k_weights': top_k_weights
        }
        
        return routing_weights, top_k_indices, routing_info


class MoEGatingSystem(nn.Module):
    """Complete MoE gating system combining fingerprinting, gating, and routing"""
    
    def __init__(self, config: GatingConfig, use_learnable_gate: bool = True):
        super().__init__()
        self.config = config
        self.use_learnable_gate = use_learnable_gate
        
        # Feature dimensions for fingerprint
        self.fingerprint_dim = 9  # Number of features in fingerprint
        
        # Gating mechanisms
        self.rule_based_gate = RuleBasedGate(config)
        
        if use_learnable_gate:
            self.learnable_gate = LearnableGate(self.fingerprint_dim, config)
        
        # Router
        self.router = TopKRouter(config)
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(self.fingerprint_dim)
        
        # Expert names for reference
        self.expert_names = ['crypto_hft', 'equity_intraday', 'equity_daily', 'futures_trend', 'low_vol_etf']
    
    def extract_fingerprint_features(self, price_data: torch.Tensor) -> torch.Tensor:
        """
        Extract fingerprint features from price data
        
        Args:
            price_data: Raw price data [batch_size, seq_len, num_price_features]
                       Expected features: [Open, High, Low, Close, Volume]
            
        Returns:
            fingerprint_features: [batch_size, fingerprint_dim]
        """
        batch_size = price_data.shape[0]
        device = price_data.device
        
        fingerprint_features = torch.zeros(batch_size, self.fingerprint_dim, device=device)
        
        # Convert to numpy for fingerprint extraction
        price_data_np = price_data.detach().cpu().numpy()
        
        fingerprinter = DataFingerprinter()
        
        for i in range(batch_size):
            # Convert sequence to DataFrame format
            seq_data = price_data_np[i]  # [seq_len, num_features]
            
            # Create minimal DataFrame (assuming OHLCV format)
            if seq_data.shape[1] >= 4:  # At least OHLC
                df_dict = {
                    'Open': seq_data[:, 0],
                    'High': seq_data[:, 1], 
                    'Low': seq_data[:, 2],
                    'Close': seq_data[:, 3]
                }
                
                if seq_data.shape[1] >= 5:  # Include volume
                    df_dict['Volume'] = seq_data[:, 4]
                
                import pandas as pd
                df = pd.DataFrame(df_dict)
                
                try:
                    # Check if we have enough data points
                    if len(df) < 30:
                        # Use simplified features for small datasets
                        close_prices = df['Close'].values
                        if len(close_prices) > 1:
                            volatility = np.std(np.diff(close_prices) / close_prices[:-1])
                            trend = (close_prices[-1] - close_prices[0]) / close_prices[0]
                        else:
                            volatility = 0.0
                            trend = 0.0
                        
                        feature_vector = [
                            volatility * 252**0.5,  # Annualized vol
                            abs(trend),  # Trend strength
                            0.5,  # Default hurst
                            0.001,  # Default spread
                            0.0,  # Volume z-score
                            volatility,  # ATR proxy
                            0.0,  # Default skewness
                            3.0,  # Default kurtosis
                            trend   # Momentum proxy
                        ]
                    else:
                        # Extract full fingerprint
                        fingerprint = fingerprinter.extract_fingerprint(df)
                        
                        # Convert to tensor features
                        feature_vector = [
                            fingerprint.get('vol30', 0.0),
                            fingerprint.get('trend_r2', 0.0),
                            fingerprint.get('hurst', 0.5),
                            fingerprint.get('spread_estimate', 0.0),
                            fingerprint.get('volume_z', 0.0),
                            fingerprint.get('atr_pct', 0.0),
                            fingerprint.get('skewness', 0.0),
                            fingerprint.get('kurtosis', 0.0),
                            fingerprint.get('momentum_10', 0.0)
                        ]
                    
                    fingerprint_features[i] = torch.tensor(feature_vector, device=device)
                    
                except Exception as e:
                    # Fallback to default features for this sample
                    print(f"Warning: Fingerprint extraction failed for sample {i}: {e}")
                    fingerprint_features[i] = torch.zeros(self.fingerprint_dim, device=device)
        
        # Normalize features
        fingerprint_features = self.feature_norm(fingerprint_features)
        
        return fingerprint_features
    
    def forward(self, price_data: torch.Tensor, use_rule_based: bool = None) -> Dict[str, torch.Tensor]:
        """
        Complete gating forward pass
        
        Args:
            price_data: Input price data [batch_size, seq_len, num_features]
            use_rule_based: Whether to use rule-based gating (None = auto-decide)
            
        Returns:
            Dictionary containing routing information
        """
        # Extract fingerprint features
        fingerprint_features = self.extract_fingerprint_features(price_data)
        
        # Decide which gating to use
        if use_rule_based is None:
            use_rule_based = (not self.use_learnable_gate) or self.config.use_rule_based_warmstart
        
        # Apply gating
        if use_rule_based:
            gates, load_balance_loss = self.rule_based_gate(fingerprint_features)
            gating_type = "rule_based"
        else:
            gates, load_balance_loss = self.learnable_gate(fingerprint_features)
            gating_type = "learnable"
        
        # Apply routing
        routing_weights, expert_indices, routing_info = self.router(gates)
        
        # Combine all information
        output = {
            'routing_weights': routing_weights,  # [batch_size, num_experts]
            'expert_indices': expert_indices,   # [batch_size, top_k]
            'fingerprint_features': fingerprint_features,  # [batch_size, fingerprint_dim]
            'gates': gates,  # [batch_size, num_experts]
            'load_balance_loss': load_balance_loss,  # scalar
            'gating_type': gating_type,
            **routing_info
        }
        
        return output
    
    def get_expert_distribution(self, routing_weights: torch.Tensor) -> Dict[str, float]:
        """Get distribution of samples across experts"""
        expert_usage = torch.sum(routing_weights > 0, dim=0).float()
        total_usage = torch.sum(expert_usage)
        
        distribution = {}
        for i, expert_name in enumerate(self.expert_names):
            distribution[expert_name] = (expert_usage[i] / total_usage).item() if total_usage > 0 else 0.0
        
        return distribution


# Utility functions for integration

def create_moe_gating_system(
    use_learnable: bool = True,
    top_k: int = 2,
    load_balance_weight: float = 0.01,
    use_sparsemax: bool = False,
    use_gumbel_hard: bool = False,
    temperature_start: float = 2.0,
    temperature_end: float = 0.5,
    temperature_decay: float = 5e-4
) -> MoEGatingSystem:
    """Create MoE gating system with standard configuration"""
    config = GatingConfig(
        num_experts=5,
        top_k=top_k,
        hidden_dim=64,
        intermediate_dim=32,
        load_balance_weight=load_balance_weight,
        noise_std=0.1,
        expert_capacity_factor=1.25,
        use_rule_based_warmstart=True,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        temperature_decay=temperature_decay,
        use_sparsemax=use_sparsemax,
        use_gumbel_hard=use_gumbel_hard
    )
    
    return MoEGatingSystem(config, use_learnable_gate=use_learnable)


# Testing and example usage
if __name__ == "__main__":
    # Test the gating system
    torch.manual_seed(42)
    
    # Create test data (batch_size=8, seq_len=50, features=5 for OHLCV)
    batch_size, seq_len, num_features = 8, 50, 5
    price_data = torch.randn(batch_size, seq_len, num_features)
    
    # Simulate realistic OHLCV data
    price_data[:, :, 3] = torch.cumsum(torch.randn(batch_size, seq_len) * 0.02, dim=1) + 100  # Close
    price_data[:, :, 0] = price_data[:, :, 3] + torch.randn(batch_size, seq_len) * 0.5  # Open
    price_data[:, :, 1] = price_data[:, :, 3] + torch.abs(torch.randn(batch_size, seq_len)) * 2  # High
    price_data[:, :, 2] = price_data[:, :, 3] - torch.abs(torch.randn(batch_size, seq_len)) * 2  # Low
    price_data[:, :, 4] = torch.abs(torch.randn(batch_size, seq_len)) * 1000000  # Volume
    
    # Test rule-based gating
    print("Testing Rule-based Gating:")
    gating_system = create_moe_gating_system(use_learnable=False)
    
    with torch.no_grad():
        output = gating_system(price_data, use_rule_based=True)
    
    print(f"Routing weights shape: {output['routing_weights'].shape}")
    print(f"Expert indices shape: {output['expert_indices'].shape}")
    print(f"Load balance loss: {output['load_balance_loss'].item():.4f}")
    print(f"Expert utilization: {output['expert_utilization']}")
    
    distribution = gating_system.get_expert_distribution(output['routing_weights'])
    print(f"Expert distribution: {distribution}")
    
    # Test learnable gating
    print("\nTesting Learnable Gating:")
    gating_system = create_moe_gating_system(use_learnable=True)
    
    output = gating_system(price_data, use_rule_based=False)
    
    print(f"Routing weights shape: {output['routing_weights'].shape}")
    print(f"Load balance loss: {output['load_balance_loss'].item():.4f}")
    print(f"Expert utilization: {output['expert_utilization']}")
    
    distribution = gating_system.get_expert_distribution(output['routing_weights'])
    print(f"Expert distribution: {distribution}")
    
    # Test gradient flow
    print("\nTesting gradient flow:")
    dummy_loss = output['load_balance_loss'] + torch.sum(output['routing_weights'])
    dummy_loss.backward()
    
    for name, param in gating_system.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
        else:
            print(f"{name}: no gradient")