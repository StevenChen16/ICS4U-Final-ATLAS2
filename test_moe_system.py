#!/usr/bin/env python3
"""
Test script for ATLAS MoE System

This script tests the MoE components to ensure they work correctly.

Author: Steven Chen
Date: 2025-06-14
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_data_fingerprint():
    """Test data fingerprint extraction"""
    print("=" * 50)
    print("Testing Data Fingerprint System")
    print("=" * 50)
    
    try:
        from src.data_fingerprint import DataFingerprinter, classify_market_archetype, get_expert_weights
        
        # Create synthetic test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # High volatility data (crypto-like)
        close_high_vol = 100 * np.exp(np.cumsum(np.random.normal(0, 0.1, 100)))
        high_high_vol = close_high_vol * (1 + np.random.uniform(0, 0.05, 100))
        low_high_vol = close_high_vol * (1 - np.random.uniform(0, 0.05, 100))
        open_high_vol = close_high_vol + np.random.normal(0, 2, 100)
        volume_high_vol = np.random.lognormal(12, 1, 100)
        
        df_high_vol = pd.DataFrame({
            'Date': dates,
            'Open': open_high_vol,
            'High': high_high_vol,
            'Low': low_high_vol,
            'Close': close_high_vol,
            'Volume': volume_high_vol
        })
        
        # Low volatility data (bond ETF-like)
        close_low_vol = 100 + np.cumsum(np.random.normal(0, 0.01, 100))
        high_low_vol = close_low_vol * (1 + np.random.uniform(0, 0.005, 100))
        low_low_vol = close_low_vol * (1 - np.random.uniform(0, 0.005, 100))
        open_low_vol = close_low_vol + np.random.normal(0, 0.1, 100)
        volume_low_vol = np.random.lognormal(8, 0.5, 100)
        
        df_low_vol = pd.DataFrame({
            'Date': dates,
            'Open': open_low_vol,
            'High': high_low_vol,
            'Low': low_low_vol,
            'Close': close_low_vol,
            'Volume': volume_low_vol
        })
        
        fingerprinter = DataFingerprinter()
        
        # Test high volatility data
        fp_high = fingerprinter.extract_fingerprint(df_high_vol)
        archetype_high = classify_market_archetype(fp_high)
        weights_high = get_expert_weights(fp_high)
        
        print("High Volatility Data:")
        print(f"  Volatility: {fp_high['vol30']:.3f}")
        print(f"  Classified as: {archetype_high}")
        print(f"  Top expert weights: {dict(list(weights_high.items())[:3])}")
        
        # Test low volatility data  
        fp_low = fingerprinter.extract_fingerprint(df_low_vol)
        archetype_low = classify_market_archetype(fp_low)
        weights_low = get_expert_weights(fp_low)
        
        print("\nLow Volatility Data:")
        print(f"  Volatility: {fp_low['vol30']:.3f}")
        print(f"  Classified as: {archetype_low}")
        print(f"  Top expert weights: {dict(list(weights_low.items())[:3])}")
        
        print("✅ Data fingerprint test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data fingerprint test failed: {e}")
        return False


def test_expert_kernels():
    """Test expert kernel banks"""
    print("\n" + "=" * 50)
    print("Testing Expert Kernel Banks")
    print("=" * 50)
    
    try:
        from src.expert_kernels import ExpertKernelManager
        
        manager = ExpertKernelManager()
        
        print("Available experts:")
        for expert_name in manager.get_all_experts():
            kernels = manager.get_expert_kernels(expert_name)
            counts = manager.get_expert_kernel_counts(expert_name)
            total_kernels = sum(counts.values())
            
            print(f"  {expert_name}: {total_kernels} kernels ({counts})")
            
            # Check kernel shapes
            for img_type, kernel_list in kernels.items():
                if kernel_list:
                    shape = kernel_list[0].shape
                    print(f"    {img_type}: {len(kernel_list)} kernels, shape {shape}")
        
        # Test mixed kernels
        expert_weights = {'crypto_hft': 0.3, 'equity_daily': 0.7}
        mixed_kernels = manager.create_mixed_kernels(expert_weights)
        
        print(f"\nMixed kernels test:")
        print(f"  Input weights: {expert_weights}")
        print(f"  Output kernel counts: {dict((k, len(v)) for k, v in mixed_kernels.items())}")
        
        print("✅ Expert kernels test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Expert kernels test failed: {e}")
        return False


def test_moe_gating():
    """Test MoE gating system"""
    print("\n" + "=" * 50)
    print("Testing MoE Gating System")
    print("=" * 50)
    
    try:
        from src.moe_gating import create_moe_gating_system
        
        # Create test data
        torch.manual_seed(42)
        batch_size, seq_len, num_features = 4, 50, 5
        price_data = torch.randn(batch_size, seq_len, num_features)
        
        # Simulate realistic OHLCV data
        price_data[:, :, 3] = torch.cumsum(torch.randn(batch_size, seq_len) * 0.02, dim=1) + 100  # Close
        price_data[:, :, 0] = price_data[:, :, 3] + torch.randn(batch_size, seq_len) * 0.5  # Open
        price_data[:, :, 1] = price_data[:, :, 3] + torch.abs(torch.randn(batch_size, seq_len)) * 2  # High
        price_data[:, :, 2] = price_data[:, :, 3] - torch.abs(torch.randn(batch_size, seq_len)) * 2  # Low
        price_data[:, :, 4] = torch.abs(torch.randn(batch_size, seq_len)) * 1000000  # Volume
        
        # Test rule-based gating
        print("Testing rule-based gating:")
        gating_system = create_moe_gating_system(use_learnable=False)
        
        with torch.no_grad():
            output = gating_system(price_data, use_rule_based=True)
        
        print(f"  Routing weights shape: {output['routing_weights'].shape}")
        print(f"  Expert utilization: {output['expert_utilization']}")
        print(f"  Load balance loss: {output['load_balance_loss'].item():.4f}")
        
        # Test learnable gating
        print("\nTesting learnable gating:")
        gating_system = create_moe_gating_system(use_learnable=True)
        
        output = gating_system(price_data, use_rule_based=False)
        
        print(f"  Routing weights shape: {output['routing_weights'].shape}")
        print(f"  Expert utilization: {output['expert_utilization']}")
        print(f"  Load balance loss: {output['load_balance_loss'].item():.4f}")
        
        # Test gradient flow
        dummy_loss = output['load_balance_loss'] + torch.sum(output['routing_weights'])
        dummy_loss.backward()
        
        grad_count = 0
        for name, param in gating_system.named_parameters():
            if param.grad is not None:
                grad_count += 1
        
        print(f"  Parameters with gradients: {grad_count}")
        
        print("✅ MoE gating test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MoE gating test failed: {e}")
        return False


def test_moe_model():
    """Test complete MoE model"""
    print("\n" + "=" * 50)
    print("Testing Complete MoE Model")
    print("=" * 50)
    
    try:
        from src.atlas_moe import create_atlas_moe_model
        
        torch.manual_seed(42)
        
        # Test data
        batch_size, height, width, channels = 2, 50, 50, 4
        x = torch.randn(batch_size, channels, height, width)
        price_data = torch.randn(batch_size, 50, 5)
        
        # Test MoE model
        print("Testing MoE model:")
        model_moe = create_atlas_moe_model(
            input_shape=(height, width, channels),
            enable_moe=True,
            top_k=2,
            use_learnable_gate=True
        )
        
        print(f"  Model parameters: {sum(p.numel() for p in model_moe.parameters()):,}")
        
        with torch.no_grad():
            output = model_moe(x, price_data)
        
        print(f"  Output shape: {output['predictions'].shape}")
        print(f"  Routing weights shape: {output['routing_weights'].shape}")
        print(f"  Load balance loss: {output['load_balance_loss'].item():.4f}")
        
        # Test backward compatibility
        print("\nTesting backward compatibility:")
        from src.atlas_moe import ATLASModel
        
        model_single = ATLASModel(input_shape=(height, width, channels))
        
        with torch.no_grad():
            output_single = model_single(x)
        
        print(f"  Single model output shape: {output_single.shape}")
        print(f"  Single model parameters: {sum(p.numel() for p in model_single.parameters()):,}")
        
        print("✅ MoE model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MoE model test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("ATLAS MoE System Test Suite")
    print("Testing all components...")
    
    tests = [
        test_data_fingerprint,
        test_expert_kernels, 
        test_moe_gating,
        test_moe_model
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [func.__name__ for func in tests]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MoE system is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)