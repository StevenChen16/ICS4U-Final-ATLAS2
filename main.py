#!/usr/bin/env python3
"""
ATLAS: Advanced Technical Learning Analysis System
Main Entry Point

Author: Steven Chen
Course: ICS4U Computer Science
Date: May 2025

Usage:
    python main.py                    # Show menu
    python main.py --train           # Train model
    python main.py --dashboard       # Start dashboard
    python main.py --demo            # Run demo
    python main.py --test           # Run performance test
    python main.py --data           # Prepare data
    python main.py --inference      # Run inference
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def print_header():
    """Print ATLAS header"""
    print("ATLAS: Advanced Technical Learning Analysis System")
    print("AI-Powered Stock Market Pattern Recognition")
    print("Author: Steven Chen | Course: ICS4U Final Project")
    print("=" * 60)

def print_menu():
    """Print main menu"""
    print("\nAvailable Commands:")
    print("  1. Train AI Model           (python main.py --train)")
    print("  2. Train MoE Model          (python main.py --train-moe)")
    print("  3. Start Dashboard          (python main.py --dashboard)")
    print("  4. Run Demo                 (python main.py --demo)")
    print("  5. Performance Test         (python main.py --test)")
    print("  6. Prepare Data             (python main.py --data)")
    print("  7. Run Inference            (python main.py --inference)")
    print("  8. Show Help                (python main.py --help)")
    print("\nQuick Start:")
    print("  New Users: python main.py --demo")
    print("  Researchers: python main.py --train")
    print("  MoE Research: python main.py --train-moe")
    print("  Traders: python main.py --dashboard")
    print("  Predictions: python main.py --inference")

def run_data_preparation():
    """Run data preparation"""
    print("\nStarting data download and preparation...")
    try:
        from src.data import main as data_main
        data_main()
        print("Data preparation completed successfully!")
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return False
    return True

def run_training(disable_auto_tuning=False):
    """
    Run model training with optional auto-tuning
    
    Args:
        disable_auto_tuning (bool): If True, disable nnUNet-style auto-parameter tuning
    """
    print("\nStarting ATLAS model training...")
    
    # Check for available data directories
    if os.path.exists("data_short"):
        data_dir = "data_short"
    elif os.path.exists("data"):
        data_dir = "data"
    else:
        print("❌ No data directory found! Please run 'python main.py --data' first.")
        return
    
    # Use a reasonable default ticker list
    ticker_list = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "META", "AMD", "INTC", "CRM"
    ]
    
    print(f"\n📊 Training Configuration:")
    print(f"   📁 Data directory: {data_dir}/")
    print(f"   📈 Training stocks: {len(ticker_list)} tickers")
    print(f"   🎯 Stocks: {', '.join(ticker_list)}")
    
    if not disable_auto_tuning:
        print(f"   🎆 Auto-Tuning: Enabled (nnUNet-style)")
        print(f"      • Automatic parameter optimization based on data characteristics")
        print(f"      • No manual hyperparameter tuning required")
        print(f"      • Adaptive to different market conditions and data sizes")
    else:
        print(f"   📋 Auto-Tuning: Disabled")
        print(f"      • Using predefined hyperparameters")
        print(f"      • Expert-level parameter tuning required for optimal results")
    
    print(f"   ⏱️ Estimated time: 10-30 minutes depending on hardware")
    
    confirm = input("\nContinue with training? (y/N): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("Training cancelled")
        return
    
    try:
        # Import and run training pipeline
        from src.atlas2 import run_atlas_binary_pipeline
        
        print(f"\n🚀 Starting training pipeline...")
        
        model, test_data = run_atlas_binary_pipeline(
            ticker_list=ticker_list,
            data_dir=data_dir,
            epochs=30,  # Reasonable for demo
            enable_auto_tuning=not disable_auto_tuning,  # 🎆 Key feature!
        )
        
        print("\n🎉 Model training completed successfully!")
        print("   📁 Model saved to models/ directory")
        print("   📊 Training results saved to results/ directory")
        
        if not disable_auto_tuning:
            print("\n✨ Auto-tuning optimization completed!")
            print("   ⚙️ Configuration automatically adapted to your data characteristics")
        else:
            print("\n📋 Manual configuration training completed!")
            print("   💡 Consider enabling auto-tuning next time: --train (without --no-auto-tuning)")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n🔧 Troubleshooting suggestions:")
        print("   1. Check data file integrity with 'python main.py --demo'")
        print("   2. Ensure sufficient disk space (>1GB free)")
        print("   3. Verify GPU/CPU resources")
        print("   4. Try manual mode: 'python main.py --train --no-auto-tuning'")
        sys.exit(1)

def run_moe_training(disable_auto_tuning=False):
    """
    Run MoE model training with expert routing
    
    Args:
        disable_auto_tuning (bool): If True, disable nnUNet-style auto-parameter tuning
    """
    print("\nStarting ATLAS MoE model training...")
    print("🧠 Mixture-of-Experts (MoE) enables adaptive expert routing based on market characteristics")
    
    # Check for available data directories
    if os.path.exists("data_short"):
        data_dir = "data_short"
    elif os.path.exists("data"):
        data_dir = "data"
    else:
        print("❌ No data directory found! Please run 'python main.py --data' first.")
        return
    
    # Use a reasonable default ticker list
    ticker_list = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "META", "AMD", "INTC", "CRM",
        "JPM", "BAC", "BABA", 
        'BNB-USD', 'BTC-USD', 'DOGE-USD', 'ETH-USD', 'LINK-USD', 'LTC-USD', 'XRP-USD', 
        'EURGBP', 'EURUSD', 'GBPUSD', 'USDCHF', 'USDCNY', 'USDJPY',
        'processed_000001_XSHE_short', 'processed_000002_XSHE_short', 'processed_000004_XSHE_short', 'processed_000005_XSHE_short'
    ]
    
    print(f"\n📊 MoE Training Configuration:")
    print(f"   📁 Data directory: {data_dir}/")
    print(f"   📈 Training stocks: {len(ticker_list)} tickers")
    print(f"   🎯 Stocks: {', '.join(ticker_list[:10])}{'...' if len(ticker_list) > 10 else ''}")
    print(f"   🧠 Experts: crypto_hft, equity_intraday, equity_daily, futures_trend, low_vol_etf")
    print(f"   🔀 Top-K routing: 2 experts per sample")
    
    if not disable_auto_tuning:
        print(f"   🎆 Auto-Tuning: Enabled (nnUNet-style)")
        print(f"      • Automatic parameter optimization for each expert")
        print(f"      • Data-driven expert selection and routing")
    else:
        print(f"   📋 Auto-Tuning: Disabled")
        print(f"      • Using predefined hyperparameters")
    
    print(f"   ⏱️ Estimated time: 15-45 minutes depending on hardware")
    
    # Import and run MoE training pipeline
    from src.atlas_moe_pipeline import run_atlas_moe_pipeline
    
    print(f"\n🚀 Starting MoE training pipeline...")
    
    model, test_data, history = run_atlas_moe_pipeline(
        ticker_list=ticker_list,
        data_dir=data_dir,
        epochs=30,  # Reasonable for demo
        enable_auto_tuning=not disable_auto_tuning,
        enable_moe=True,
        use_learnable_gate=True,
        top_k=2,
        load_balance_weight=0.01
    )
    
    print("\n🎉 MoE model training completed successfully!")
    print("   📁 Model saved to models/ directory")
    print("   📊 Training results saved to results/ directory")
    print("   🧠 Expert routing information included")
    
    # Show expert utilization summary
    _, _, _, _, routing_info = test_data
    if routing_info:
        print("\n📈 Expert Utilization Summary:")
        expert_names = ['crypto_hft', 'equity_intraday', 'equity_daily', 'futures_trend', 'low_vol_etf']
        avg_utilization = sum([info['expert_utilization'] for info in routing_info]) / len(routing_info)
        
        for i, name in enumerate(expert_names):
            print(f"   {name}: {avg_utilization[i]:.1%}")
    
    if not disable_auto_tuning:
        print("\n✨ Auto-tuning optimization completed!")
        print("   ⚙️ Configuration automatically adapted for each expert")
    else:
        print("\n📋 Manual configuration training completed!")
        print("   💡 Consider enabling auto-tuning next time for better performance")

def run_dashboard():
    """Start dashboard"""
    print("\nStarting ATLAS real-time dashboard...")
    print("Dashboard will be available at http://localhost:8050")
    print("Note: Please ensure internet connection for real-time data")
    
    try:
        from src.dashboard import app
        print("Starting Dash server...")
        app.run(debug=False, host="0.0.0.0", port=8050)
    except Exception as e:
        print(f"Dashboard startup failed: {e}")

def run_inference():
    """Run inference module"""
    print("\nStarting ATLAS inference system...")
    print("This will launch the interactive stock prediction interface.")
    print("You can predict individual stocks, batch process multiple stocks,")
    print("or analyze data from CSV files.")
    
    try:
        # 直接导入 ATLASInferenceEngine 而不是调用 main
        from src.inference import ATLASInferenceEngine
        
        # 初始化推理引擎
        engine = ATLASInferenceEngine()
        
        # 交互式模式
        print("\n🎯 Interactive Prediction Mode")
        print("Please select prediction method:")
        print("1. Single stock prediction")
        print("2. Batch prediction") 
        print("3. File prediction")
        
        choice = input("\nPlease enter choice (1-3): ").strip()
        
        if choice == "1":
            ticker = input("Please enter stock ticker: ").strip().upper()
            start_date = input("Please enter start date (default 2023-01-01): ").strip() or "2023-01-01"
            result = engine.predict_single_ticker(ticker, start_date=start_date)
            
        elif choice == "2":
            tickers_input = input("Please enter stock tickers (comma separated): ").strip()
            ticker_list = [t.strip().upper() for t in tickers_input.split(',')]
            start_date = input("Please enter start date (default 2023-01-01): ").strip() or "2023-01-01"
            result = engine.predict_batch(ticker_list, start_date=start_date)
            
        elif choice == "3":
            file_path = input("Please enter CSV file path: ").strip()
            result = engine.predict_from_file(file_path)
            
        else:
            print("❌ Invalid choice")
            return False
            
        print(f"\n✅ Prediction completed!")
        print("\n💡 Disclaimer: This prediction is for reference only and does not constitute investment advice.")
        
    except ImportError as e:
        print(f"❌ Failed to import inference module: {e}")
        print("Please ensure inference.py is in the src/ directory")
        return False
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False
    
    return True

def run_demo():
    """Run system demo"""
    print("\nATLAS System Demo")
    print("-" * 30)
    
    # Check required files
    model_path = project_root / "models" / "atlas_binary_model_best.pth"
    data_path = project_root / "data_short"
    
    if not model_path.exists():
        print("Error: No trained model found!")
        print("Please run: python main.py --train")
        return
    
    if not data_path.exists() or len(list(data_path.glob("*.csv"))) == 0:
        print("Error: No test data found!")
        print("Please run: python main.py --data")
        return
    
    print("System check passed!")
    print("\nDemo includes:")
    print("1. Model architecture and performance metrics")
    print("2. Sample prediction demonstration")
    print("3. Cross-platform performance results")
    
    try:
        # Import and run demo
        import torch
        import pandas as pd
        from src.atlas2 import ATLASModel, create_specialized_kernels
        import joblib
        
        print("\nModel Information:")
        model_info = joblib.load("models/atlas_binary_model_info.pkl")
        print(f"  Validation Accuracy: {model_info['test_accuracy']:.1%}")
        print(f"  Window Size: {model_info['window_size']}")
        print(f"  Input Shape: {model_info['input_shape']}")
        print(f"  Classes: {', '.join(model_info['class_names'])}")

        print("\nModel Architecture:")
        kernels = create_specialized_kernels()
        model = ATLASModel(input_shape=model_info['input_shape'], kernels=kernels)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Model Size: {total_params * 4 / (1024*1024):.2f} MB")
        print(f"  Specialized Kernels: {len(kernels['gasf']) + len(kernels['gadf']) + len(kernels['rp']) + len(kernels['mtf'])}")

        print("\nDataset Information:")
        data_files = list(Path("data_short").glob("*.csv"))
        print(f"  Test Stocks: {len(data_files)}")
        print(f"  Stock Symbols: {', '.join([f.stem for f in data_files[:10]])}{'...' if len(data_files) > 10 else ''}")

        print("\nDemo completed! System is ready.")
        
    except Exception as e:
        print(f"Demo failed: {e}")

def run_performance_test():
    """Run performance test"""
    print("\nATLAS Performance Benchmark Test")
    try:
        from tests.atlas_performance_test import main as test_main
        test_main()
    except Exception as e:
        print(f"Performance test failed: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="ATLAS: Advanced Technical Learning Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python main.py                    # Show interactive menu
  python main.py --train           # Train model
  python main.py --train-moe       # Train MoE model
  python main.py --dashboard       # Start dashboard
  python main.py --demo            # Run demo
  python main.py --test           # Performance test
  python main.py --data           # Prepare data
        """
    )
    
    parser.add_argument("--train", action="store_true", help="Train ATLAS model with optional auto-tuning (nnUNet-style)")
    parser.add_argument("--train-moe", action="store_true", help="Train ATLAS MoE model with expert routing")
    parser.add_argument("--no-auto-tuning", action="store_true", help="Disable automatic parameter optimization (use predefined parameters)")
    parser.add_argument("--dashboard", action="store_true", help="Start real-time dashboard")
    parser.add_argument("--demo", action="store_true", help="Run system demo")
    parser.add_argument("--test", action="store_true", help="Run performance benchmark")
    parser.add_argument("--data", action="store_true", help="Download and prepare data")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    
    args = parser.parse_args()
    
    # Always show header
    print_header()
    
    # Execute based on arguments
    if args.train:
        run_training(disable_auto_tuning=args.no_auto_tuning)
    elif args.train_moe:
        run_moe_training(disable_auto_tuning=args.no_auto_tuning)
    elif args.dashboard:
        run_dashboard()
    elif args.demo:
        run_demo()
    elif args.test:
        run_performance_test()
    elif args.data:
        run_data_preparation()
    elif args.inference:
        run_inference()
    else:
        # Show interactive menu if no arguments
        print_menu()
        
        while True:
            try:
                choice = input("\nSelect option (1-8, or 'q' to quit): ").strip().lower()
                
                if choice in ['q', 'quit', 'exit']:
                    print("Thank you for using ATLAS!")
                    break
                elif choice == '1':
                    # Ask about auto-tuning in interactive mode
                    print("\n🤖 Training Mode Selection:")
                    print("  1. Auto-Tuning Mode (Recommended) - nnUNet-style parameter optimization")
                    print("  2. Manual Mode - Use predefined parameters")
                    mode_choice = input("Select training mode (1/2): ").strip()
                    disable_auto = mode_choice == '2'
                    run_training(disable_auto_tuning=disable_auto)
                elif choice == '2':
                    # Ask about auto-tuning for MoE
                    print("\n🧠 MoE Training Mode Selection:")
                    print("  1. Auto-Tuning Mode (Recommended) - Expert-specific parameter optimization")
                    print("  2. Manual Mode - Use predefined parameters")
                    mode_choice = input("Select MoE training mode (1/2): ").strip()
                    disable_auto = mode_choice == '2'
                    run_moe_training(disable_auto_tuning=disable_auto)
                elif choice == '3':
                    run_dashboard()
                elif choice == '4':
                    run_demo()
                elif choice == '5':
                    run_performance_test()
                elif choice == '6':
                    run_data_preparation()
                elif choice == '7':
                    run_inference()
                elif choice == '8':
                    parser.print_help()
                else:
                    print("Invalid choice. Please enter 1-8 or 'q'")
                    
            except KeyboardInterrupt:
                print("\n\nThank you for using ATLAS!")
                break
            except Exception as e:
                print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
