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
    print("  2. Start Dashboard          (python main.py --dashboard)")
    print("  3. Run Demo                 (python main.py --demo)")
    print("  4. Performance Test         (python main.py --test)")
    print("  5. Prepare Data             (python main.py --data)")
    print("  6. Show Help                (python main.py --help)")
    print("\nQuick Start:")
    print("  New Users: python main.py --demo")
    print("  Researchers: python main.py --train")
    print("  Traders: python main.py --dashboard")

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

def run_training():
    """Run model training"""
    print("\nStarting ATLAS model training...")
    print("Note: Training may take 10-30 minutes depending on hardware")
    
    confirm = input("Continue with training? (y/N): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("Training cancelled")
        return
    
    try:
        from src.atlas2 import main as atlas_main
        atlas_main()
        print("Model training completed successfully!")
        print("Model saved to models/ directory")
        print("Training results saved to results/ directory")
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

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
  python main.py --dashboard       # Start dashboard
  python main.py --demo            # Run demo
  python main.py --test           # Performance test
  python main.py --data           # Prepare data
        """
    )
    
    parser.add_argument("--train", action="store_true", help="Train ATLAS model")
    parser.add_argument("--dashboard", action="store_true", help="Start real-time dashboard")
    parser.add_argument("--demo", action="store_true", help="Run system demo")
    parser.add_argument("--test", action="store_true", help="Run performance benchmark")
    parser.add_argument("--data", action="store_true", help="Download and prepare data")
    
    args = parser.parse_args()
    
    # Always show header
    print_header()
    
    # Execute based on arguments
    if args.train:
        run_training()
    elif args.dashboard:
        run_dashboard()
    elif args.demo:
        run_demo()
    elif args.test:
        run_performance_test()
    elif args.data:
        run_data_preparation()
    else:
        # Show interactive menu if no arguments
        print_menu()
        
        while True:
            try:
                choice = input("\nSelect option (1-6, or 'q' to quit): ").strip().lower()
                
                if choice in ['q', 'quit', 'exit']:
                    print("Thank you for using ATLAS!")
                    break
                elif choice == '1':
                    run_training()
                elif choice == '2':
                    run_dashboard()
                elif choice == '3':
                    run_demo()
                elif choice == '4':
                    run_performance_test()
                elif choice == '5':
                    run_data_preparation()
                elif choice == '6':
                    parser.print_help()
                else:
                    print("Invalid choice. Please enter 1-6 or 'q'")
                    
            except KeyboardInterrupt:
                print("\n\nThank you for using ATLAS!")
                break
            except Exception as e:
                print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
