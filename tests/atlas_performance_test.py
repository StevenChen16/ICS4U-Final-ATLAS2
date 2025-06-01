"""
ATLAS Model Performance Benchmark
测试模型的TFlops性能，包括理论和实际性能
"""

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import gc
import psutil
import GPUtil
from contextlib import contextmanager
import warnings
import sys
warnings.filterwarnings("ignore")

# 导入模型架构（需要从atlas2.py导入）
from src.atlas2 import ATLASModel, create_specialized_kernels
print("✅ Successfully imported ATLAS model")

class PerformanceBenchmark:
    """ATLAS模型性能基准测试类"""
    
    def __init__(self, model_path: str = "../models/atlas_binary_model_best.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model = None
        self.compiled_model = None
        
        print(f"🚀 Initializing Performance Benchmark")
        print(f"📱 Device: {self.device}")
        print(f"🔧 PyTorch Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    def load_model(self) -> None:
        """加载模型"""
        try:
            # 创建模型实例
            kernels = create_specialized_kernels()
            self.model = ATLASModel(
                input_shape=(50, 50, 4),
                kernels=kernels,
                dropout_rate=0.5
            ).to(self.device)
            
            # 加载权重
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # 编译模型（PyTorch 2.0+）
            # if hasattr(torch, 'compile'):
            if False:
                print("🔧 Compiling model with torch.compile()...")
                self.compiled_model = torch.compile(self.model, mode='max-autotune')
            else:
                print("⚠️  torch.compile not available, using standard model")
                self.compiled_model = self.model
                
            print("✅ Model loaded successfully")
            self._print_model_info()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _print_model_info(self) -> None:
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Information:")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Model Size: {self._get_model_size_mb():.2f} MB")

    def _get_model_size_mb(self) -> float:
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        model_size = (param_size + buffer_size) / 1024 / 1024
        return model_size

    def calculate_theoretical_flops(self, input_shape: Tuple[int, int, int, int]) -> int:
        """计算理论FLOPs"""
        try:
            # 尝试使用thop库
            from thop import profile
            dummy_input = torch.randn(input_shape).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
            return int(flops)
        except ImportError:
            # 手动估算FLOPs
            batch_size, channels, height, width = input_shape
            
            # 估算各个组件的FLOPs
            # Conv2d: output_elements * (kernel_size^2 * input_channels + 1)
            # Linear: input_features * output_features
            
            estimated_flops = 0
            
            # 4个分支，每个分支两个卷积层
            for _ in range(4):
                # 第一个卷积层：假设输出32个通道，5x5卷积核
                conv1_flops = (height * width) * (5 * 5 * 1 + 1) * 9  # 平均9个核
                # 池化后尺寸减半
                height_pool = height // 2
                width_pool = width // 2
                # 第二个卷积层：32个输出通道，3x3卷积核
                conv2_flops = (height_pool * width_pool) * (3 * 3 * 9 + 1) * 32
                estimated_flops += conv1_flops + conv2_flops
            
            # 分类器层
            classifier_flops = (128 * 64) + (64 * 32) + (32 * 1)
            estimated_flops += classifier_flops
            
            # 乘以batch size
            estimated_flops *= batch_size
            
            print(f"📊 Using estimated FLOPs calculation: {estimated_flops:,}")
            return estimated_flops

    @contextmanager
    def timer(self):
        """计时器上下文管理器"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        self.elapsed_time = end - start

    def warmup(self, input_shape: Tuple[int, int, int, int], warmup_iterations: int = 10) -> None:
        """模型预热"""
        print(f"🔥 Warming up model ({warmup_iterations} iterations)...")
        
        with torch.no_grad():
            for _ in range(warmup_iterations):
                dummy_input = torch.randn(input_shape).to(self.device)
                _ = self.compiled_model(dummy_input)
                
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ Warmup completed")

    def benchmark_inference(
        self, 
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        input_size: Tuple[int, int] = (50, 50),
        num_iterations: int = 100
    ) -> Dict[str, List[float]]:
        """推理性能基准测试"""
        print(f"\n🏃‍♂️ Starting inference benchmark...")
        print(f"   Batch sizes: {batch_sizes}")
        print(f"   Input size: {input_size}")
        print(f"   Iterations per batch: {num_iterations}")
        
        results = {
            'batch_sizes': [],
            'avg_latency_ms': [],
            'throughput_samples_per_sec': [],
            'theoretical_tflops': [],
            'actual_tflops': [],
            'gpu_utilization': [],
            'memory_usage_mb': []
        }
        
        for batch_size in batch_sizes:
            print(f"\n📊 Testing batch size: {batch_size}")
            
            input_shape = (batch_size, 4, input_size[0], input_size[1])
            
            # 预热
            self.warmup(input_shape, warmup_iterations=5)
            
            # 计算理论FLOPs
            theoretical_flops = self.calculate_theoretical_flops(input_shape)
            
            # 性能测试
            latencies = []
            
            with torch.no_grad():
                for i in range(num_iterations):
                    # 生成随机输入
                    dummy_input = torch.randn(input_shape).to(self.device)
                    
                    # 记录GPU使用情况（如果可用）
                    if torch.cuda.is_available() and i == 0:
                        gpu_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    
                    # 计时推理
                    with self.timer():
                        outputs = self.compiled_model(dummy_input)
                    
                    latencies.append(self.elapsed_time * 1000)  # Convert to ms
                    
                    if i == 0 and torch.cuda.is_available():
                        gpu_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        memory_usage = gpu_after - gpu_before
                    
                    # 进度显示
                    if (i + 1) % 20 == 0:
                        print(f"   Progress: {i + 1}/{num_iterations}")
            
            # 计算统计数据
            avg_latency = np.mean(latencies[10:])  # 忽略前10次以避免初始化开销
            std_latency = np.std(latencies[10:])
            
            throughput = batch_size / (avg_latency / 1000)  # samples per second
            actual_tflops = (theoretical_flops / (avg_latency / 1000)) / 1e12
            
            # GPU使用率（如果可用）
            gpu_util = 0
            if torch.cuda.is_available():
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_util = gpus[0].load * 100
                except:
                    gpu_util = 0
            
            # 存储结果
            results['batch_sizes'].append(batch_size)
            results['avg_latency_ms'].append(avg_latency)
            results['throughput_samples_per_sec'].append(throughput)
            results['theoretical_tflops'].append(theoretical_flops / 1e12)
            results['actual_tflops'].append(actual_tflops)
            results['gpu_utilization'].append(gpu_util)
            results['memory_usage_mb'].append(memory_usage if torch.cuda.is_available() else 0)
            
            # 打印结果
            print(f"   ⏱️  Avg Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
            print(f"   🚀 Throughput: {throughput:.1f} samples/sec")
            print(f"   ⚡ TFlops: {actual_tflops:.3f}")
            print(f"   💾 Memory: {memory_usage if torch.cuda.is_available() else 0:.1f} MB")
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return results

    def compare_compiled_vs_standard(
        self, 
        batch_size: int = 16, 
        num_iterations: int = 50
    ) -> Dict[str, float]:
        """比较编译模型vs标准模型性能"""
        print(f"\n⚔️  Comparing compiled vs standard model...")
        
        input_shape = (batch_size, 4, 50, 50)
        
        # 测试标准模型
        print("🔧 Testing standard model...")
        self.model.eval()
        latencies_standard = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                dummy_input = torch.randn(input_shape).to(self.device)
                with self.timer():
                    _ = self.model(dummy_input)
                latencies_standard.append(self.elapsed_time * 1000)
        
        # 测试编译模型
        print("⚡ Testing compiled model...")
        latencies_compiled = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                dummy_input = torch.randn(input_shape).to(self.device)
                with self.timer():
                    _ = self.compiled_model(dummy_input)
                latencies_compiled.append(self.elapsed_time * 1000)
        
        avg_standard = np.mean(latencies_standard[5:])  # 忽略前5次
        avg_compiled = np.mean(latencies_compiled[5:])
        speedup = avg_standard / avg_compiled
        
        results = {
            'standard_latency_ms': avg_standard,
            'compiled_latency_ms': avg_compiled,
            'speedup': speedup
        }
        
        print(f"   📊 Standard Model: {avg_standard:.2f} ms")
        print(f"   ⚡ Compiled Model: {avg_compiled:.2f} ms")
        print(f"   🚀 Speedup: {speedup:.2f}x")
        
        return results

    def create_performance_report(self, results: Dict, comparison: Dict) -> None:
        """生成性能报告可视化"""
        print("\n📈 Generating performance report...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ATLAS Model Performance Benchmark Report', fontsize=16, fontweight='bold')
        
        # 1. 延迟 vs 批次大小
        axes[0, 0].plot(results['batch_sizes'], results['avg_latency_ms'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Average Latency (ms)')
        axes[0, 0].set_title('Latency vs Batch Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 吞吐量 vs 批次大小
        axes[0, 1].plot(results['batch_sizes'], results['throughput_samples_per_sec'], 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Throughput (samples/sec)')
        axes[0, 1].set_title('Throughput vs Batch Size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. TFlops vs 批次大小
        axes[0, 2].plot(results['batch_sizes'], results['actual_tflops'], 'ro-', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Batch Size')
        axes[0, 2].set_ylabel('TFlops')
        axes[0, 2].set_title('Computational Performance (TFlops)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 内存使用
        if torch.cuda.is_available():
            axes[1, 0].bar(results['batch_sizes'], results['memory_usage_mb'], color='purple', alpha=0.7)
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('Memory Usage (MB)')
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'GPU not available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('GPU Memory Usage')
        
        # 5. 编译对比
        comparison_data = [comparison['standard_latency_ms'], comparison['compiled_latency_ms']]
        comparison_labels = ['Standard', 'Compiled']
        bars = axes[1, 1].bar(comparison_labels, comparison_data, color=['lightblue', 'lightgreen'])
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_title(f'Standard vs Compiled\n(Speedup: {comparison["speedup"]:.2f}x)')
        
        # 添加数值标签
        for bar, value in zip(bars, comparison_data):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{value:.1f}ms', ha='center', va='bottom')
        
        # 6. 效率分析（理论vs实际TFlops）
        efficiency = [actual/theoretical*100 for actual, theoretical in 
                     zip(results['actual_tflops'], results['theoretical_tflops'])]
        axes[1, 2].plot(results['batch_sizes'], efficiency, 'mo-', linewidth=2, markersize=8)
        axes[1, 2].set_xlabel('Batch Size')
        axes[1, 2].set_ylabel('Efficiency (%)')
        axes[1, 2].set_title('Computational Efficiency\n(Actual/Theoretical TFlops)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('atlas_performance_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Performance report saved as 'atlas_performance_report.png'")

    def print_summary_report(self, results: Dict, comparison: Dict) -> None:
        """打印性能总结报告"""
        print("\n" + "="*80)
        print("🏆 ATLAS MODEL PERFORMANCE SUMMARY REPORT")
        print("="*80)
        
        print(f"\n📊 BASIC INFORMATION:")
        print(f"   🔧 Device: {self.device}")
        print(f"   📱 Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   💾 Model Size: {self._get_model_size_mb():.2f} MB")
        
        print(f"\n⚡ PERFORMANCE HIGHLIGHTS:")
        max_tflops_idx = np.argmax(results['actual_tflops'])
        max_throughput_idx = np.argmax(results['throughput_samples_per_sec'])
        min_latency_idx = np.argmin(results['avg_latency_ms'])
        
        print(f"   🚀 Peak TFlops: {results['actual_tflops'][max_tflops_idx]:.3f} (batch size {results['batch_sizes'][max_tflops_idx]})")
        print(f"   🏃 Max Throughput: {results['throughput_samples_per_sec'][max_throughput_idx]:.1f} samples/sec (batch size {results['batch_sizes'][max_throughput_idx]})")
        print(f"   ⏱️  Min Latency: {results['avg_latency_ms'][min_latency_idx]:.2f} ms (batch size {results['batch_sizes'][min_latency_idx]})")
        
        print(f"\n🔧 TORCH.COMPILE BENEFITS:")
        print(f"   ⚡ Speedup: {comparison['speedup']:.2f}x")
        print(f"   📉 Latency Reduction: {((comparison['standard_latency_ms'] - comparison['compiled_latency_ms']) / comparison['standard_latency_ms'] * 100):.1f}%")
        
        print(f"\n📈 SCALABILITY ANALYSIS:")
        efficiency_range = [min(results['actual_tflops']), max(results['actual_tflops'])]
        print(f"   📊 TFlops Range: {efficiency_range[0]:.3f} - {efficiency_range[1]:.3f}")
        print(f"   📈 Batch Scaling: {results['throughput_samples_per_sec'][-1]/results['throughput_samples_per_sec'][0]:.1f}x improvement (BS1→BS{results['batch_sizes'][-1]})")
        
        # 推荐的使用配置
        balanced_idx = len(results['batch_sizes']) // 2  # 中等batch size通常最平衡
        print(f"\n💡 RECOMMENDED CONFIGURATION:")
        print(f"   🎯 Optimal Batch Size: {results['batch_sizes'][max_tflops_idx]} (for max TFlops)")
        print(f"   ⚖️  Balanced Configuration: Batch Size {results['batch_sizes'][balanced_idx]} ")
        print(f"      - Latency: {results['avg_latency_ms'][balanced_idx]:.2f} ms")
        print(f"      - Throughput: {results['throughput_samples_per_sec'][balanced_idx]:.1f} samples/sec")
        print(f"      - TFlops: {results['actual_tflops'][balanced_idx]:.3f}")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    print("🚀 ATLAS Model Performance Benchmark")
    print("="*50)
    
    # 初始化基准测试
    benchmark = PerformanceBenchmark("models/atlas_binary_model_best.pth")
    
    # 加载模型
    benchmark.load_model()
    
    # 运行基准测试
    results = benchmark.benchmark_inference(
        batch_sizes=[1, 2, 4, 8, 16, 32],
        input_size=(50, 50),
        num_iterations=100
    )
    
    # 比较编译vs标准模型
    comparison = benchmark.compare_compiled_vs_standard(batch_size=16, num_iterations=50)
    
    # 生成报告
    benchmark.create_performance_report(results, comparison)
    benchmark.print_summary_report(results, comparison)
    
    print("\n🎉 Benchmark completed successfully!")


if __name__ == "__main__":
    main()
