"""
Diagnostic script to understand GPU vs CPU performance for cEBMF.

Run this to identify bottlenecks.
"""

import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from cebmf_torch import cEBMF


# Constants
SEPARATOR_WIDTH = 60
WARMUP_ITERATIONS = 2
DEFAULT_ITERATIONS = 10
DEFAULT_K = 5
TRANSFER_TEST_ITERATIONS = 100
SYNC_TEST_ITERATIONS = 1000
GPU_EFFICIENCY_THRESHOLD = 1_000_000


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    cpu_time: float
    gpu_time: Optional[float] = None
    mps_time: Optional[float] = None
    speedup: Optional[float] = None
    gpu_to_cpu_time: Optional[float] = None
    item_time: Optional[float] = None
    data_size_mb: Optional[float] = None


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*SEPARATOR_WIDTH}")
    print(title)
    print(f"{'='*SEPARATOR_WIDTH}")


def synchronize_device(device: str) -> None:
    """Synchronize device if CUDA."""
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark_device(data: torch.Tensor, X_cov: torch.Tensor, device: str, niter: int = DEFAULT_ITERATIONS, K: int = DEFAULT_K) -> float:
    """Benchmark cEBMF on a specific device.

    Args:
        data: Input data tensor
        device: Device to benchmark ("cpu", "cuda", or "mps")
        niter: Number of iterations to run
        K: Number of factors

    Returns:
        Elapsed time in seconds
    """
    print_section(f"Benchmarking on: {device.upper()}")

    # Warmup
    model = cEBMF(data=data, device=torch.device(device), K=K, prior_L="cgb", X_l=X_cov)
    model.initialise_factors()
    model.fit(WARMUP_ITERATIONS)

    # Actual timing
    synchronize_device(device)
    start = time.time()

    model = cEBMF(data=data, device=torch.device(device), K=K)
    model.initialise_factors()
    model.fit(niter)

    synchronize_device(device)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f} seconds")
    print(f"Time per iteration: {elapsed/niter:.3f} seconds")
    return elapsed


def measure_transfer_time(data: torch.Tensor, direction: str, iterations: int = TRANSFER_TEST_ITERATIONS) -> float:
    """Measure average transfer time for CPU↔GPU transfers.

    Args:
        data: Tensor to transfer
        direction: "cpu_to_gpu" or "gpu_to_cpu"
        iterations: Number of iterations to average

    Returns:
        Average transfer time in seconds
    """
    start = time.time()
    for _ in range(iterations):
        if direction == "cpu_to_gpu":
            _ = data.to("cuda")
        else:  # gpu_to_cpu
            _ = data.cpu()
        torch.cuda.synchronize()
    return (time.time() - start) / iterations


def check_data_transfer_overhead(data: torch.Tensor) -> Tuple[float, float]:
    """Check overhead of CPU↔GPU transfers.

    Args:
        data: Input data tensor

    Returns:
        Tuple of (gpu_to_cpu_time, data_size_mb)
    """
    print_section("Checking CPU↔GPU Transfer Overhead")

    cpu_to_gpu_time = measure_transfer_time(data, "cpu_to_gpu")
    data_gpu = data.to("cuda")
    gpu_to_cpu_time = measure_transfer_time(data_gpu, "gpu_to_cpu")

    data_size_mb = data.numel() * 4 / 1e6  # float32 = 4 bytes

    print(f"CPU→GPU transfer: {cpu_to_gpu_time*1000:.3f} ms")
    print(f"GPU→CPU transfer: {gpu_to_cpu_time*1000:.3f} ms")
    print(f"\nData size: {data_size_mb:.2f} MB (float32)")

    return gpu_to_cpu_time, data_size_mb


def check_synchronization_overhead() -> float:
    """Check overhead of .item() calls.

    Returns:
        Average .item() call time in seconds
    """
    print_section("Checking Synchronization Overhead (.item() calls)")

    x = torch.tensor(42.0, device="cuda")

    start = time.time()
    for _ in range(SYNC_TEST_ITERATIONS):
        _ = x.item()
        torch.cuda.synchronize()
    item_time = (time.time() - start) / SYNC_TEST_ITERATIONS

    print(f".item() call overhead: {item_time*1000:.3f} ms per call")
    print("Note: Each .item() forces GPU→CPU synchronization!")

    return item_time


def check_operation_sizes(data: torch.Tensor, K: int = DEFAULT_K) -> None:
    """Check if operations are large enough to benefit from GPU.

    Args:
        data: Input data tensor
        K: Number of factors
    """
    print_section("Operation Size Analysis")

    N, P = data.shape
    operations_l_f = N * P * K
    operations_mask = N * P
    operations_rk = N * P

    print(f"Data shape: ({N}, {P})")
    print(f"Number of factors K: {K}")
    print(f"\nKey operations:")
    print(f"  - L @ F.T: ({N}, {K}) @ ({K}, {P}) = {operations_l_f:,} operations")
    print(f"  - mask @ Fk2: ({N}, {P}) @ ({P},) = {operations_mask:,} operations")
    print(f"  - Rk @ Fk: ({N}, {P}) @ ({P},) = {operations_rk:,} operations")
    print(f"\nGPU efficiency threshold: ~{GPU_EFFICIENCY_THRESHOLD:,} operations")

    if operations_l_f < GPU_EFFICIENCY_THRESHOLD:
        print("⚠️  WARNING: Operations may be too small for GPU acceleration")
        print("   Consider using CPU for this dataset size")
    else:
        print("✓ Operations are large enough for GPU")


def print_speedup_results(speedup: float, device: str = "GPU") -> None:
    """Print speedup results with appropriate messaging.

    Args:
        speedup: Speedup ratio (CPU time / GPU time)
        device: Device name for display
    """
    print_section(f"{device} Speedup: {speedup:.2f}x")

    if speedup < 1.0:
        print(f"⚠️  {device} is SLOWER than CPU!")
        if device == "GPU":
            print("   This is likely due to:")
            print("   1. Dataset too small for GPU overhead")
            print("   2. Frequent CPU↔GPU synchronization")
            print("   3. Small matrix operations")
        else:  # MPS
            print("   MPS often has overhead for small operations")
    else:
        print(f"✓ {device} is {speedup:.2f}x faster")


def print_recommendations(
    speedup: Optional[float],
    niter: int,
    item_time: Optional[float],
    gpu_to_cpu_time: Optional[float],
    num_elements: int,
    data_size_mb: Optional[float]
) -> None:
    """Print recommendations based on benchmark results.

    Args:
        speedup: Speedup ratio if available
        niter: Number of iterations
        item_time: .item() call overhead time
        gpu_to_cpu_time: GPU→CPU transfer time
        num_elements: Total number of elements
        data_size_mb: Data size in MB
    """
    print_section("Recommendations")

    if speedup is not None and speedup < 1.0 and torch.cuda.is_available() and item_time is not None:
        print("⚠️  GPU is SLOWER than CPU for this dataset size!")
        print("\nRoot causes:")
        print(f"  1. .item() called every iteration (~{item_time*1000:.2f} ms overhead each)")
        print(f"     With {niter} iterations: ~{item_time*1000*niter:.1f} ms total sync overhead")
        if gpu_to_cpu_time is not None:
            print(f"  2. GPU→CPU transfer overhead: {gpu_to_cpu_time*1000:.2f} ms per transfer")
        print(f"  3. Dataset size ({num_elements:,} elements) is borderline for GPU efficiency")
        print("\nSolutions:")
        print("  A. Use CPU for this dataset size (RECOMMENDED)")
        print("  B. Optimize cEBMF code to reduce .item() calls:")
        print("     - Keep loss as tensor on GPU during fitting")
        print("     - Only convert to float when logging/checking convergence")
        print("     - Batch multiple .item() calls together")
        print("  C. Use larger datasets (> 50M elements) where GPU overhead is amortized")
    else:
        print("1. For small datasets (< 1M elements), CPU is often faster")
        print("2. Avoid frequent .item() calls - batch them")
        print("3. Minimize CPU↔GPU transfers")
        print("4. Use GPU only for datasets > 10M elements")
        print("5. Profile without torch.profiler first to check baseline")


def run_benchmark_for_dimension(N: int, P: int, niter: int = DEFAULT_ITERATIONS, K: int = DEFAULT_K) -> BenchmarkResult:
    """Run complete benchmark for a given dimension.

    Args:
        N: Number of observations
        P: Number of features
        niter: Number of iterations
        K: Number of factors

    Returns:
        BenchmarkResult with all timing information
    """
    print("GPU Performance Diagnostic for cEBMF")
    print("="*SEPARATOR_WIDTH)

    data = torch.randn(N, P)
    x = torch.randn(N)
    y = torch.randn(N)
    X_cov = torch.stack([x, y], dim=1)
    num_elements = N * P
    data_size_mb = num_elements * 4 / 1e6  # float32 = 4 bytes

    print(f"\nTest dataset: {N} × {P} = {num_elements:,} elements")
    print(f"Memory: {data_size_mb:.2f} MB (float32)")

    check_operation_sizes(data, K=K)

    # Initialize result
    result = BenchmarkResult(cpu_time=0.0, data_size_mb=data_size_mb)

    # Check transfer overhead if CUDA available
    if torch.cuda.is_available():
        result.gpu_to_cpu_time, _ = check_data_transfer_overhead(data)
        result.item_time = check_synchronization_overhead()

    # Benchmark CPU
    result.cpu_time = benchmark_device(data, X_cov, "cpu", niter=niter, K=K)

    # Benchmark GPU/MPS if available
    if torch.cuda.is_available():
        result.gpu_time = benchmark_device(data, X_cov, "cuda", niter=niter, K=K)
        result.speedup = result.cpu_time / result.gpu_time
        print_speedup_results(result.speedup, "GPU")
    elif torch.backends.mps.is_available():
        result.mps_time = benchmark_device(data, X_cov, "mps", niter=niter, K=K)
        result.speedup = result.cpu_time / result.mps_time
        print_speedup_results(result.speedup, "MPS")

    print_recommendations(
        result.speedup, niter, result.item_time, result.gpu_to_cpu_time,
        num_elements, result.data_size_mb
    )

    return result


def main():
    """Run all diagnostics."""
    if len(sys.argv) < 2:
        print("Usage: python gpu_performance_diagnostic.py <output_directory>")
        sys.exit(1)

    out_dir = pathlib.Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    dimensions = [
        # (500, 200),        # 100K elements
        # (1000, 500),       # 500K elements
        # (1600, 625),       # 1M elements
        # (5000, 2000),     # 10M elements
        (16000, 6250),     # 100M elements
        # (32000, 16000),    # 500M elements
    ]

    json_data = {}
    for N, P in dimensions:
        result = run_benchmark_for_dimension(N, P)

        json_data[N * P] = {
            "number_of_elements": N * P,
            "data_size_mb": result.data_size_mb,
            "gpu_to_cpu_time_ms": result.gpu_to_cpu_time * 1000 if result.gpu_to_cpu_time else None,
            "item_time_ms": result.item_time * 1000 if result.item_time else None,
            "cpu_time": result.cpu_time,
            "gpu_time": result.gpu_time,
            "mps_time": result.mps_time,
            "speedup": result.speedup,
        }

    output_file = out_dir / "gpu_performance_diagnostic.json"
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

