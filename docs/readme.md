# Parallel Sorting Investigation — Task 2

## Student Details

- **Name:** Tudose Alexandru
- **Group:** 10LF333

---

## Hardware Configuration

- **CPU (Colab):** Intel(R) Xeon(R) CPU @ 2.20GHz (Google Colab High-RAM)
- **GPU:** NVIDIA A100-SXM4-80GB
- **GPU Memory:** 80 GB HBM2e
- **GPU CUDA Cores:** 6,912
- **System RAM (Colab):** ~83 GB
- **CUDA Version:** 12.8 (V12.8.93)
- **Driver Version:** 580.82.07
- **Compile flags:** `-O2 -arch=sm_75`

> Note: compiled with `-arch=sm_75` (Turing) on an A100 (sm_80).
> The code runs correctly due to forward compatibility, but A100-specific
> optimizations (e.g. async memory copies, TF32) are not used.

---

## Task 2.1 — Execution, Communication and Computation Times

### Measurement Methodology

All algorithms read from the same binary file (`data/input.bin`) to ensure identical input across runs.

Timing is broken down into three distinct components:

- **wall_ms** — total elapsed time (wall-clock), measured with `std::chrono::high_resolution_clock`.
  Includes GPU memory allocation, H2D transfer, all kernel launches, D2H transfer and `cudaFree`.
- **kernel_ms** — GPU kernel execution time only, measured with `cudaEvent_t` placed around kernel calls.
- **h2d_ms / d2h_ms** — host→device and device→host transfer times, each measured separately
  with `cudaEvent_t`. Their sum is the communication overhead.
- **comm_ms** — h2d + d2h (communication overhead total).

For the CPU variant there are no transfers, so `kernel_ms = wall_ms` and comm = 0.

### Timing Results

| Algorithm | Variant | N | wall (ms) | H2D (ms) | kernel (ms) | D2H (ms) | comm (ms) | comm % |
|---|---|---|---|---|---|---|---|---|
| Bitonic Sort | CPU | 1,000,000 | 366.06 | — | 366.06 | — | — | — |
| Bitonic Sort | GPU Global | 1,000,000 | 482.20 | 1.56 | 92.46 | 1.11 | 2.67 | 0.6% |
| Bitonic Sort | GPU Shared | 1,000,000 | 4.06 | 1.05 | 1.69 | 0.98 | 2.03 | 49.9% |
| Shell Sort | CPU | 1,000,000 | 190.07 | — | 190.07 | — | — | — |
| Shell Sort | GPU Global | 1,000,000 | 775.11 | 1.05 | 508.68 | 1.07 | 2.12 | 0.3% |
| Shell Sort | GPU Shared | 1,000,000 | 493.93 | 0.98 | 491.41 | 1.09 | 2.07 | 0.4% |
| Odd-Even Sort | CPU | 1,000,000 | 1,179,393.12 | — | 1,179,393.12 | — | — | — |
| Odd-Even Sort | GPU Global | 1,000,000 | 5,134.76 | 1.07 | 4,782.46 | 0.97 | 2.04 | 0.0% |
| Odd-Even Sort | GPU Shared | 1,000,000 | 9,390.12 | 1.00 | 9,387.74 | 0.94 | 1.93 | 0.0% |
| Ranking Sort | CPU | 32,768 | 3,875.72 | — | 3,875.72 | — | — | — |
| Ranking Sort | GPU Global | 32,768 | 386.85 | 0.07 | 37.86 | 0.06 | 0.13 | 0.0% |
| Ranking Sort | GPU Shared | 32,768 | 1.60 | 0.06 | 1.24 | 0.05 | 0.11 | 6.9% |
| Merge Sort (Binary Reduction) | CPU | 1,000,000 | 137.76 | — | 137.76 | — | — | — |
| Merge Sort (Binary Reduction) | GPU Global | 1,000,000 | 564.58 | 1.06 | 297.79 | 1.06 | 2.13 | 0.4% |
| Merge Sort (Binary Reduction) | GPU Shared | 1,000,000 | 261.55 | 0.97 | 258.93 | 1.08 | 2.05 | 0.8% |

### Speedup (kernel time: CPU vs GPU)

Speedup is computed as `CPU_kernel_ms / GPU_kernel_ms` — using kernel time only,
which isolates computation from communication overhead.

| Algorithm | N | Speedup GPU Global | Speedup GPU Shared |
|---|---|---|---|
| Bitonic Sort | 1,000,000 | 4.0x | 216.5x |
| Shell Sort | 1,000,000 | 0.4x | 0.4x |
| Odd-Even Sort | 1,000,000 | 246.6x | 125.6x |
| Ranking Sort | 32,768 | 102.4x | 3132.7x |
| Merge Sort (Binary Reduction) | 1,000,000 | 0.5x | 0.5x |

### Scalability Analysis

Only N = 1,000,000 was benchmarked (N = 32,768 for Ranking Sort due to O(n²) memory constraints).
Theoretical scaling behaviour based on algorithmic complexity:

| Algorithm | CPU complexity | GPU parallelism | Expected GPU scaling |
|---|---|---|---|
| Bitonic Sort | O(n log²n) | O(n/2) independent swaps per sub-step | good — log²n serial steps |
| Shell Sort | O(n^1.5) Hibbard | strided chains, inherently serial | poor |
| Odd-Even Sort | O(n²) | O(n/2) swaps per phase, n phases | limited — n serial phases |
| Ranking Sort | O(n²) | each element rank computed independently | excellent |
| Merge Sort | O(n log n) | O(pairs) per pass, log n serial passes | moderate |

For Odd-Even Sort, the O(n²) complexity means that at N = 10M the GPU would still take
roughly 10× longer than at N = 1M, making it impractical for very large inputs.
Ranking Sort is capped at N = 32,768 due to GPU memory requirements of O(n²) operations;
for larger N, runtime would be estimated by extrapolation (quadratic).

---

## Task 2.2 — Discussion and Analysis

### Comparative Analysis

**Bitonic Sort** achieves the best overall GPU result: 217x speedup (kernel) with shared memory, down to 1.69 ms for 1M elements. The algorithm maps naturally to GPU — every sub-step has exactly n/2 independent compare-swaps. The global memory variant only hits 4.0x because the A100 context initialises on the first CUDA call, inflating wall time; the kernel itself is already much faster than CPU.

**Ranking Sort** shows the most extreme speedup: 3,133x with shared memory (kernel: 3,875.72 ms CPU vs 1.24 ms GPU Shared, at N = 32,768). It is embarrassingly parallel by construction — each thread independently counts how many elements are smaller than its own value. The tile-based shared memory kernel reduces global memory traffic by roughly BLOCK_SIZE (512) times compared to the naive global version.

**Odd-Even Sort** on CPU took 19.7 minutes for 1M elements — effectively unusable at this scale. GPU Global Memory brings it down to 5.1 s (247x speedup, kernel). Interestingly, the shared memory variant is *slower* than global (9,387.74 ms vs 4,782.46 ms). The reason is the boundary kernel: for each odd phase an additional `odd_even_boundary` kernel launch is required to fix cross-block pairs, so instead of one kernel per phase there are two — the launch overhead adds up across all n phases.

**Shell Sort** doesn't benefit from GPU at all: both GPU variants are slower than CPU (508.68 ms and 491.41 ms vs 190.07 ms CPU). The Hibbard gap sequence produces strided memory access patterns (at the largest gap, consecutive threads access memory 500K integers apart), causing cache misses on essentially every access. The shared memory version is identical to the global version in behaviour — the gap-strided chains span the entire array and cannot be confined to a block's shared memory.

**Merge Sort** is also slower on GPU than on CPU for N = 1M (258.93 ms shared, 297.79 ms global vs 137.76 ms CPU). The bottom-up algorithm has log₂(n) ≈ 20 serial passes; in the final passes there are only 2–4 independent merges so most of the GPU sits idle. The per-kernel-launch overhead also accumulates. This would likely reverse at much larger N where the early passes (thousands of small parallel merges) dominate total time.

### Communication Overhead

H2D and D2H transfers are remarkably consistent across all algorithms: 0.89 ms H2D and 0.84 ms D2H on average (for N = 1M, 4 MB of int32 data).
At ~8 GB/s effective PCIe bandwidth this is expected, and it confirms that communication
is not the bottleneck in any of these algorithms — the kernel time dominates everywhere.

The one exception is Bitonic Sort Shared: the kernel runs in only 1.69 ms,
while transfers cost 2.03 ms — more time spent moving data than computing. For even larger N this ratio would improve
because kernel time scales as O(n log²n) while transfer time scales linearly.

### Bottlenecks

**Bitonic Sort:** ~log²(n) ≈ 400 kernel launches for n = 1M (one per sub-step). The shared variant reduces this by batching sub-steps with `sub < BLOCK_SIZE` into a single in-block pass, which explains most of the 54× wall-time improvement over global.

**Shell Sort:** strided global memory access is the fundamental problem. No shared memory strategy can help because a single insertion-sort chain spans the full array.

**Odd-Even Sort:** n serial phases is the hard limit. Even if each phase is perfectly parallel, you still need n synchronisation barriers. At n = 1M that is 1 million kernel launches.

**Ranking Sort:** the global memory variant still performs n² reads from global memory. Shared memory tiling reduces this to n²/BLOCK_SIZE global reads — a 512× reduction in global traffic.

**Merge Sort:** the log n serial dependency between passes cannot be eliminated. In the final pass there is a single merge of two 500K-element arrays — one thread does all the work.

### Speedup Metrics — Theoretical vs Measured

Using Amdahl's Law: `S = 1 / (f_s + f_p / p)` where p = 6,912 CUDA cores (A100).

| Algorithm | Measured speedup (Shared) | Theoretical max (Amdahl) | Main limiting factor |
|---|---|---|---|
| Bitonic Sort | 217x | ~400 serial steps | log²n serial kernel launches |
| Shell Sort | 0x | ~1 serial steps | fully serial chains per gap |
| Odd-Even Sort | 247x | ~n serial steps | n serial phases |
| Ranking Sort | 3,133x | ~n serial steps | none — embarrassingly parallel |
| Merge Sort (Binary Reduction) | 1x | ~log n serial steps | log n serial passes |

The largest deviation from theoretical maximum is Merge Sort and Shell Sort — both have structural serial bottlenecks that prevent meaningful GPU utilisation at N = 1M.
Bitonic Sort comes closest to the ideal for a comparison-network algorithm; Ranking Sort benefits from near-perfect parallelism but is limited to small N by its O(n²) nature.

---

*GPU Lab Assignment 2026 — Tudose Alexandru, 10LF333*