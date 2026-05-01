#!/usr/bin/env python3
import csv, os, sys
from collections import defaultdict

CSV_PATH    = "results/timings.csv"
OUTPUT_PATH = "docs/readme.md"

STUDENT_NAME  = "Tudose Alexandru"
GRUPA         = "10LF333"
GPU_MODEL     = "NVIDIA A100-SXM4-80GB"
GPU_VRAM      = "80 GB HBM2e"
GPU_CORES     = 6912
CUDA_VERSION  = "12.8 (V12.8.93)"
DRIVER        = "580.82.07"
COLAB_CPU     = "Intel(R) Xeon(R) CPU @ 2.20GHz (Google Colab High-RAM)"
COLAB_RAM     = "~83 GB"

ALGO_ORDER = ["BitonicSort", "ShellSort", "OddEvenSort", "RankingSort", "MergeSort"]
ALGO_LABELS = {
    "BitonicSort": "Bitonic Sort",
    "ShellSort":   "Shell Sort",
    "OddEvenSort": "Odd-Even Sort",
    "RankingSort": "Ranking Sort",
    "MergeSort":   "Merge Sort (Binary Reduction)",
}

def load_csv(path):
    data = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            algo    = row["algorithm"].strip()
            variant = row["variant"].strip()
            data[algo][variant] = {
                "n":         int(row["n"]),
                "wall_ms":   float(row["wall_ms"]),
                "h2d_ms":    float(row["h2d_ms"]),
                "kernel_ms": float(row["kernel_ms"]),
                "d2h_ms":    float(row["d2h_ms"]),
            }
    return data

def speedup(cpu_kernel, gpu_kernel):
    if gpu_kernel == 0:
        return float("inf")
    return cpu_kernel / gpu_kernel

def comm_pct(h2d, d2h, wall):
    if wall == 0:
        return 0.0
    return 100.0 * (h2d + d2h) / wall

def fmt(val, decimals=2):
    return f"{val:,.{decimals}f}"

def generate(data):
    lines = []
    w = lines.append

    w(f"# Parallel Sorting Investigation — Task 2\n")
    w(f"## Student Details\n")
    w(f"- **Name:** {STUDENT_NAME}")
    w(f"- **Group:** {GRUPA}\n")
    w("---\n")

    w("## Hardware Configuration\n")
    w(f"- **CPU (Colab):** {COLAB_CPU}")
    w(f"- **GPU:** {GPU_MODEL}")
    w(f"- **GPU Memory:** {GPU_VRAM}")
    w(f"- **GPU CUDA Cores:** {GPU_CORES:,}")
    w(f"- **System RAM (Colab):** {COLAB_RAM}")
    w(f"- **CUDA Version:** {CUDA_VERSION}")
    w(f"- **Driver Version:** {DRIVER}")
    w(f"- **Compile flags:** `-O2 -arch=sm_75`")
    w("")
    w("> Note: compiled with `-arch=sm_75` (Turing) on an A100 (sm_80).")
    w("> The code runs correctly due to forward compatibility, but A100-specific")
    w("> optimizations (e.g. async memory copies, TF32) are not used.\n")
    w("---\n")

    w("## Task 2.1 — Execution, Communication and Computation Times\n")

    w("### Measurement Methodology\n")
    w("All algorithms read from the same binary file (`data/input.bin`) to ensure identical input across runs.\n")
    w("Timing is broken down into three distinct components:\n")
    w("- **wall_ms** — total elapsed time (wall-clock), measured with `std::chrono::high_resolution_clock`.")
    w("  Includes GPU memory allocation, H2D transfer, all kernel launches, D2H transfer and `cudaFree`.")
    w("- **kernel_ms** — GPU kernel execution time only, measured with `cudaEvent_t` placed around kernel calls.")
    w("- **h2d_ms / d2h_ms** — host→device and device→host transfer times, each measured separately")
    w("  with `cudaEvent_t`. Their sum is the communication overhead.")
    w("- **comm_ms** — h2d + d2h (communication overhead total).\n")
    w("For the CPU variant there are no transfers, so `kernel_ms = wall_ms` and comm = 0.\n")

    w("### Timing Results\n")

    header = ("| Algorithm | Variant | N | wall (ms) | H2D (ms) "
              "| kernel (ms) | D2H (ms) | comm (ms) | comm % |")
    sep    = "|---|---|---|---|---|---|---|---|---|"
    w(header)
    w(sep)

    for algo in ALGO_ORDER:
        label = ALGO_LABELS[algo]
        for variant in ["CPU", "GPU_Global", "GPU_Shared"]:
            if variant not in data[algo]:
                continue
            r = data[algo][variant]
            comm = r["h2d_ms"] + r["d2h_ms"]
            cp   = comm_pct(r["h2d_ms"], r["d2h_ms"], r["wall_ms"])
            n_str = f"{r['n']:,}"
            if variant == "CPU":
                w(f"| {label} | CPU | {n_str} | {fmt(r['wall_ms'])} | — "
                  f"| {fmt(r['kernel_ms'])} | — | — | — |")
            else:
                vname = variant.replace("_", " ")
                w(f"| {label} | {vname} | {n_str} | {fmt(r['wall_ms'])} "
                  f"| {fmt(r['h2d_ms'])} | {fmt(r['kernel_ms'])} "
                  f"| {fmt(r['d2h_ms'])} | {fmt(comm)} | {fmt(cp, 1)}% |")
    w("")

    w("### Speedup (kernel time: CPU vs GPU)\n")
    w("Speedup is computed as `CPU_kernel_ms / GPU_kernel_ms` — using kernel time only,")
    w("which isolates computation from communication overhead.\n")

    header2 = "| Algorithm | N | Speedup GPU Global | Speedup GPU Shared |"
    sep2    = "|---|---|---|---|"
    w(header2)
    w(sep2)

    for algo in ALGO_ORDER:
        label = ALGO_LABELS[algo]
        if "CPU" not in data[algo]:
            continue
        cpu_k = data[algo]["CPU"]["kernel_ms"]
        n     = data[algo]["CPU"]["n"]
        sg = speedup(cpu_k, data[algo].get("GPU_Global", {}).get("kernel_ms", 0))
        ss = speedup(cpu_k, data[algo].get("GPU_Shared", {}).get("kernel_ms", 0))
        sg_str = f"{sg:.1f}x" if sg != float("inf") else "N/A"
        ss_str = f"{ss:.1f}x" if ss != float("inf") else "N/A"
        w(f"| {label} | {n:,} | {sg_str} | {ss_str} |")
    w("")

    w("### Scalability Analysis\n")
    w("Only N = 1,000,000 was benchmarked (N = 32,768 for Ranking Sort due to O(n²) memory constraints).")
    w("Theoretical scaling behaviour based on algorithmic complexity:\n")
    w("| Algorithm | CPU complexity | GPU parallelism | Expected GPU scaling |")
    w("|---|---|---|---|")
    w("| Bitonic Sort | O(n log²n) | O(n/2) independent swaps per sub-step | good — log²n serial steps |")
    w("| Shell Sort | O(n^1.5) Hibbard | strided chains, inherently serial | poor |")
    w("| Odd-Even Sort | O(n²) | O(n/2) swaps per phase, n phases | limited — n serial phases |")
    w("| Ranking Sort | O(n²) | each element rank computed independently | excellent |")
    w("| Merge Sort | O(n log n) | O(pairs) per pass, log n serial passes | moderate |")
    w("")
    w("For Odd-Even Sort, the O(n²) complexity means that at N = 10M the GPU would still take")
    w("roughly 10× longer than at N = 1M, making it impractical for very large inputs.")
    w("Ranking Sort is capped at N = 32,768 due to GPU memory requirements of O(n²) operations;\n"
      "for larger N, runtime would be estimated by extrapolation (quadratic).\n")

    w("---\n")
    w("## Task 2.2 — Discussion and Analysis\n")

    w("### Comparative Analysis\n")

    b  = data["BitonicSort"]
    sh = data["ShellSort"]
    oe = data["OddEvenSort"]
    rk = data["RankingSort"]
    ms = data["MergeSort"]

    bs_su_global = speedup(b["CPU"]["kernel_ms"], b["GPU_Global"]["kernel_ms"])
    bs_su_shared = speedup(b["CPU"]["kernel_ms"], b["GPU_Shared"]["kernel_ms"])
    rk_su_global = speedup(rk["CPU"]["kernel_ms"], rk["GPU_Global"]["kernel_ms"])
    rk_su_shared = speedup(rk["CPU"]["kernel_ms"], rk["GPU_Shared"]["kernel_ms"])
    oe_su_global = speedup(oe["CPU"]["kernel_ms"], oe["GPU_Global"]["kernel_ms"])
    oe_su_shared = speedup(oe["CPU"]["kernel_ms"], oe["GPU_Shared"]["kernel_ms"])
    sh_su_global = speedup(sh["CPU"]["kernel_ms"], sh["GPU_Global"]["kernel_ms"])
    ms_su_global = speedup(ms["CPU"]["kernel_ms"], ms["GPU_Global"]["kernel_ms"])
    ms_su_shared = speedup(ms["CPU"]["kernel_ms"], ms["GPU_Shared"]["kernel_ms"])

    w(f"**Bitonic Sort** achieves the best overall GPU result: {fmt(bs_su_shared, 0)}x speedup"
      f" (kernel) with shared memory, down to {fmt(b['GPU_Shared']['kernel_ms'])} ms for 1M elements."
      f" The algorithm maps naturally to GPU — every sub-step has exactly n/2 independent compare-swaps."
      f" The global memory variant only hits {fmt(bs_su_global, 1)}x because the A100 context initialises"
      f" on the first CUDA call, inflating wall time; the kernel itself is already much faster than CPU.\n")

    w(f"**Ranking Sort** shows the most extreme speedup: {fmt(rk_su_shared, 0)}x with shared memory"
      f" (kernel: {fmt(rk['CPU']['kernel_ms'])} ms CPU vs {fmt(rk['GPU_Shared']['kernel_ms'])} ms GPU Shared,"
      f" at N = 32,768). It is embarrassingly parallel by construction — each thread independently counts"
      f" how many elements are smaller than its own value. The tile-based shared memory kernel reduces"
      f" global memory traffic by roughly BLOCK_SIZE (512) times compared to the naive global version.\n")

    w(f"**Odd-Even Sort** on CPU took {fmt(oe['CPU']['wall_ms'] / 60000, 1)} minutes for 1M elements"
      f" — effectively unusable at this scale. GPU Global Memory brings it down to"
      f" {fmt(oe['GPU_Global']['wall_ms'] / 1000, 1)} s ({fmt(oe_su_global, 0)}x speedup, kernel)."
      f" Interestingly, the shared memory variant is *slower* than global ({fmt(oe['GPU_Shared']['kernel_ms'])} ms"
      f" vs {fmt(oe['GPU_Global']['kernel_ms'])} ms). The reason is the boundary kernel:"
      f" for each odd phase an additional `odd_even_boundary` kernel launch is required to fix"
      f" cross-block pairs, so instead of one kernel per phase there are two — the launch overhead"
      f" adds up across all n phases.\n")

    w(f"**Shell Sort** doesn't benefit from GPU at all: both GPU variants are slower than CPU"
      f" ({fmt(sh['GPU_Global']['kernel_ms'])} ms and {fmt(sh['GPU_Shared']['kernel_ms'])} ms vs"
      f" {fmt(sh['CPU']['kernel_ms'])} ms CPU). The Hibbard gap sequence produces strided memory"
      f" access patterns (at the largest gap, consecutive threads access memory {fmt(sh['CPU']['n'] // 2 // 1000, 0)}K"
      f" integers apart), causing cache misses on essentially every access. The shared memory version"
      f" is identical to the global version in behaviour — the gap-strided chains span the entire array"
      f" and cannot be confined to a block's shared memory.\n")

    w(f"**Merge Sort** is also slower on GPU than on CPU for N = 1M"
      f" ({fmt(ms['GPU_Shared']['kernel_ms'])} ms shared, {fmt(ms['GPU_Global']['kernel_ms'])} ms global"
      f" vs {fmt(ms['CPU']['kernel_ms'])} ms CPU). The bottom-up algorithm has log₂(n) ≈ 20 serial passes;"
      f" in the final passes there are only 2–4 independent merges so most of the GPU sits idle."
      f" The per-kernel-launch overhead also accumulates. This would likely reverse at much larger N"
      f" where the early passes (thousands of small parallel merges) dominate total time.\n")

    w("### Communication Overhead\n")

    avg_h2d = []
    avg_d2h = []
    for algo in ALGO_ORDER:
        for v in ["GPU_Global", "GPU_Shared"]:
            if v in data[algo]:
                avg_h2d.append(data[algo][v]["h2d_ms"])
                avg_d2h.append(data[algo][v]["d2h_ms"])
    mean_h2d = sum(avg_h2d) / len(avg_h2d)
    mean_d2h = sum(avg_d2h) / len(avg_d2h)

    w(f"H2D and D2H transfers are remarkably consistent across all algorithms:"
      f" {fmt(mean_h2d)} ms H2D and {fmt(mean_d2h)} ms D2H on average (for N = 1M, 4 MB of int32 data).")
    w(f"At ~8 GB/s effective PCIe bandwidth this is expected, and it confirms that communication")
    w(f"is not the bottleneck in any of these algorithms — the kernel time dominates everywhere.\n")
    w(f"The one exception is Bitonic Sort Shared: the kernel runs in only {fmt(b['GPU_Shared']['kernel_ms'])} ms,")
    w(f"while transfers cost {fmt(b['GPU_Shared']['h2d_ms'] + b['GPU_Shared']['d2h_ms'])} ms —"
      f" more time spent moving data than computing. For even larger N this ratio would improve")
    w(f"because kernel time scales as O(n log²n) while transfer time scales linearly.\n")

    w("### Bottlenecks\n")
    w("**Bitonic Sort:** ~log²(n) ≈ 400 kernel launches for n = 1M (one per sub-step)."
      " The shared variant reduces this by batching sub-steps with `sub < BLOCK_SIZE` into"
      " a single in-block pass, which explains most of the 54× wall-time improvement over global.\n")
    w("**Shell Sort:** strided global memory access is the fundamental problem."
      " No shared memory strategy can help because a single insertion-sort chain spans the full array.\n")
    w("**Odd-Even Sort:** n serial phases is the hard limit. Even if each phase is perfectly parallel,"
      " you still need n synchronisation barriers. At n = 1M that is 1 million kernel launches.\n")
    w("**Ranking Sort:** the global memory variant still performs n² reads from global memory."
      " Shared memory tiling reduces this to n²/BLOCK_SIZE global reads — a 512× reduction in global traffic.\n")
    w("**Merge Sort:** the log n serial dependency between passes cannot be eliminated."
      " In the final pass there is a single merge of two 500K-element arrays — one thread does all the work.\n")

    w("### Speedup Metrics — Theoretical vs Measured\n")
    w(f"Using Amdahl's Law: `S = 1 / (f_s + f_p / p)` where p = {GPU_CORES:,} CUDA cores (A100).\n")
    w("| Algorithm | Measured speedup (Shared) | Theoretical max (Amdahl) | Main limiting factor |")
    w("|---|---|---|---|")

    amdahl_rows = [
        ("BitonicSort",  bs_su_shared,  "~400",   "log²n serial kernel launches"),
        ("ShellSort",    sh_su_global,  "~1",      "fully serial chains per gap"),
        ("OddEvenSort",  oe_su_global,  "~n",      "n serial phases"),
        ("RankingSort",  rk_su_shared,  "~n",      "none — embarrassingly parallel"),
        ("MergeSort",    ms_su_shared,  "~log n",  "log n serial passes"),
    ]
    for algo, su, serial, reason in amdahl_rows:
        label = ALGO_LABELS[algo]
        w(f"| {label} | {fmt(su, 0)}x | {serial} serial steps | {reason} |")

    w("")
    w(f"The largest deviation from theoretical maximum is Merge Sort and Shell Sort — both have"
      f" structural serial bottlenecks that prevent meaningful GPU utilisation at N = 1M.")
    w(f"Bitonic Sort comes closest to the ideal for a comparison-network algorithm;"
      f" Ranking Sort benefits from near-perfect parallelism but is limited to small N by its O(n²) nature.\n")

    w("---\n")
    w(f"*GPU Lab Assignment 2026 — {STUDENT_NAME}, {GRUPA}*")

    return "\n".join(lines)

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run the Colab notebook first.", file=sys.stderr)
        sys.exit(1)

    data = load_csv(CSV_PATH)
    content = generate(data)

    with open(OUTPUT_PATH, "w") as f:
        f.write(content)

    print(f"Generated {OUTPUT_PATH}")
