#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "common.cuh"
#include "timer.cuh"

#define DATA_PATH   "data/input.bin"
#define RESULT_PATH "results/timings.csv"

void cpu_bitonic_sort(DataType* arr, int n) {
    for (int step = 2; step <= n; step <<= 1) {
        for (int sub = step >> 1; sub > 0; sub >>= 1) {
            for (int i = 0; i < n; i++) {
                int j = i ^ sub;
                if (j > i) {
                    bool ascending = ((i & step) == 0);
                    if ((ascending && arr[i] > arr[j]) ||
                        (!ascending && arr[i] < arr[j]))
                        std::swap(arr[i], arr[j]);
                }
            }
        }
    }
}

__global__ void bitonic_step_global(DataType* d_arr, int n, int step, int sub) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int j = i ^ sub;
    if (j > i) {
        bool ascending = ((i & step) == 0);
        if ((ascending && d_arr[i] > d_arr[j]) ||
            (!ascending && d_arr[i] < d_arr[j])) {
            DataType tmp = d_arr[i];
            d_arr[i]     = d_arr[j];
            d_arr[j]     = tmp;
        }
    }
}

void gpu_bitonic_global(DataType* d_arr, int n, TimingResult* r) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GpuTimer kt;
    kt.begin();
    for (int step = 2; step <= n; step <<= 1)
        for (int sub = step >> 1; sub > 0; sub >>= 1)
            bitonic_step_global<<<blocks, BLOCK_SIZE>>>(d_arr, n, step, sub);
    kt.end();
    r->kernel_ms = kt.elapsed_ms();
}

__global__ void bitonic_step_shared(DataType* d_arr, int n, int step, int sub) {
    extern __shared__ DataType s[];

    int tid  = threadIdx.x;
    int base = blockIdx.x * blockDim.x;
    int gi   = base + tid;

    s[tid] = (gi < n) ? d_arr[gi] : INT_MAX;
    __syncthreads();

    int j_global = gi ^ sub;
    int j_local  = j_global - base;

    if (j_local >= 0 && j_local < blockDim.x && j_global > gi) {
        bool ascending = ((gi & step) == 0);
        if ((ascending && s[tid] > s[j_local]) ||
            (!ascending && s[tid] < s[j_local])) {
            DataType tmp = s[tid];
            s[tid]       = s[j_local];
            s[j_local]   = tmp;
        }
    }
    __syncthreads();

    if (gi < n) d_arr[gi] = s[tid];
}

void gpu_bitonic_shared(DataType* d_arr, int n, TimingResult* r) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem = BLOCK_SIZE * sizeof(DataType);
    GpuTimer kt;
    kt.begin();
    for (int step = 2; step <= n; step <<= 1) {
        for (int sub = step >> 1; sub > 0; sub >>= 1) {
            if (sub < BLOCK_SIZE)
                bitonic_step_shared<<<blocks, BLOCK_SIZE, smem>>>(d_arr, n, step, sub);
            else
                bitonic_step_global<<<blocks, BLOCK_SIZE>>>(d_arr, n, step, sub);
        }
    }
    kt.end();
    r->kernel_ms = kt.elapsed_ms();
}

static DataType* pad_to_pow2(DataType* src, int n, int* out_n_padded) {
    int np = next_power_of_two(n);
    *out_n_padded = np;
    DataType* arr = (DataType*)malloc((size_t)np * sizeof(DataType));
    for (int i = 0; i < n; i++)  arr[i] = src[i];
    for (int i = n; i < np; i++) arr[i] = INT_MAX;
    return arr;
}

static void run_gpu_variant(DataType* h_padded, int np,
                             void (*sort_fn)(DataType*, int, TimingResult*),
                             TimingResult* r, TimePoint* t0_out) {
    size_t bytes = (size_t)np * sizeof(DataType);
    DataType* d_arr;

    TimePoint t0 = wall_now();
    *t0_out = t0;

    CUDA_CHECK(cudaMalloc(&d_arr, bytes));

    GpuTimer h2d;
    h2d.begin();
    CUDA_CHECK(cudaMemcpy(d_arr, h_padded, bytes, cudaMemcpyHostToDevice));
    h2d.end();
    r->h2d_ms = h2d.elapsed_ms();

    sort_fn(d_arr, np, r);

    GpuTimer d2h;
    d2h.begin();
    CUDA_CHECK(cudaMemcpy(h_padded, d_arr, bytes, cudaMemcpyDeviceToHost));
    d2h.end();
    r->d2h_ms = d2h.elapsed_ms();

    CUDA_CHECK(cudaFree(d_arr));
}

int main() {
    int n = 0;
    DataType* master = load_or_generate(DATA_PATH, DEFAULT_N, &n);

    int np;
    DataType* master_pad = pad_to_pow2(master, n, &np);

    printf("\n=== Bitonic Sort  (N = %d, padded to %d) ===\n", n, np);
    printf("%-22s | %-10s | %-12s | %-11s | %-14s | %-11s | %-11s | %s\n",
           "Variant", "N", "wall(ms)", "H2D(ms)", "kernel(ms)", "D2H(ms)",
           "comm(ms)", "comp%");

    {
        DataType* arr = copy_array(master_pad, np);
        TimePoint t0 = wall_now();
        cpu_bitonic_sort(arr, np);
        TimePoint t1 = wall_now();

        TimingResult r = {};
        r.wall_ms   = wall_elapsed_ms(t0, t1);
        r.kernel_ms = r.wall_ms;
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("CPU Sequential", n, r);
        append_csv(RESULT_PATH, "BitonicSort", "CPU", n, r);
        free(arr);
    }

    {
        DataType* arr = copy_array(master_pad, np);
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(arr, np, gpu_bitonic_global, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Global Mem", n, r);
        append_csv(RESULT_PATH, "BitonicSort", "GPU_Global", n, r);
        free(arr);
    }

    {
        DataType* arr = copy_array(master_pad, np);
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(arr, np, gpu_bitonic_shared, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Shared Mem", n, r);
        append_csv(RESULT_PATH, "BitonicSort", "GPU_Shared", n, r);
        free(arr);
    }

    free(master);
    free(master_pad);
    return 0;
}
