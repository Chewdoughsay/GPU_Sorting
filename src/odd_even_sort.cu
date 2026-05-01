#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "common.cuh"
#include "timer.cuh"

#define DATA_PATH   "data/input.bin"
#define RESULT_PATH "results/timings.csv"

void cpu_odd_even_sort(DataType* arr, int n) {
    for (int phase = 0; phase < n; phase++) {
        int start = phase & 1;
        for (int i = start; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1])
                std::swap(arr[i], arr[i + 1]);
        }
    }
}

__global__ void odd_even_step_global(DataType* d_arr, int n, int phase) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + (phase & 1);
    if (i < n - 1 && d_arr[i] > d_arr[i + 1]) {
        DataType tmp = d_arr[i];
        d_arr[i]     = d_arr[i + 1];
        d_arr[i + 1] = tmp;
    }
}

void gpu_odd_even_global(DataType* d_arr, int n, TimingResult* r) {
    int pairs  = n / 2;
    int blocks = (pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    GpuTimer kt;
    kt.begin();
    for (int phase = 0; phase < n; phase++)
        odd_even_step_global<<<blocks, BLOCK_SIZE>>>(d_arr, n, phase);
    kt.end();
    r->kernel_ms = kt.elapsed_ms();
}

__global__ void odd_even_step_shared(DataType* d_arr, int n, int phase) {
    extern __shared__ DataType s[];

    int tid  = threadIdx.x;
    int base = blockIdx.x * blockDim.x * 2;

    int gi0 = base + tid * 2;
    int gi1 = gi0 + 1;

    s[tid * 2]     = (gi0 < n) ? d_arr[gi0] : INT_MAX;
    s[tid * 2 + 1] = (gi1 < n) ? d_arr[gi1] : INT_MAX;
    __syncthreads();

    int local_start = phase & 1;
    int li = local_start + tid * 2;
    if (li + 1 < blockDim.x * 2 && s[li] > s[li + 1]) {
        DataType tmp = s[li];
        s[li]        = s[li + 1];
        s[li + 1]    = tmp;
    }
    __syncthreads();

    if (gi0 < n) d_arr[gi0] = s[tid * 2];
    if (gi1 < n) d_arr[gi1] = s[tid * 2 + 1];
}

__global__ void odd_even_boundary(DataType* d_arr, int n, int stride) {
    int k   = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = (k + 1) * stride - 1;
    if (pos >= n - 1) return;
    if (d_arr[pos] > d_arr[pos + 1]) {
        DataType tmp   = d_arr[pos];
        d_arr[pos]     = d_arr[pos + 1];
        d_arr[pos + 1] = tmp;
    }
}

void gpu_odd_even_shared(DataType* d_arr, int n, TimingResult* r) {
    int elems_per_block = BLOCK_SIZE * 2;
    int blocks          = (n + elems_per_block - 1) / elems_per_block;
    size_t smem         = (size_t)elems_per_block * sizeof(DataType);

    int num_boundaries = (n - 1) / elems_per_block;
    int bound_blocks   = (num_boundaries + BLOCK_SIZE - 1) / BLOCK_SIZE;

    GpuTimer kt;
    kt.begin();
    for (int phase = 0; phase < n; phase++) {
        odd_even_step_shared<<<blocks, BLOCK_SIZE, smem>>>(d_arr, n, phase);
        if ((phase & 1) && num_boundaries > 0)
            odd_even_boundary<<<bound_blocks, BLOCK_SIZE>>>(d_arr, n, elems_per_block);
    }
    kt.end();
    r->kernel_ms = kt.elapsed_ms();
}

static void run_gpu_variant(DataType* h_in, int n,
                             void (*sort_fn)(DataType*, int, TimingResult*),
                             TimingResult* r, TimePoint* t0_out) {
    size_t bytes = (size_t)n * sizeof(DataType);
    DataType* d_arr;

    TimePoint t0 = wall_now();
    *t0_out = t0;

    CUDA_CHECK(cudaMalloc(&d_arr, bytes));

    GpuTimer h2d;
    h2d.begin();
    CUDA_CHECK(cudaMemcpy(d_arr, h_in, bytes, cudaMemcpyHostToDevice));
    h2d.end();
    r->h2d_ms = h2d.elapsed_ms();

    sort_fn(d_arr, n, r);

    GpuTimer d2h;
    d2h.begin();
    CUDA_CHECK(cudaMemcpy(h_in, d_arr, bytes, cudaMemcpyDeviceToHost));
    d2h.end();
    r->d2h_ms = d2h.elapsed_ms();

    CUDA_CHECK(cudaFree(d_arr));
}

int main() {
    int n = 0;
    DataType* master = load_or_generate(DATA_PATH, DEFAULT_N, &n);

    printf("\n=== Odd-Even Sort  (N = %d) ===\n", n);
    printf("%-22s | %-10s | %-12s | %-11s | %-14s | %-11s | %-11s | %s\n",
           "Variant", "N", "wall(ms)", "H2D(ms)", "kernel(ms)", "D2H(ms)",
           "comm(ms)", "comp%");

    {
        DataType* arr = copy_array(master, n);
        TimePoint t0 = wall_now();
        cpu_odd_even_sort(arr, n);
        TimePoint t1 = wall_now();

        TimingResult r = {};
        r.wall_ms   = wall_elapsed_ms(t0, t1);
        r.kernel_ms = r.wall_ms;
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("CPU Sequential", n, r);
        append_csv(RESULT_PATH, "OddEvenSort", "CPU", n, r);
        free(arr);
    }

    {
        DataType* arr = copy_array(master, n);
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(arr, n, gpu_odd_even_global, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Global Mem", n, r);
        append_csv(RESULT_PATH, "OddEvenSort", "GPU_Global", n, r);
        free(arr);
    }

    {
        DataType* arr = copy_array(master, n);
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(arr, n, gpu_odd_even_shared, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Shared Mem", n, r);
        append_csv(RESULT_PATH, "OddEvenSort", "GPU_Shared", n, r);
        free(arr);
    }

    free(master);
    return 0;
}
