#include <cstdio>
#include <cstdlib>
#include "common.cuh"
#include "timer.cuh"

#define DATA_PATH   "data/input.bin"
#define RESULT_PATH "results/timings.csv"

static void build_hibbard_gaps(int n, int* gaps, int* num_gaps) {
    int k = 0;
    for (int g = 1; g < n; g = (g << 1) | 1)
        gaps[k++] = g;
    for (int i = 0, j = k - 1; i < j; i++, j--) {
        int tmp = gaps[i]; gaps[i] = gaps[j]; gaps[j] = tmp;
    }
    *num_gaps = k;
}

void cpu_shell_sort(DataType* arr, int n) {
    int gaps[64]; int ng;
    build_hibbard_gaps(n, gaps, &ng);

    for (int gi = 0; gi < ng; gi++) {
        int gap = gaps[gi];
        for (int i = gap; i < n; i++) {
            DataType tmp = arr[i];
            int j = i;
            while (j >= gap && arr[j - gap] > tmp) {
                arr[j] = arr[j - gap];
                j -= gap;
            }
            arr[j] = tmp;
        }
    }
}

__global__ void shell_step_global(DataType* d_arr, int n, int gap) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    if (start >= gap) return;

    for (int i = start + gap; i < n; i += gap) {
        DataType tmp = d_arr[i];
        int j = i;
        while (j >= gap && d_arr[j - gap] > tmp) {
            d_arr[j] = d_arr[j - gap];
            j -= gap;
        }
        d_arr[j] = tmp;
    }
}

void gpu_shell_global(DataType* d_arr, int n, TimingResult* r) {
    int gaps[64]; int ng;
    build_hibbard_gaps(n, gaps, &ng);

    GpuTimer kt;
    kt.begin();
    for (int gi = 0; gi < ng; gi++) {
        int gap    = gaps[gi];
        int blocks = (gap + BLOCK_SIZE - 1) / BLOCK_SIZE;
        shell_step_global<<<blocks, BLOCK_SIZE>>>(d_arr, n, gap);
    }
    kt.end();
    r->kernel_ms = kt.elapsed_ms();
}

__global__ void shell_step_shared(DataType* d_arr, int n, int gap) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    if (start >= gap) return;

    for (int i = start + gap; i < n; i += gap) {
        DataType tmp = d_arr[i];
        int j = i;
        while (j >= gap && d_arr[j - gap] > tmp) {
            d_arr[j] = d_arr[j - gap];
            j -= gap;
        }
        d_arr[j] = tmp;
    }
}

void gpu_shell_shared(DataType* d_arr, int n, TimingResult* r) {
    int gaps[64]; int ng;
    build_hibbard_gaps(n, gaps, &ng);

    GpuTimer kt;
    kt.begin();
    for (int gi = 0; gi < ng; gi++) {
        int gap    = gaps[gi];
        int blocks = (gap + BLOCK_SIZE - 1) / BLOCK_SIZE;
        shell_step_shared<<<blocks, BLOCK_SIZE>>>(d_arr, n, gap);
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

    printf("\n=== Shell Sort  (N = %d, Hibbard gaps) ===\n", n);
    printf("%-22s | %-10s | %-12s | %-11s | %-14s | %-11s | %-11s | %s\n",
           "Variant", "N", "wall(ms)", "H2D(ms)", "kernel(ms)", "D2H(ms)",
           "comm(ms)", "comp%");

    {
        DataType* arr = copy_array(master, n);
        TimePoint t0 = wall_now();
        cpu_shell_sort(arr, n);
        TimePoint t1 = wall_now();

        TimingResult r = {};
        r.wall_ms   = wall_elapsed_ms(t0, t1);
        r.kernel_ms = r.wall_ms;
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("CPU Sequential", n, r);
        append_csv(RESULT_PATH, "ShellSort", "CPU", n, r);
        free(arr);
    }

    {
        DataType* arr = copy_array(master, n);
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(arr, n, gpu_shell_global, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Global Mem", n, r);
        append_csv(RESULT_PATH, "ShellSort", "GPU_Global", n, r);
        free(arr);
    }

    {
        DataType* arr = copy_array(master, n);
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(arr, n, gpu_shell_shared, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Shared Mem", n, r);
        append_csv(RESULT_PATH, "ShellSort", "GPU_Shared", n, r);
        free(arr);
    }

    free(master);
    return 0;
}
