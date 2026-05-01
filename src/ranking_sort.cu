#include <cstdio>
#include <cstdlib>
#include "common.cuh"
#include "timer.cuh"

#define RANKING_N   (1 << 15)
#define DATA_PATH   "data/input.bin"
#define RESULT_PATH "results/timings.csv"

void cpu_ranking_sort(DataType* in, DataType* out, int n) {
    for (int i = 0; i < n; i++) {
        int rank = 0;
        for (int j = 0; j < n; j++) {
            if (in[j] < in[i] || (in[j] == in[i] && j < i))
                rank++;
        }
        out[rank] = in[i];
    }
}

__global__ void ranking_sort_global(DataType* d_in, DataType* d_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    DataType val = d_in[i];
    int rank = 0;
    for (int j = 0; j < n; j++) {
        if (d_in[j] < val || (d_in[j] == val && j < i))
            rank++;
    }
    d_out[rank] = val;
}

__global__ void ranking_sort_shared(DataType* d_in, DataType* d_out, int n) {
    extern __shared__ DataType s_tile[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    DataType val = (i < n) ? d_in[i] : 0;
    int rank = 0;

    for (int tile_start = 0; tile_start < n; tile_start += blockDim.x) {
        int load_idx = tile_start + threadIdx.x;
        s_tile[threadIdx.x] = (load_idx < n) ? d_in[load_idx] : INT_MAX;
        __syncthreads();

        if (i < n) {
            for (int t = 0; t < blockDim.x && (tile_start + t) < n; t++) {
                int j = tile_start + t;
                DataType jval = s_tile[t];
                if (jval < val || (jval == val && j < i))
                    rank++;
            }
        }
        __syncthreads();
    }

    if (i < n) d_out[rank] = val;
}

static void run_gpu_variant(DataType* h_in, DataType* h_out, int n,
                             void (*kernel_fn)(DataType*, DataType*, int),
                             bool use_smem, TimingResult* r, TimePoint* t0_out) {
    size_t bytes = (size_t)n * sizeof(DataType);
    DataType *d_in, *d_out;

    TimePoint t0 = wall_now();
    *t0_out = t0;

    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    GpuTimer h2d;
    h2d.begin();
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    h2d.end();
    r->h2d_ms = h2d.elapsed_ms();

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem = use_smem ? (size_t)BLOCK_SIZE * sizeof(DataType) : 0;

    GpuTimer kt;
    kt.begin();
    kernel_fn<<<blocks, BLOCK_SIZE, smem>>>(d_in, d_out, n);
    kt.end();
    r->kernel_ms = kt.elapsed_ms();

    GpuTimer d2h;
    d2h.begin();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    d2h.end();
    r->d2h_ms = d2h.elapsed_ms();

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    int file_n = 0;
    DataType* master_full = load_or_generate(DATA_PATH, DEFAULT_N, &file_n);

    int n = (file_n < RANKING_N) ? file_n : RANKING_N;
    DataType* master = copy_array(master_full, n);
    free(master_full);

    printf("\n=== Ranking Sort  (N = %d, O(n^2)) ===\n", n);
    printf("NOTE: N capped at %d due to O(n^2) complexity.\n", RANKING_N);
    printf("%-22s | %-10s | %-12s | %-11s | %-14s | %-11s | %-11s | %s\n",
           "Variant", "N", "wall(ms)", "H2D(ms)", "kernel(ms)", "D2H(ms)",
           "comm(ms)", "comp%");

    {
        DataType* in  = copy_array(master, n);
        DataType* out = (DataType*)malloc((size_t)n * sizeof(DataType));

        TimePoint t0 = wall_now();
        cpu_ranking_sort(in, out, n);
        TimePoint t1 = wall_now();

        TimingResult r = {};
        r.wall_ms   = wall_elapsed_ms(t0, t1);
        r.kernel_ms = r.wall_ms;
        bool ok = verify_sorted(out, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("CPU Sequential", n, r);
        append_csv(RESULT_PATH, "RankingSort", "CPU", n, r);
        free(in); free(out);
    }

    {
        DataType* in  = copy_array(master, n);
        DataType* out = (DataType*)malloc((size_t)n * sizeof(DataType));
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(in, out, n, ranking_sort_global, false, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(out, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Global Mem", n, r);
        append_csv(RESULT_PATH, "RankingSort", "GPU_Global", n, r);
        free(in); free(out);
    }

    {
        DataType* in  = copy_array(master, n);
        DataType* out = (DataType*)malloc((size_t)n * sizeof(DataType));
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(in, out, n, ranking_sort_shared, true, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(out, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Shared Mem", n, r);
        append_csv(RESULT_PATH, "RankingSort", "GPU_Shared", n, r);
        free(in); free(out);
    }

    free(master);
    return 0;
}
