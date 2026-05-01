#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "common.cuh"
#include "timer.cuh"

#define DATA_PATH   "data/input.bin"
#define RESULT_PATH "results/timings.csv"

static void merge_cpu(DataType* src, DataType* dst, int lo, int mid, int hi) {
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi)
        dst[k++] = (src[i] <= src[j]) ? src[i++] : src[j++];
    while (i < mid) dst[k++] = src[i++];
    while (j < hi)  dst[k++] = src[j++];
}

void cpu_merge_sort(DataType* arr, int n) {
    DataType* tmp = (DataType*)malloc((size_t)n * sizeof(DataType));
    for (int width = 1; width < n; width <<= 1) {
        for (int lo = 0; lo < n; lo += width * 2) {
            int mid = lo + width;
            int hi  = lo + width * 2;
            if (mid > n) { memcpy(tmp + lo, arr + lo, (n - lo) * sizeof(DataType)); continue; }
            if (hi  > n) hi = n;
            merge_cpu(arr, tmp, lo, mid, hi);
        }
        memcpy(arr, tmp, (size_t)n * sizeof(DataType));
    }
    free(tmp);
}

__global__ void merge_step_global(DataType* d_in, DataType* d_out, int n, int width) {
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lo      = pair_id * width * 2;
    if (lo >= n) return;

    int mid = lo + width;
    int hi  = lo + width * 2;
    if (mid > n) {
        for (int i = lo; i < n; i++) d_out[i] = d_in[i];
        return;
    }
    if (hi > n) hi = n;

    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi)
        d_out[k++] = (d_in[i] <= d_in[j]) ? d_in[i++] : d_in[j++];
    while (i < mid) d_out[k++] = d_in[i++];
    while (j < hi)  d_out[k++] = d_in[j++];
}

void gpu_merge_global(DataType* d_buf0, DataType* d_buf1, int n, TimingResult* r) {
    GpuTimer kt;
    kt.begin();

    DataType* src = d_buf0;
    DataType* dst = d_buf1;

    for (int width = 1; width < n; width <<= 1) {
        int pairs  = (n + width * 2 - 1) / (width * 2);
        int blocks = (pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        merge_step_global<<<blocks, BLOCK_SIZE>>>(src, dst, n, width);
        DataType* t = src; src = dst; dst = t;
    }

    if (src != d_buf0)
        CUDA_CHECK(cudaMemcpy(d_buf0, src, (size_t)n * sizeof(DataType),
                              cudaMemcpyDeviceToDevice));

    kt.end();
    r->kernel_ms = kt.elapsed_ms();
}

__global__ void merge_step_shared(DataType* d_in, DataType* d_out, int n, int width) {
    extern __shared__ DataType s[];

    int pair_id = blockIdx.x;
    int lo      = pair_id * width * 2;
    if (lo >= n) return;

    int mid    = lo + width;
    int hi     = lo + width * 2;
    if (hi > n) hi = n;
    if (mid > n) {
        int tid = threadIdx.x;
        for (int i = lo + tid; i < n && i < lo + width * 2; i += blockDim.x)
            d_out[i] = d_in[i];
        return;
    }

    int left_len  = mid - lo;
    int right_len = hi - mid;
    int total     = left_len + right_len;

    int tid = threadIdx.x;
    for (int i = tid; i < left_len;  i += blockDim.x) s[i]            = d_in[lo + i];
    for (int i = tid; i < right_len; i += blockDim.x) s[left_len + i] = d_in[mid + i];
    __syncthreads();

    for (int k = tid; k < total; k += blockDim.x) {
        int i_lo = (k - right_len > 0) ? k - right_len : 0;
        int i_hi = (k < left_len)      ? k             : left_len;

        int i = i_lo;
        while (i < i_hi) {
            int im = i + (i_hi - i) / 2;
            if (s[im] <= s[left_len + k - im - 1])
                i = im + 1;
            else
                i_hi = im;
        }
        int j = k - i;
        DataType val_l = (i < left_len)  ? s[i]            : INT_MAX;
        DataType val_r = (j < right_len) ? s[left_len + j] : INT_MAX;
        d_out[lo + k] = (val_l <= val_r) ? val_l : val_r;
    }
}

void gpu_merge_shared(DataType* d_buf0, DataType* d_buf1, int n, TimingResult* r) {
    GpuTimer kt;
    kt.begin();

    DataType* src = d_buf0;
    DataType* dst = d_buf1;

    for (int width = 1; width < n; width <<= 1) {
        int pairs = (n + width * 2 - 1) / (width * 2);

        if (width <= BLOCK_SIZE / 2) {
            size_t smem = (size_t)(width * 2) * sizeof(DataType);
            merge_step_shared<<<pairs, BLOCK_SIZE, smem>>>(src, dst, n, width);
        } else {
            int blocks = (pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
            merge_step_global<<<blocks, BLOCK_SIZE>>>(src, dst, n, width);
        }

        DataType* t = src; src = dst; dst = t;
    }

    if (src != d_buf0)
        CUDA_CHECK(cudaMemcpy(d_buf0, src, (size_t)n * sizeof(DataType),
                              cudaMemcpyDeviceToDevice));

    kt.end();
    r->kernel_ms = kt.elapsed_ms();
}

typedef void (*MergeSortFn)(DataType*, DataType*, int, TimingResult*);

static void run_gpu_variant(DataType* h_in, int n,
                             MergeSortFn sort_fn,
                             TimingResult* r, TimePoint* t0_out) {
    size_t bytes = (size_t)n * sizeof(DataType);
    DataType *d_buf0, *d_buf1;

    TimePoint t0 = wall_now();
    *t0_out = t0;

    CUDA_CHECK(cudaMalloc(&d_buf0, bytes));
    CUDA_CHECK(cudaMalloc(&d_buf1, bytes));

    GpuTimer h2d;
    h2d.begin();
    CUDA_CHECK(cudaMemcpy(d_buf0, h_in, bytes, cudaMemcpyHostToDevice));
    h2d.end();
    r->h2d_ms = h2d.elapsed_ms();

    sort_fn(d_buf0, d_buf1, n, r);

    GpuTimer d2h;
    d2h.begin();
    CUDA_CHECK(cudaMemcpy(h_in, d_buf0, bytes, cudaMemcpyDeviceToHost));
    d2h.end();
    r->d2h_ms = d2h.elapsed_ms();

    CUDA_CHECK(cudaFree(d_buf0));
    CUDA_CHECK(cudaFree(d_buf1));
}

int main() {
    int n = 0;
    DataType* master = load_or_generate(DATA_PATH, DEFAULT_N, &n);

    printf("\n=== Merge Sort (Binary Reduction)  (N = %d) ===\n", n);
    printf("%-22s | %-10s | %-12s | %-11s | %-14s | %-11s | %-11s | %s\n",
           "Variant", "N", "wall(ms)", "H2D(ms)", "kernel(ms)", "D2H(ms)",
           "comm(ms)", "comp%");

    {
        DataType* arr = copy_array(master, n);
        TimePoint t0 = wall_now();
        cpu_merge_sort(arr, n);
        TimePoint t1 = wall_now();

        TimingResult r = {};
        r.wall_ms   = wall_elapsed_ms(t0, t1);
        r.kernel_ms = r.wall_ms;
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("CPU Sequential", n, r);
        append_csv(RESULT_PATH, "MergeSort", "CPU", n, r);
        free(arr);
    }

    {
        DataType* arr = copy_array(master, n);
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(arr, n, gpu_merge_global, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Global Mem", n, r);
        append_csv(RESULT_PATH, "MergeSort", "GPU_Global", n, r);
        free(arr);
    }

    {
        DataType* arr = copy_array(master, n);
        TimingResult r = {};
        TimePoint t0;
        run_gpu_variant(arr, n, gpu_merge_shared, &r, &t0);
        TimePoint t1 = wall_now();
        r.wall_ms = wall_elapsed_ms(t0, t1);
        bool ok = verify_sorted(arr, n);
        printf("[%s] ", ok ? "PASS" : "FAIL");
        print_timing("GPU Shared Mem", n, r);
        append_csv(RESULT_PATH, "MergeSort", "GPU_Shared", n, r);
        free(arr);
    }

    free(master);
    return 0;
}
