#pragma once

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include "common.cuh"

struct TimingResult {
    float wall_ms;
    float h2d_ms;
    float kernel_ms;
    float d2h_ms;
};

struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    void end()   { CUDA_CHECK(cudaEventRecord(stop));  }
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline TimePoint wall_now() { return Clock::now(); }

inline float wall_elapsed_ms(TimePoint t0, TimePoint t1) {
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

inline void print_timing(const char* label, int n, TimingResult& r) {
    float comm_ms  = r.h2d_ms + r.d2h_ms;
    float comp_pct = (r.wall_ms > 0.f) ? 100.f * r.kernel_ms / r.wall_ms : 0.f;
    printf("%-22s | N=%-9d | wall=%8.2f ms | H2D=%7.2f ms | kernel=%8.2f ms"
           " | D2H=%7.2f ms | comm=%7.2f ms | comp%%=%5.1f%%\n",
           label, n, r.wall_ms, r.h2d_ms, r.kernel_ms, r.d2h_ms, comm_ms, comp_pct);
}

inline void append_csv(const char* path, const char* algorithm,
                        const char* variant, int n, TimingResult& r) {
    FILE* f = fopen(path, "a");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0)
        fprintf(f, "algorithm,variant,n,wall_ms,h2d_ms,kernel_ms,d2h_ms\n");
    fprintf(f, "%s,%s,%d,%.4f,%.4f,%.4f,%.4f\n",
            algorithm, variant, n, r.wall_ms, r.h2d_ms, r.kernel_ms, r.d2h_ms);
    fclose(f);
}
