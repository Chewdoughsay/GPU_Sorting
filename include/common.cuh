#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <climits>
#include <cuda_runtime.h>

typedef int DataType;

#define BLOCK_SIZE 512
#define DEFAULT_N  (1 << 20)

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t _e = (err);                                                \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d - %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

inline int next_power_of_two(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

inline int ilog2(int n) {
    int k = 0;
    while ((1 << k) < n) k++;
    return k;
}

inline void generate_random_data(DataType* arr, int n) {
    srand((unsigned)time(nullptr));
    for (int i = 0; i < n; i++)
        arr[i] = rand();
}

inline void write_data_to_file(const char* path, DataType* arr, int n) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror("write_data_to_file: fopen"); exit(EXIT_FAILURE); }
    fwrite(&n, sizeof(int), 1, f);
    fwrite(arr, sizeof(DataType), n, f);
    fclose(f);
}

inline DataType* read_data_from_file(const char* path, int* out_n) {
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;
    int n = 0;
    if (fread(&n, sizeof(int), 1, f) != 1) { fclose(f); return nullptr; }
    DataType* arr = (DataType*)malloc((size_t)n * sizeof(DataType));
    if (fread(arr, sizeof(DataType), n, f) != (size_t)n) {
        free(arr); fclose(f); return nullptr;
    }
    fclose(f);
    *out_n = n;
    return arr;
}

inline DataType* load_or_generate(const char* path, int default_n, int* out_n) {
    DataType* arr = read_data_from_file(path, out_n);
    if (arr) return arr;
    *out_n = default_n;
    arr = (DataType*)malloc((size_t)default_n * sizeof(DataType));
    generate_random_data(arr, default_n);
    write_data_to_file(path, arr, default_n);
    printf("Generated %d elements -> %s\n", default_n, path);
    return arr;
}

inline bool verify_sorted(DataType* arr, int n) {
    for (int i = 1; i < n; i++)
        if (arr[i] < arr[i - 1]) return false;
    return true;
}

inline DataType* copy_array(DataType* src, int n) {
    DataType* dst = (DataType*)malloc((size_t)n * sizeof(DataType));
    memcpy(dst, src, (size_t)n * sizeof(DataType));
    return dst;
}
