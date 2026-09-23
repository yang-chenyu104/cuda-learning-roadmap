// ============================================================
// cuBLAS 学习 02: Level-1 向量运算
// ------------------------------------------------------------
// 目标架构: 通用 (RTX 4090 sm_89)
// 学习目标: axpy/dot/nrm2 的用法
// 关键函数: cublasSaxpy / cublasSdot / cublasSnrm2
//
// 编译:
//   nvcc -O2 -std=c++17 -arch=sm_89 cublas_learning/02_level1_vector.cu -o out -lcublas && ./out
//
// 参考: https://docs.nvidia.com/cuda/cublas/
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    printf("cuBLAS 学习 02: Level-1 向量运算 (骨架, 待补全)\n");
    // TODO(学习): 用 cublasSaxpy / cublasSdot / cublasSnrm2 完成本节目标
    return 0;
}
