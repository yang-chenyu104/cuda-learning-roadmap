// ============================================================
// cuBLAS 学习 04: Level-3 矩阵乘 (核心)
// ------------------------------------------------------------
// 目标架构: 通用 (RTX 4090 sm_89)
// 学习目标: GEMM, alpha/beta, leading dimension
// 关键函数: cublasSgemm
//
// 编译:
//   nvcc -O2 -std=c++17 -arch=sm_89 cublas_learning/04_level3_gemm.cu -o out -lcublas && ./out
//
// 参考: https://docs.nvidia.com/cuda/cublas/
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    printf("cuBLAS 学习 04: Level-3 矩阵乘 (核心) (骨架, 待补全)\n");
    // TODO(学习): 用 cublasSgemm 完成本节目标
    return 0;
}
