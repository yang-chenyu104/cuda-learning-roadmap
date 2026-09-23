// ============================================================
// cuBLAS 学习 06: 批量 GEMM
// ------------------------------------------------------------
// 目标架构: 通用 (RTX 4090 sm_89)
// 学习目标: 小矩阵批处理 (深度学习常用)
// 关键函数: cublasSgemmBatched / cublasSgemmStridedBatched
//
// 编译:
//   nvcc -O2 -std=c++17 -arch=sm_89 cublas_learning/06_batched_gemm.cu -o out -lcublas && ./out
//
// 参考: https://docs.nvidia.com/cuda/cublas/
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    printf("cuBLAS 学习 06: 批量 GEMM (骨架, 待补全)\n");
    // TODO(学习): 用 cublasSgemmBatched / cublasSgemmStridedBatched 完成本节目标
    return 0;
}
