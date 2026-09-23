// ============================================================
// cuBLAS 学习 03: Level-2 矩阵×向量
// ------------------------------------------------------------
// 目标架构: 通用 (RTX 4090 sm_89)
// 学习目标: 矩阵-向量乘, 理解 op(A) 转置参数
// 关键函数: cublasSgemv
//
// 编译:
//   nvcc -O2 -std=c++17 -arch=sm_89 cublas_learning/03_level2_matvec.cu -o out -lcublas && ./out
//
// 参考: https://docs.nvidia.com/cuda/cublas/
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    printf("cuBLAS 学习 03: Level-2 矩阵×向量 (骨架, 待补全)\n");
    // TODO(学习): 用 cublasSgemv 完成本节目标
    return 0;
}
