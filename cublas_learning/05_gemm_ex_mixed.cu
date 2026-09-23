// ============================================================
// cuBLAS 学习 05: 混合精度 GEMM
// ------------------------------------------------------------
// 目标架构: 通用 (RTX 4090 sm_89)
// 学习目标: FP16/TF32 输入 + FP32 累加, 算法选择
// 关键函数: cublasGemmEx
//
// 编译:
//   nvcc -O2 -std=c++17 -arch=sm_89 cublas_learning/05_gemm_ex_mixed.cu -o out -lcublas && ./out
//
// 参考: https://docs.nvidia.com/cuda/cublas/
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    printf("cuBLAS 学习 05: 混合精度 GEMM (骨架, 待补全)\n");
    // TODO(学习): 用 cublasGemmEx 完成本节目标
    return 0;
}
