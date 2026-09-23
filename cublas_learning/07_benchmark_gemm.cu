// ============================================================
// cuBLAS 学习 07: 性能基准
// ------------------------------------------------------------
// 目标架构: 通用 (RTX 4090 sm_89)
// 学习目标: 测 TFLOPS/有效带宽, 供 CUTLASS 对照
// 关键函数: cublasSgemm + cudaEvent 计时
//
// 编译:
//   nvcc -O2 -std=c++17 -arch=sm_89 cublas_learning/07_benchmark_gemm.cu -o out -lcublas && ./out
//
// 参考: https://docs.nvidia.com/cuda/cublas/
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    printf("cuBLAS 学习 07: 性能基准 (骨架, 待补全)\n");
    // TODO(学习): 用 cublasSgemm + cudaEvent 计时 完成本节目标
    return 0;
}
