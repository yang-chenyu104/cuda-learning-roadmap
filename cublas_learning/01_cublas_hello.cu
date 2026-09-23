// ============================================================
// cuBLAS 学习 01: 句柄与第一次调用 (SGEMM)
// ------------------------------------------------------------
// 学习目标:
//   1. cublasHandle_t 的创建与销毁
//   2. 理解 cuBLAS 的列主序 (column-major) 约定
//   3. 跑通一个 float GEMM: C = alpha*A*B + beta*C
//
// 目标架构: 通用 (RTX 4090 sm_89 可直接跑)
//
// 编译 (cuBLAS 随 CUDA 自带, 需链接 -lcublas):
//   nvcc -O2 -std=c++17 -arch=sm_89 \
//        cublas_learning/01_cublas_hello.cu -o cublas_hello -lcublas
//   ./cublas_hello
//
// 参考: https://docs.nvidia.com/cuda/cublas/
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x)  do { cudaError_t e=(x); if(e){ \
    printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; } } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s){ \
    printf("cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,(int)s); return 1; } } while(0)

int main() {
    // 计算 2x2 的 C = A * B (alpha=1, beta=0)
    // 注意: cuBLAS 是列主序! 下面按列主序摆放数据。
    const int M = 2, N = 2, K = 2;
    // A (列主序) = [[1,3],[2,4]] 表示矩阵 [[1,2],[3,4]]
    float hA[] = {1.f, 3.f, 2.f, 4.f};   // 列主序
    float hB[] = {5.f, 7.f, 6.f, 8.f};   // 列主序 = [[5,6],[7,8]]
    float hC[4] = {0};

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(hC)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.f, beta = 0.f;
    // C = alpha * A(MxK) * B(KxN) + beta * C ; ld 均为列数(列主序下为行数)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K, &alpha,
                             dA, M, dB, K, &beta, dC, M));

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost));

    // 期望 [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    printf("结果 C (列主序读出):\n");
    printf("  [%.0f  %.0f]\n", hC[0], hC[2]);
    printf("  [%.0f  %.0f]\n", hC[1], hC[3]);
    printf("期望:\n  [19  22]\n  [43  50]\n");

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
