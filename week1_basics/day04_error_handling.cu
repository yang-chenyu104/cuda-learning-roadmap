/*
 * Day 04: CUDA 错误处理 — 写健壮的 GPU 代码
 * ===========================================
 * 学习目标:
 *   1. 掌握 CUDA 错误检查的两种方式: 返回码和 kernel 执行错误
 *   2. 编写可复用的错误检查宏
 *   3. 理解常见的 CUDA 错误类型及其原因
 *   4. 学会使用 cudaDeviceSynchronize() 捕获异步错误
 *
 * 关键概念:
 *   - CUDA Runtime API 函数返回 cudaError_t 错误码
 *   - Kernel 启动是异步的, 错误不会立即返回
 *   - 需要 cudaDeviceSynchronize() 或 cudaMemcpy 来捕获 kernel 错误
 *   - 开发阶段应检查每一个 CUDA 调用
 *
 * 编译: nvcc day04_error_handling.cu -o day04
 * 运行: ./day04
 */

#include <stdio.h>
#include <stdlib.h>

// ============================================================
// 错误检查宏 — 这是 CUDA 编程的标准实践
// ============================================================

// 检查 CUDA Runtime API 调用
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA 错误 [%s:%d]: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// 检查 Kernel 启动后的错误 (包括异步执行错误)
#define CUDA_CHECK_KERNEL()                                                 \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "Kernel 启动错误 [%s:%d]: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
        err = cudaDeviceSynchronize();                                      \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "Kernel 执行错误 [%s:%d]: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// 不退出版本 (用于演示)
#define CUDA_CHECK_SOFT(call)                                               \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            printf("  [预期错误] %s\n", cudaGetErrorString(err));            \
        }                                                                   \
    } while (0)

// 简单 kernel
__global__ void safeAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// 故意越界访问的 kernel (用于演示错误捕获)
__global__ void badKernel(float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 故意越界: 不做边界检查
    C[idx] = 1.0f / C[idx + n];  // 越界!
}

int main() {
    printf("============================================\n");
    printf("       Day 04: CUDA 错误处理\n");
    printf("============================================\n\n");

    // --- 1. 正常流程: 每一步都检查 ---
    printf("--- 1. 正常流程 (每步检查) ---\n");
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < n; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    safeAdd<<<(n + 255) / 256, 256>>>(d_A, d_B, d_C, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    printf("  ✓ 正常流程完成, h_C[0] = %f (期望 3.0)\n\n", h_C[0]);

    // --- 2. 模拟常见错误 ---
    printf("--- 2. 常见错误演示 ---\n\n");

    // 错误 A: 分配过大内存
    printf("  [A] 尝试分配 100GB 内存...\n");
    float *d_huge;
    CUDA_CHECK_SOFT(cudaMalloc(&d_huge, (size_t)100 * 1024 * 1024 * 1024));

    // 错误 B: 无效设备序号
    printf("\n  [B] 尝试设置不存在的设备...\n");
    CUDA_CHECK_SOFT(cudaSetDevice(999));

    // 错误 C: 配置过大的 block
    printf("\n  [C] 尝试启动 blockSize=2048 (超过限制)...\n");
    badKernel<<<1, 2048>>>(d_C, n);
    CUDA_CHECK_SOFT(cudaGetLastError());

    // 错误 D: 越界访问 (需要同步才能捕获)
    printf("\n  [D] 越界访问 kernel...\n");
    badKernel<<<1, 256>>>(d_C, n);
    CUDA_CHECK_SOFT(cudaDeviceSynchronize());

    // --- 3. 常见错误类型速查表 ---
    printf("\n--- 3. 常见 CUDA 错误速查 ---\n");
    printf("  ┌─────────────────────────────┬──────────────────────────────────┐\n");
    printf("  │ 错误码                      │ 常见原因                         │\n");
    printf("  ├─────────────────────────────┼──────────────────────────────────┤\n");
    printf("  │ cudaErrorMemoryAllocation   │ 显存不足或请求过大               │\n");
    printf("  │ cudaErrorInvalidValue       │ 参数非法 (如负数大小)            │\n");
    printf("  │ cudaErrorInvalidDevice      │ 设备序号不存在                   │\n");
    printf("  │ cudaErrorInvalidConfiguration│ grid/block 配置超出限制         │\n");
    printf("  │ cudaErrorLaunchFailure      │ kernel 执行失败 (如越界)         │\n");
    printf("  │ cudaErrorLaunchOutOfResources│ 寄存器/共享内存不足              │\n");
    printf("  │ cudaErrorLaunchTimeout      │ 执行超时 (可能被OS抢占)          │\n");
    printf("  │ cudaErrorIllegalAddress     │ 非法内存地址访问                 │\n");
    printf("  │ cudaErrorAssert             │ device 代码中 assert 触发        │\n");
    printf("  └─────────────────────────────┴──────────────────────────────────┘\n");

    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    printf("\n✓ Day 04 完成! 养成检查每个 CUDA 调用的习惯。\n");
    return 0;
}
