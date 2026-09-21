/*
 * Day 01: Hello CUDA — 向量加法
 * ================================
 * 学习目标:
 *   1. 理解 CUDA 编程模型：Host (CPU) 与 Device (GPU) 的异构协作
 *   2. 掌握 kernel 函数的编写 (__global__)
 *   3. 理解 grid / block / thread 的层次结构
 *   4. 学会 cudaMalloc / cudaMemcpy / cudaFree 的基本内存管理
 *
 * 核心概念:
 *   - __global__: 在 device 上执行，从 host 调用 (可从 compute capability 3.x 起也从 device 调用)
 *   - <<<gridSize, blockSize>>>: 执行配置，指定 grid 中 block 数量和每个 block 中 thread 数量
 *   - threadIdx.x / blockIdx.x / blockDim.x / gridDim.x: 内建变量，用于计算全局线程索引
 *
 * 编译: nvcc day01_hello_cuda.cu -o day01
 * 运行: ./day01
 */

#include <stdio.h>

// ============================================================
// Kernel: 向量加法 C = A + B
// 每个 thread 负责一个元素的加法
// ============================================================
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    // 计算当前 thread 的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：防止越界访问
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // --- 1. 设置向量大小 ---
    int n = 1 << 20;  // 1M 个元素
    size_t size = n * sizeof(float);
    printf("向量大小: %d 元素 (%.2f MB)\n\n", n, (float)size / 1e6);

    // --- 2. 分配 Host 内存并初始化 ---
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i * 0.001f;
        h_B[i] = (float)(n - i) * 0.001f;
    }

    // --- 3. 分配 Device 内存 ---
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // --- 4. 将数据从 Host 拷贝到 Device (H2D) ---
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // --- 5. 配置 kernel 执行参数 ---
    int blockSize = 256;   // 每个 block 256 个 thread
    // 向上取整计算 grid 大小
    int gridSize = (n + blockSize - 1) / blockSize;

    printf("Kernel 配置:\n");
    printf("  Grid size:  %d blocks\n", gridSize);
    printf("  Block size: %d threads\n", blockSize);
    printf("  总线程数:   %d\n\n", gridSize * blockSize);

    // --- 6. 启动 Kernel ---
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // --- 7. 将结果从 Device 拷贝回 Host (D2H) ---
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // --- 8. 验证结果 ---
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            if (errors < 5) {
                printf("  错误: h_C[%d] = %f, 期望 = %f\n", i, h_C[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("验证通过! 所有 %d 个元素计算正确。\n", n);
    } else {
        printf("验证失败! 共 %d 个错误。\n", errors);
    }

    // 打印几个示例值
    printf("\n示例结果:\n");
    printf("  h_C[0]     = %f\n", h_C[0]);
    printf("  h_C[%d] = %f\n", n / 2, h_C[n / 2]);
    printf("  h_C[%d] = %f\n", n - 1, h_C[n - 1]);

    // --- 9. 释放内存 ---
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\n✓ Day 01 完成! 你已经写了第一个 CUDA 程序。\n");
    return 0;
}
