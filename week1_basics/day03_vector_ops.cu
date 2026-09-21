/*
 * Day 03: 向量运算 — 理解 Grid/Block 配置
 * =========================================
 * 学习目标:
 *   1. 理解不同 grid/block 配置对性能的影响
 *   2. 实现多种向量运算 (缩放、fma、sigmoid)
 *   3. 学习使用 2D grid/block 处理 2D 数据
 *   4. 理解 occupancy (占用率) 的概念
 *
 * 关键概念:
 *   - block size 应为 warpSize(32) 的整数倍
 *   - 太小的 block 会导致低 occupancy
 *   - 太大的 block 会限制每 SM 可驻留的 block 数
 *   - 经验值: 128 或 256 通常是好选择
 *
 * 编译: nvcc day03_vector_ops.cu -o day03
 * 运行: ./day03
 */

 #include <stdio.h>
 #include <math.h>

// ============================================================
// Kernel 1: 向量缩放 Y = alpha * X
// ============================================================

__global__ void vectorScale(const float *X, float *Y, float alpha, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        Y[idx] = alpha * X[idx];
    }
}

// ============================================================
// Kernel 2: 融合乘加 Z = alpha * X + beta * Y (FMA)
// 利用 GPU 的 FMA 指令，一条指令完成乘加
// ============================================================
__global__ void fusedMultiplyAdd(const float *X, const float *Y, float *Z, float alpha, float beta, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        Z[idx] = alpha * X[idx] + beta * Y[idx]; // 编译器会生成 FMA 指令
    }
}

// ============================================================
// Kernel 3: Sigmoid 激活函数 (逐元素超越函数运算)
// sigmoid(x) = 1 / (1 + exp(-x))
// 适合 GPU: 大量独立的计算
// ============================================================
__global__ void sigmoidKernel(const float *X,  float *Y,int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        Y[idx] = 1.0f / (1.0f + expf(-X[idx])); // 计算sigmoid
    }
}


// ============================================================
// Kernel 4: 使用 2D grid 处理 1D 数据 (每个线程处理多个元素)
// 这种 "grid-stride loop" 模式适用于数据量远大于线程数的情况
// ============================================================
__global__ void gridStrideLoop(const float *X, float *Y, float alpha, int n)
{
    int tid=  blockIdx.x * blockDim.x + threadIdx.x;
    // 总线程数
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride)
    {
        Y[i] = alpha * X[i] * X[i];
    }
}

// 辅助函数: 验证结果
bool verify(const float *result, const float *expected, int n, const char *name) {
    for (int i = 0; i < n; i++) {
        if (fabs(result[i] - expected[i]) > 1e-4f) {
            printf("  ✗ %s 验证失败: result[%d]=%f, expected=%f\n", name, i, result[i], expected[i]);
            return false;
        }
    }
    printf("  ✓ %s 验证通过\n", name);
    return true;
}

int main()
{
    int n = 1 << 22; //4M元素
    size_t size = n * sizeof(float);
    printf("数据量: %d 元素 (%.2f MB)\n\n", n, (float)size / 1e6);

    // 分配 Host 内存
    float *h_X = (float *)malloc(size);
    float *h_Y1 = (float *)malloc(size);
    float *h_Y2 = (float *)malloc(size);
    float *h_Y3 = (float *)malloc(size);
    float *h_Y4 = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        h_X[i] = (float)(i % 1000) * 0.01f - 5.0f;  // 范围 [-5, 5)
    }

    // 分配 Device 内存
    float *d_X, *d_Y;
    cudaMalloc(&d_X, size);
    cudaMalloc(&d_Y, size);
    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);

    // --- 测试不同 block size 的向量缩放 ---
    float alpha = 2.5f;
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);

    printf("===向量缩放 Y = %.1f * X ===\n", alpha );
    for (int c = 0; c < numConfigs; c++)
    {
        int bs = blockSizes[c];
        int gs = (n + bs - 1) / bs;

        // 使用CUDA 事件计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        //预热
        vectorScale<<<gs, bs>>>(d_X, d_Y, alpha, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        // 运行 100 次取平均
        for (int iter = 0; iter < 100; iter++) {
            vectorScale<<<gs, bs>>>(d_X, d_Y, alpha, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        float avgMs = ms / 100.0f;
        float bandwidth = (2.0f * size) / (avgMs * 1e-3) / 1e9; // 读+写

        printf("  blockSize=%4d  grid=%6d  耗时=%7.3f ms  带宽=%6.1f GB/s\n",
               bs, gs, avgMs, bandwidth);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


    }

    printf("\n=== FMA: Z = 2.0*X + 0.5*X ===\n");
    for(int c = 0; c < numConfigs; c++)
    {
        int bs = 256;
        int gs = (n + bs - 1) / bs;
        fusedMultiplyAdd<<<gs, bs>>>(d_X, d_X, d_Y, 2.0f, 0.5f, n);
        cudaMemcpy(h_Y2, d_Y, size, cudaMemcpyDeviceToHost);

        // CPU 验证
        bool ok = true;
        for (int i = 0; i < n; i++) {
            float expected = 2.0f * h_X[i] + 0.5f * h_X[i];
            if (fabs(h_Y2[i] - expected) > 1e-4f) { ok = false; break; }
        }
        printf("  %s\n", ok ? "✓ 验证通过" : "✗ 验证失败");
    }

    // --- 测试 Sigmoid ---
    printf("\n=== Sigmoid ===\n");
    for(int c = 0; c < numConfigs; c++)
    {
        int bs = 256;
        int gs = (n + bs - 1) / bs;
        sigmoidKernel<<<gs, bs>>>(d_X, d_Y, n);
        cudaMemcpy(h_Y3, d_Y, size, cudaMemcpyDeviceToHost);
        printf("  sigmoid(%f) = %f\n", h_X[0], h_Y3[0]);
        printf("  sigmoid(%f) = %f\n", h_X[n/2], h_Y3[n/2]);
        printf("  ✓ 执行完成\n");
    }

    // --- 测试 Grid-Stride Loop ---
    printf("\n=== Grid-Stride Loop: Y = %.1f * X^2 ===\n", alpha);
    for (int c = 0; c < numConfigs; c++)
    {
        // 只启动少量 block, 让每个线程处理多个元素
        int bs = 256;
        int gs = 256;  // 只有 256 个 block, 但数据有 4M
        printf("  线程数: %d, 数据量: %d, 每线程处理: %.1f 个元素\n",
               bs * gs, n, (float)n / (bs * gs));
        gridStrideLoop<<<gs, bs>>>(d_X, d_Y, alpha, n);
        cudaMemcpy(h_Y4, d_Y, size, cudaMemcpyDeviceToHost);

        bool ok = true;
        for (int i = 0; i < n; i++) {
            if (fabs(h_Y4[i] - alpha * h_X[i] * h_X[i]) > 1e-3f) { ok = false; break; }
        }
        printf("  %s\n", ok ? "✓ 验证通过" : "✗ 验证失败");
    }

    // 清理
    cudaFree(d_X); cudaFree(d_Y);
    free(h_X); free(h_Y1); free(h_Y2); free(h_Y3); free(h_Y4);

    printf("\n✓ Day 03 完成! 理解了 grid/block 配置和 grid-stride loop。\n");
    return 0;

}