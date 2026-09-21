/*
 * Day 06: 向量点积 — 理解归约 (Reduction)
 * =========================================
 * 学习目标:
 *   1. 理解并行归约 (parallel reduction) 的概念
 *   2. 学习使用 __syncthreads() 进行线程同步
 *   3. 理解 block 内归约 + block 间归约的两阶段策略
 *   4. 对比不同归约策略的性能
 *
 * 关键概念:
 *   - 点积需要全局归约, 但 GPU 没有全局同步原语
 *   - 解决方案: 每个 block 做局部归约 -> 写回全局内存 -> 最终在 CPU 或另一个 kernel 归约
 *   - __syncthreads() 只能同步同一个 block 内的线程
 *   - 共享内存用于 block 内的高效数据交换
 *.  - cuda13.0
 * 编译: nvcc day06_dot_product.cu -o day06
 * 运行: ./day06
 */

 #include <stdio.h>

 #define BLOCK_SIZE 256
 #define WARP_SIZE 32

// ============================================================
// Kernel 1: 朴素归约 (interleaved addressing)
// 每个 block 处理一段数据, 归约到 block 内, 写入部分和
// ============================================================
__global__ void dotProductNaive(const float *A, const float *B, float *partialSums, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step1: 每个线程计算一个元素乘积,载入共享内存
    sdata[tid] = (idx < n) ? A[idx] * B[idx] : 0.0f;
    __syncthreads();

    // Step2: 归约 (interleaved addressing)
    // 注意: : 这种方式会有 bank conflict
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (2 * stride) == 0)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Step3: block 0 的thread 0 写入部分和
    if (tid == 0)
    {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// Kernel 2: 改进归约 (sequential addressing)
// 避免 bank conflict, 使用连续地址访问
// ============================================================
__global__ void dotProductImproved(const float *A, const float *B, float *partialSums, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? A[idx] * B[idx] : 0.0f;
    __syncthreads();

    // sequential addressing: tid 访问 sdata[tid] 和 sdata[tid + offset]
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        partialSums[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// Kernel 3: 最终阶段归约 (将所有 partial sum 归约为一个值)
// 用一个 block 处理所有 partial sums
// ============================================================
__global__ void finalReduction(float *partialSums, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;

    sdata[tid] = (tid < n) ? partialSums[tid] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        partialSums[0] = sdata[0];
    }

}

// CPU实现
float cpuDotProduct(const float *A, const float *B, int n)
{
    double sum = 0.0;
    for(int i = 0; i < n; i++)
    {
        sum += (double)A[i] * (double)B[i];
    }
    return (float)sum;
}

int main()
{
    printf("============================================\n");
    printf("       Day 06: 向量点积 — 并行归约\n");
    printf("============================================\n\n");

    int n = 1 << 24; //16M元素
    size_t size = n * sizeof(float);
    printf("数据量: %d 元素 (%.2f MB)\n\n", n, (float)size / 1e6);

    // Host数据
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);

    for (int i = 0; i <n; i++)
    {
        h_A[i] = (rand() % 1000) / 1000.0f;
        h_B[i] = (rand() % 1000) / 1000.0f;
    }

    //CPU 参考计算
    printf("--- CPU 参考计算 ---\n");
    float cpuResult = cpuDotProduct(h_A, h_B, n);
    printf("  CPU 点积结果: %f\n\n", cpuResult);

    // Device数据
    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t partialSize = gridSize * sizeof(float);
    float *d_partial;
    cudaMalloc(&d_partial, partialSize);

    // --- 朴素归约 ---
    printf("--- 朴素归约 (interleaved) ---\n");
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // 预热
        dotProductNaive<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_partial, n);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < 50; i++) {
            dotProductNaive<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_partial, n);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        //最终归约
        finalReduction<<<1, BLOCK_SIZE>>>(d_partial, gridSize);
        float gpuResult;
        cudaMemcpy(&gpuResult, d_partial, sizeof(float), cudaMemcpyDeviceToHost);

         printf("  GPU 结果: %f\n", gpuResult);
        printf("  误差: %e\n", fabs(gpuResult - cpuResult));
        printf("  耗时: %.3f ms (kernel only, 50次平均)\n", ms / 50);
        printf("  吞吐: %.2f GFLOPS\n", (2.0 * n) / (ms / 50 * 1e6));
        cudaEventDestroy(start); cudaEventDestroy(stop);
    } 

    // --- 改进归约 ---
    printf("\n- 改进归约(sequential) ---\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    //预热
    dotProductImproved<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_partial, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 50; i++) {
        dotProductImproved<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_partial, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    finalReduction<<<1, BLOCK_SIZE>>>(d_partial, gridSize);
    float gpuResult;
    cudaMemcpy(&gpuResult, d_partial, sizeof(float), cudaMemcpyDeviceToHost);

    printf("  GPU 结果: %f\n", gpuResult);
    printf("  误差: %e\n", fabs(gpuResult - cpuResult));
    printf("  耗时: %.3f ms (kernel only, 50次平均)\n", ms / 50);
    printf("  吞吐: %.2f GFLOPS\n", (2.0 * n) / (ms / 50 * 1e6));
    cudaEventDestroy(start); cudaEventDestroy(stop);

    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_partial);
    free(h_A); free(h_B);
    printf("\n✓ Day 06 完成! 掌握了并行归约的基本策略。\n");
    printf("  关键收获: sequential addressing 比 interleaved 更快 (无 bank conflict)。\n");
    return 0;
}  
