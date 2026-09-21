/*
 * Day 05: GPU 计时 — 精确测量性能
 * =================================
 * 学习目标:
 *   1. 使用 CUDA Event 精确计时 GPU 操作
 *   2. 理解 CPU 计时和 GPU 计时的区别
 *   3. 学习预热(warmup)的重要性
 *   4. 对比 CPU 和 GPU 的性能差异
 *
 * 关键概念:
 *   - CUDA Event 在 GPU 时间线上记录, 不受 CPU 时钟影响
 *   - cudaEventRecord() 是异步的, 需要 synchronize
 *   - 首次 kernel 启动有初始化开销, 需要预热
 *   - 数据传输 H2D/D2H 的时间也需要计入
 *   - cuda13.0 -
 * 编译: nvcc day05_gpu_timer.cu -o day05
 * 运行: ./day05
 */

#include <stdio.h>
#include <sys/time.h>

// CPU 计时函数
double cpuSecond()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

// 计算密集型 kernel: 每个元素做多次三角函数运算
__global__ void heavyCompute(const float *in, float *out, int n, int iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = in[idx];
        for (int i = 0; i < iters; ++i)
        {
            val = sinf(val) + cosf(val * 0.5f);
            val = val * val;
        }
        out[idx] = val;
    }
}

// CPU版本
void cpuCompute(const float *in, float *out, int n, int iters)
{
    for (int i = 0; i < n; ++i)
    {
        float val = in[i];
        for (int j = 0; j < iters; ++j)
        {
            val = sinf(val) + cosf(val * 0.5f);
            val = val * val;
        }
        out[i] = val;
    }
}

int main()
{
    printf("============================================\n");
    printf("       Day 05: GPU 计时与性能对比\n");
    printf("============================================\n\n");

    int n = 1 << 22;
    int iters = 100;
    size_t size = n * sizeof(float);

    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);

    for (int i = 0; i < n; ++i)
    {
        h_in[i] = (float)i * 0.0001f;
    }

    // --- CPU 计时 ---
    printf("--- CPU 计算 ---\n");
    double cpuStart = cpuSecond();
    cpuCompute(h_in, h_out, n, iters);
    double cpuTime = cpuSecond() - cpuStart;
    printf("  CPU 耗时: %.3f ms\n\n", cpuTime * 1000);

    // --- GPU 计时 ---
     printf("--- GPU 计算 ---\n");
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    //创建 CUDA events
    cudaEvent_t start, stop, startTotal, stopTotal;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    //记录总时间(包括数据传输)
    cudaEventRecord(startTotal);

    // 预热(首次 kernel 有开销)
    heavyCompute<<<gridSize, blockSize>>>(d_in, d_out, n, 1);
    cudaDeviceSynchronize();

    // 精确计时 kernel
    cudaEventRecord(start);
    heavyCompute<<<gridSize, blockSize>>>(d_in, d_out, n, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // D2H 传输
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopTotal);
    cudaEventSynchronize(stopTotal);

    float kernelMs = 0, totalMs = 0;
    cudaEventElapsedTime(&kernelMs, start, stop);
    cudaEventElapsedTime(&totalMs, startTotal, stopTotal);

    float h2dMs = 0, d2hMs = 0;
    // 单独测传输时间
    cudaEventRecord(start);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dMs, start, stop);

    cudaEventRecord(start);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hMs, start, stop);

    printf("  Kernel 计算耗时:  %8.3f ms\n", kernelMs);
    printf("  H2D 传输耗时:     %8.3f ms\n", h2dMs);
    printf("  D2H 传输耗时:     %8.3f ms\n", d2hMs);
    printf("  总耗时 (含传输):  %8.3f ms\n", totalMs);
    printf("\n");

    // 性能对比
    printf("--- 性能对比 ---\n");
    printf("  CPU 耗时:         %8.3f ms\n", cpuTime * 1000);
    printf("  GPU Kernel 耗时:  %8.3f ms\n", kernelMs);
    printf("  GPU 总耗时:       %8.3f ms (含数据传输)\n", totalMs);
    printf("  加速比 (kernel):  %8.1f x\n", (cpuTime * 1000) / kernelMs);
    printf("  加速比 (总计):    %8.1f x\n", (cpuTime * 1000) / totalMs);
    printf("\n");

    // 吞吐量分析
    float gflops = (2.0f * n * iters * 2) / (kernelMs * 1e6); // sin+cos=2 ops, *2 for mul+add
    printf("--- 吞吐量分析 ---\n");
    printf("  计算量: %.2f GFLOP\n", (2.0f * n * iters * 2) / 1e9);
    printf("  计算吞吐: %.2f GFLOPS\n", gflops);
    printf("  内存带宽: %.2f GB/s (kernel)\n", (2.0f * size) / (kernelMs * 1e6));

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    printf("\n✓ Day 05 完成! 学会精确测量 GPU 性能。\n");
    printf("  关键收获: 数据传输开销不可忽视, 计算量足够大时 GPU 优势明显。\n");
    return 0;

}