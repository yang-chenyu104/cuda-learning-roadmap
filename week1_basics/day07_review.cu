/*
 * Day 07: Week 1 综合练习 — CUDA 矩阵运算
 * ==========================================
 * 本日综合运用 Week 1 学到的所有知识:
 *   - Kernel 编写与执行配置
 *   - 错误处理宏
 *   - CUDA Event 计时
 *   - 共享内存初步使用
 *   - CPU/GPU 性能对比
 *
 * 任务: 实现 1D 卷积 (模糊滤波)
 *   output[i] = sum_{j=-R}^{R} input[i+j] * kernel[j] / (2R+1)
 * 这比向量加法更复杂, 需要处理边界和数据复用
 *
 * 编译: nvcc day07_week1_review.cu -o day07
 * 运行: ./day07
 */

#include <stdio.h>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

 #define BLOCK_SIZE 256
 #define RADIUS 7 // 卷积半径(kernel 大小 = 2 * RADIUS + 1 = 15)

// ============================================================
// Kernel 1: 朴素 1D 卷积 (无共享内存)
// 每个线程独立从全局内存读取数据, 有大量冗余读取
// ============================================================
__global__ void conv1dNaive(const float *input, float *output, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float sum = 0.0f;
    for (int j = -RADIUS; j <= RADIUS; j++){
        int sIdx = idx + j;
        // 边界处理: clamp 到 [0, n-1]
        sIdx = max(0, min(n - 1, sIdx));
        sum += input[sIdx];
    }
    output[idx] = sum / (2 * RADIUS + 1);
}

// ============================================================
// Kernel 2: 使用共享内存的 1D 卷积
// 每个 block 协作加载一段数据到共享内存, 减少全局内存访问
// ============================================================
__global__ void conv1dShared(const float *input,float *output, int n){
    // 共享内存: blockSize + 2*RADIUS (halo 区域)
    __shared__ float sdata[BLOCK_SIZE + 2 * RADIUS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载中心区域
    int loadIdx = gid - RADIUS;
    if (loadIdx >= 0 && loadIdx < n) {
        sdata[tid] = input[loadIdx];
    } else {
        sdata[tid] = 0.0f;
    }

    // 第一个block需要halo(左侧)
    if (tid < RADIUS){
        int haloIdx = gid - RADIUS - (RADIUS - tid);
        // 用clamp处理边界
        haloIdx = max(0, min(n - 1, haloIdx));
        // 其实是 sdata[0..RADIUS-1] 对应 input[gid-RADIUS-RADIUS+tid]
    }
    // 修正:重新设计共享内存布局
    //sdata[0 .. RADIUS-1] -> 左侧halo
    // sdata[RADIUS .. RADIUS+BLOCK_SIZE-1] -> 中心数据
    // sdata[RADIUS+BLOCK_SIZE ..] -> 右侧 halo

    // 简化: 直接加载
    // 每个 thread 加载一个中心元素
    int centerIdx = gid;
    if (centerIdx < n){
        sdata[RADIUS + tid] = input[centerIdx];
    }
    else {
        sdata[RADIUS + tid] = 0.0f;
    }

    // 加载左侧 halo (前 RADIUS 个线程)
    if (tid < RADIUS) {
        int lIdx = blockIdx.x * blockDim.x - RADIUS + tid;
        sdata[tid] = (lIdx >= 0) ? input[lIdx] : input[0];
    }

    // 加载右侧 halo (后 RADIUS 个线程)
    if (tid >= BLOCK_SIZE - RADIUS) {
        int rIdx = (blockIdx.x + 1) * blockDim.x + (tid - (BLOCK_SIZE - RADIUS));
        sdata[RADIUS + BLOCK_SIZE + (tid - (BLOCK_SIZE - RADIUS))] =
            (rIdx < n) ? input[rIdx] : input[n - 1];
    }

    __syncthreads();

    // 计算卷积
    if (gid < n){
        float sum = 0.0f;
        for(int j = 0; j < 2 * RADIUS + 1; j++){
            sum += sdata[tid + j];
        }
        output[gid] = sum / (2 * RADIUS + 1);
    }
}

// CPU实现
void conv1dCPU(const float *input, float *output, int n){
    for(int i = 0; i < n; i++){
        float sum = 0.0f;
        for (int j = -RADIUS; j <= RADIUS; j++){
            int idx = i + j;
            idx = (idx < 0) ? 0 : (idx >= n ? n - 1 : idx);
            sum += input[idx];
        }
        output[i] = sum / (2 * RADIUS + 1);
    }
}


int main()
{
    printf("============================================\n");
    printf("   Day 07: Week1 综合 — 1D 卷积\n");
    printf("============================================\n\n");

    int n = 1 << 24;  // 16M
    size_t size = n * sizeof(float);
    printf("数据量: %d 元素\n", n);
    printf("卷积核大小: %d (radius=%d)\n\n", 2 * RADIUS + 1, RADIUS);

    float *h_in  = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    float *h_ref = (float *)malloc(size);

    // 生成测试数据: 模拟有噪声的信号
    for (int i = 0; i < n; i++) {
        h_in[i] = sinf(i * 0.001f) * 100.0f + (rand() % 100 - 50) * 0.1f;
    }

    // CPU参考
    printf("--- CPU 计算 ---\n");
    auto cpuT1 = std::chrono::high_resolution_clock::now();
    for (int i =0; i < 10;i++){
        conv1dCPU(h_in, h_ref, n);
    }
    auto cpuT2 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(cpuT2 - cpuT1).count();
    printf(" CPU计算完成，耗时: %.3f ms(10次平均)\n", cpuMs / 10);
    // CPU计时用CUDA event 不准,用粗略方式

    // GPU
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_out, size));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 朴素版本

    printf("--- 朴素卷积 (全局内存) ---\n");
    cudaEvent_t start, stop;
    float ms;
    float maxErr = 0;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // 预热
    conv1dNaive<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++){
        conv1dNaive<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // 验证
    for (int i = 0; i < n; i++) {
        maxErr = fmaxf(maxErr, fabsf(h_out[i] - h_ref[i]));
    }
    printf("  耗时: %.3f ms (10次平均)\n", ms / 10);
    printf("  最大误差: %e\n", maxErr);
    printf("  有效带宽: %.2f GB/s\n", (2.0f * size) / (ms / 10 * 1e6));
    cudaEventDestroy(start); cudaEventDestroy(stop);
   
    // 共享内存版本
    printf("\n--- 共享内存卷积 ---\n");
    maxErr = 0;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // 预热
    conv1dShared<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        conv1dShared<<<gridSize, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    
    for (int i = 0; i < n; i++) {
        maxErr = fmaxf(maxErr, fabsf(h_out[i] - h_ref[i]));
    }
    printf("  耗时: %.3f ms (10次平均)\n", ms / 10);
    printf("  最大误差: %e\n", maxErr);
    printf("  有效带宽: %.2f GB/s\n", (2.0f * size) / (ms / 10 * 1e6));
    cudaEventDestroy(start); cudaEventDestroy(stop);

    // 清理,释放资源
    CUDA_CHECK(cudaFree(d_in));CUDA_CHECK(cudaFree(d_out));
    free(h_in); free(h_out); free(h_ref);

    printf("  本周学到了: kernel编写 → 设备查询 → 配置优化 → 错误处理 → 计时 → 归约 → 综合应用\n");
    printf("  下周将深入 CUDA 内存模型!\n");
    return 0;
}