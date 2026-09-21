/*
 * Day 02: GPU 设备查询 — 理解你的硬件
 * =====================================
 * 学习目标:
 *   1. 使用 CUDA Runtime API 查询 GPU 设备信息
 *   2. 理解关键硬件参数对编程的影响
 *   3. 学会根据硬件特性选择合适的 grid/block 配置
 *
 * 关键参数:
 *   - compute capability: 决定支持的特性集
 *   - multiProcessorCount (SM数): 决定并行度上限
 *   - maxThreadsPerBlock: 每 block 最大线程数 (通常 1024)
 *   - sharedMemPerBlock: 每 block 共享内存大小
 *   - warpSize: warp 大小 (通常 32)
 *   - maxThreadsDim[3]: 每 block 各维度最大线程数
 *   - maxGridSize[3]: grid 各维度最大 block 数
 *
 * 编译: nvcc day02_device_query.cu -o day02
 * 运行: ./day02
 */

#include <stdio.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    printf("============================================\n");
    printf("        CUDA 设备查询 — 共 %d 个设备\n", deviceCount);
    printf("============================================\n\n");

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("--- 设备 %d ---\n", dev);
        printf("  设备名称:            %s\n", prop.name);
        printf("  计算能力:            %d.%d\n", prop.major, prop.minor);
        printf("  总全局内存:          %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  SM 数量:             %d\n", prop.multiProcessorCount);
        printf("  每 SM 最大线程数:    %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  每 block 最大线程数: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp 大小:           %d\n", prop.warpSize);
        printf("  共享内存/block:      %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  共享内存/SM:         %.2f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
        printf("  常量内存:            %.2f KB\n", prop.totalConstMem / 1024.0);
        printf("  L2 缓存:             %.2f KB\n", prop.l2CacheSize / 1024.0);
        printf("  内存时钟频率:        %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  内存总线宽度:        %d bits\n", prop.memoryBusWidth);
        printf("  内存带宽:            %.2f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
        printf("  最大 grid 维度:      (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  最大 block 维度:     (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  是否支持统一内存:    %s\n", prop.unifiedAddressing ? "是" : "否");
        printf("  是否支持协作组:      %s\n", prop.cooperativeLaunch ? "是" : "否");
        printf("  多 GPU 板:           %s\n", prop.isMultiGpuBoard ? "是" : "否");
        printf("  异步引擎数量:        %d\n", prop.asyncEngineCount);
        printf("\n");

        // 计算理论最大并行线程数
        int maxThreads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
        printf("  ★ 理论最大并行线程: %d (%d 万)\n", maxThreads, maxThreads / 10000);
        printf("  ★ 建议 blockSize: 256 (warpSize 的整数倍)\n");
        printf("  ★ 建议 gridSize: 取决于数据量，保证足够 block 充满所有 SM\n\n");
    }

    // 演示如何根据硬件选择配置
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        printf("============================================\n");
        printf("  配置建议 (基于设备 0)\n");
        printf("============================================\n");
        int blockSize = 256;
        // 经验法则: 至少 2 倍 SM 数量的 block 来隐藏延迟
        int minBlocks = prop.multiProcessorCount * 2;
        int elementsPerThread = 4; // 每个 thread 处理多个元素
        int minElements = minBlocks * blockSize * elementsPerThread;

        printf("  推荐 blockSize: %d\n", blockSize);
        printf("  最少 block 数:  %d (2x SM)\n", minBlocks);
        printf("  最少元素数:     %d (每个thread处理%d个元素)\n", minElements, elementsPerThread);
        printf("  适合的数据规模: %d 万+ 元素\n\n", minElements / 10000);
    }

    printf("✓ Day 02 完成! 了解你的硬件是高效编程的基础。\n");
    return 0;
}
