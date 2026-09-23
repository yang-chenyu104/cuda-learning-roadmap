// ============================================================
// GPU 硬件基础: 查询本机 GPU 硬件参数
// ------------------------------------------------------------
// 目的: 把抽象的硬件概念(SM/warp/内存)变成本机的具体数字,
//       建立对自己 GPU 的直观认识。
//
// 编译:
//   nvcc -O2 -std=c++17 -arch=sm_89 hardware_basics/query_device.cu -o query
//   ./query
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 0;
    cudaGetDeviceCount(&n);
    if (n == 0) { printf("未检测到 CUDA 设备\n"); return 1; }
    printf("检测到 %d 个 CUDA 设备\n", n);

    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("\n========== 设备 %d: %s ==========\n", i, p.name);
        printf("计算能力 (Compute Capability): %d.%d  (sm_%d%d)\n",
               p.major, p.minor, p.major, p.minor);
        printf("SM 数量 (Streaming Multiprocessors): %d\n", p.multiProcessorCount);
        printf("Warp 大小: %d 线程\n", p.warpSize);
        printf("每 Block 最大线程数: %d\n", p.maxThreadsPerBlock);
        printf("每 SM 最大线程数: %d\n", p.maxThreadsPerMultiProcessor);
        printf("每 SM 最大 Block 数: %d\n", p.maxBlocksPerMultiProcessor);

        printf("--- 内存 ---\n");
        printf("全局内存 (显存): %.1f GB\n", p.totalGlobalMem / 1024.0 / 1024 / 1024);
        printf("每 Block 共享内存: %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("每 SM 共享内存: %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
        printf("每 Block 寄存器数: %d\n", p.regsPerBlock);
        printf("常量内存: %zu KB\n", p.totalConstMem / 1024);
        printf("L2 缓存: %.1f MB\n", p.l2CacheSize / 1024.0 / 1024);

        printf("--- 时钟与带宽 ---\n");
        printf("GPU 时钟: %.0f MHz\n", p.clockRate / 1000.0);
        printf("显存时钟: %.0f MHz\n", p.memoryClockRate / 1000.0);
        printf("显存位宽: %d bit\n", p.memoryBusWidth);
        // 理论带宽 = 显存时钟(Hz) * 位宽(bit) * 2(DDR) / 8(bit->byte)
        double bw = 2.0 * p.memoryClockRate * 1000.0 * (p.memoryBusWidth / 8) / 1e9;
        printf("理论显存带宽: %.0f GB/s\n", bw);
    }
    return 0;
}
