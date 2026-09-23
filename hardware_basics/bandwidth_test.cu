// ============================================================
// GPU 硬件基础: 实测显存带宽
// ------------------------------------------------------------
// 目的: 亲手测出 H2D / D2H / D2D 的实际带宽, 理解"带宽墙"——
//       很多 kernel 的性能上限是显存带宽, 而非算力。
//
// 编译:
//   nvcc -O2 -std=c++17 -arch=sm_89 hardware_basics/bandwidth_test.cu -o bw
//   ./bw
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e){ \
    printf("CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} }while(0)

// 计时一次拷贝, 返回带宽 GB/s
static float bench(void* dst, const void* src, size_t bytes,
                   cudaMemcpyKind kind, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    // 预热
    cudaMemcpy(dst, src, bytes, kind);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) cudaMemcpy(dst, src, bytes, kind);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    double gb = (double)bytes * iters / 1e9;
    return (float)(gb / (ms / 1000.0));
}

int main() {
    const size_t N = 256 * 1024 * 1024;   // 256 MB
    const int iters = 20;

    char *hPage = (char*)malloc(N);        // 可分页内存
    char *hPin = nullptr; CK(cudaMallocHost(&hPin, N));  // 固定内存(pinned)
    char *d1, *d2; CK(cudaMalloc(&d1, N)); CK(cudaMalloc(&d2, N));

    printf("数据量: %zu MB, 迭代 %d 次\n\n", N/1024/1024, iters);

    printf("H2D (可分页): %.1f GB/s\n", bench(d1, hPage, N, cudaMemcpyHostToDevice, iters));
    printf("H2D (固定内存): %.1f GB/s   <- pinned 更快\n", bench(d1, hPin, N, cudaMemcpyHostToDevice, iters));
    printf("D2H (固定内存): %.1f GB/s\n", bench(hPin, d1, N, cudaMemcpyDeviceToHost, iters));
    printf("D2D (设备内部): %.1f GB/s   <- 最接近理论带宽\n", bench(d2, d1, N, cudaMemcpyDeviceToDevice, iters));

    printf("\n提示: 用 query_device 看到的'理论带宽'做对比,\n");
    printf("      D2D 通常能达到理论值的 ~80%%+。\n");

    free(hPage); cudaFreeHost(hPin); cudaFree(d1); cudaFree(d2);
    return 0;
}
