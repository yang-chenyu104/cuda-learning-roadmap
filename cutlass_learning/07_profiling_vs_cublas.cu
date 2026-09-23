// ============================================================
// CUTLASS 学习 07: 性能对比 cuBLAS
// ------------------------------------------------------------
// 目标架构: 通用
// 学习目标: 用 CUTLASS profiler / 自测, 对比 cuBLAS 有效算力
//
// 参考: https://github.com/NVIDIA/cutlass
//
// 编译:
//   CUTLASS=~/cutlass
//   nvcc -O2 -std=c++17 -arch=sm_89 \
//        -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//        cutlass_learning/07_profiling_vs_cublas.cu -o out && ./out
// ============================================================

#include <cstdio>

int main() {
    printf("CUTLASS 学习 07: 性能对比 cuBLAS (骨架, 待补全)\n");
    // TODO(学习): 见文件头目标与参考链接, 逐步实现
    return 0;
}
