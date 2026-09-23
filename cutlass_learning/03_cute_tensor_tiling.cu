// ============================================================
// CUTLASS 学习 03: CuTe Tensor 与分块
// ------------------------------------------------------------
// 目标架构: 通用
// 学习目标: make_tensor / local_tile, 把全局内存切成 CTA tile
//
// 参考: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
//
// 编译:
//   CUTLASS=~/cutlass
//   nvcc -O2 -std=c++17 -arch=sm_89 \
//        -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//        cutlass_learning/03_cute_tensor_tiling.cu -o out && ./out
// ============================================================

#include <cstdio>

int main() {
    printf("CUTLASS 学习 03: CuTe Tensor 与分块 (骨架, 待补全)\n");
    // TODO(学习): 见文件头目标与参考链接, 逐步实现
    return 0;
}
