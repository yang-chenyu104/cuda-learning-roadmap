// ============================================================
// CUTLASS 学习 04: 手写 CuTe GEMM mainloop
// ------------------------------------------------------------
// 目标架构: sm_80 / sm_89 (Ampere/Ada)
// 学习目标: cute::copy / cute::gemm, shared memory 流水线 mainloop
//
// 参考: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
//
// 编译:
//   CUTLASS=~/cutlass
//   nvcc -O2 -std=c++17 -arch=sm_89 \
//        -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//        cutlass_learning/04_cute_copy_gemm.cu -o out && ./out
// ============================================================

#include <cstdio>

int main() {
    printf("CUTLASS 学习 04: 手写 CuTe GEMM mainloop (骨架, 待补全)\n");
    // TODO(学习): 见文件头目标与参考链接, 逐步实现
    return 0;
}
