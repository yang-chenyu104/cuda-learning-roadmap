// ============================================================
// CUTLASS 学习 02: CuTe Layout 基础
// ------------------------------------------------------------
// 学习目标:
//   1. 理解 CuTe 的核心抽象 Layout = (Shape, Stride)
//   2. 用 Layout 把逻辑坐标映射到物理内存偏移
//   3. 掌握 print_layout / print 观察布局
//
// 为什么重要:
//   CUTLASS 3.x 起整个 GEMM 层次都建立在 CuTe 之上。
//   Layout 是理解 tiling、合并访问、bank conflict 的统一语言。
//
// 目标架构: 通用 (纯 host 端演示 Layout 概念, 也可 device 端使用)
//
// 编译:
//   CUTLASS=~/cutlass
//   nvcc -O2 -std=c++17 -arch=sm_89 \
//        -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//        cutlass_learning/02_cute_layout_basics.cu -o cute_layout
//   ./cute_layout
//
// 参考:
//   https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
// ============================================================

#include <cstdio>

// TODO(学习): 配好 CUTLASS include 后取消注释
// #include <cute/tensor.hpp>
// using namespace cute;

int main() {
    printf("CUTLASS 学习 02: CuTe Layout 基础\n");
    printf("--------------------------------------------\n");
    printf("核心概念: Layout = (Shape, Stride)\n");
    printf("  逻辑坐标 (i, j)  --Layout-->  物理偏移 offset\n");
    printf("  行主序 4x4: Shape=(4,4) Stride=(4,1)\n");
    printf("  列主序 4x4: Shape=(4,4) Stride=(1,4)\n\n");

    // TODO(学习): 取消注释体验 CuTe Layout
    //   auto layout = make_layout(make_shape(Int<4>{}, Int<4>{}),
    //                             make_stride(Int<4>{}, Int<1>{}));
    //   print_layout(layout);            // 打印布局网格
    //   printf("offset(2,3) = %d\n", (int)layout(2,3));  // = 2*4+3*1 = 11

    printf("请按 TODO 补全，观察 print_layout 输出，\n");
    printf("尝试改变 Stride 观察行主序/列主序差异。\n");
    return 0;
}
