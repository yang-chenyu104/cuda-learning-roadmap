// ============================================================
// CUTLASS 学习 01: 环境搭建 + 第一个 GEMM
// ------------------------------------------------------------
// 学习目标:
//   1. 配置 CUTLASS header-only 库的 include 路径
//   2. 用 device-level API 跑通一个 FP32 GEMM (C = alpha*A*B + beta*C)
//   3. 验证结果正确性
//
// 目标架构: 通用 (本示例用 SIMT FP32, RTX 4090 sm_89 可直接跑)
//
// 编译 (假设 cutlass 克隆在 ~/cutlass):
//   CUTLASS=~/cutlass
//   nvcc -O2 -std=c++17 -arch=sm_89 \
//        -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//        cutlass_learning/01_setup_and_first_gemm.cu -o first_gemm
//   ./first_gemm
//
// 参考:
//   https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html
//   https://github.com/NVIDIA/cutlass/discussions/1742
// ============================================================

#include <cstdio>

// TODO(学习): 取消注释并补全，需要先 clone CUTLASS 并配好 -I 路径
// #include <cutlass/gemm/device/gemm.h>
// #include <cutlass/util/host_tensor.h>
// #include <cutlass/util/reference/host/gemm.h>

int main() {
    printf("CUTLASS 学习 01: 第一个 GEMM\n");
    printf("--------------------------------------------\n");
    printf("步骤:\n");
    printf("  1. git clone https://github.com/NVIDIA/cutlass ~/cutlass\n");
    printf("  2. 取消本文件顶部 #include 的注释\n");
    printf("  3. 定义 cutlass::gemm::device::Gemm<...> 类型\n");
    printf("  4. 分配 A/B/C, 调用 gemm_op(args), 与参考实现对比\n");
    printf("\n请按 TODO 逐步补全。参考 README 与官方文档链接。\n");

    // TODO(学习): 定义 GEMM 类型并执行, 例如:
    //   using Gemm = cutlass::gemm::device::Gemm<
    //       float, cutlass::layout::RowMajor,   // A
    //       float, cutlass::layout::RowMajor,   // B
    //       float, cutlass::layout::RowMajor>;  // C
    //   Gemm gemm_op;
    //   Gemm::Arguments args({M,N,K}, {dA,lda}, {dB,ldb}, {dC,ldc}, {dC,ldc}, {alpha,beta});
    //   cutlass::Status st = gemm_op(args);

    return 0;
}
