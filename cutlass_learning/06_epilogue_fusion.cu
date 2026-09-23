// ============================================================
// CUTLASS 学习 06: Epilogue 融合 (bias/激活/scale)
// ------------------------------------------------------------
// 目标架构: sm_80 / sm_89
// 学习目标: 把 bias/激活/scale 融进 GEMM 尾声, 减少 kernel 启动
//
// 参考: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html
//
// 编译:
//   CUTLASS=~/cutlass
//   nvcc -O2 -std=c++17 -arch=sm_89 \
//        -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//        cutlass_learning/06_epilogue_fusion.cu -o out && ./out
// ============================================================

#include <cstdio>

int main() {
    printf("CUTLASS 学习 06: Epilogue 融合 (bias/激活/scale) (骨架, 待补全)\n");
    // TODO(学习): 见文件头目标与参考链接, 逐步实现
    return 0;
}
