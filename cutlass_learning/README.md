# CUTLASS 学习模块

> CUTLASS = **CUDA Templates for Linear Algebra Subroutines**
> NVIDIA 开源的 C++ 模板库（现也含 Python CuTe DSL），用于编写达到 cuBLAS 级别性能的 GEMM / 卷积等线性代数 kernel。
> 是从"手写 CUDA kernel"进阶到"生产级高性能 kernel"的关键一环。

## 为什么学 CUTLASS

- 手写 Tensor Core（WMMA/MMA）kernel 极其繁琐，CUTLASS 用模板把 tiling、流水线、异步拷贝等最佳实践封装好
- CUTLASS 3.x 起以 **CuTe**（Layout/Tensor 抽象）为核心，理解 CuTe 是理解现代 CUDA 高性能编程的基础
- PyTorch、FlashAttention、各类推理引擎底层大量借鉴 / 使用 CUTLASS

## 前置要求

完成本仓库 Week 1~6（尤其是共享内存、tiled matmul、Tensor Core/WMMA、异步拷贝）后再学效果最佳。

## 环境说明（针对本机 RTX 4090）

| 项目 | 说明 |
|------|------|
| GPU | RTX 4090 (Ada Lovelace, **sm_89**) |
| CUTLASS 版本 | 建议 4.x（`git clone https://github.com/NVIDIA/cutlass`） |
| 编译 | header-only 库，`nvcc` 时用 `-I<cutlass>/include -I<cutlass>/tools/util/include` |

> ⚠️ **架构注意**：CUTLASS 很多前沿教程针对 **Hopper (sm_90, WGMMA)** 或 **Blackwell (sm_100, Tensor Memory)**，
> 这些指令 **RTX 4090 (sm_89) 不支持**。4090 上应聚焦 **CuTe 基础 + Ampere/Ada 的 `cp.async` + `mma` 路径**（sm_80/sm_89 tile）。
> 学习时留意教程标注的目标架构。

---

## 学习路线（渐进式）

| 阶段 | 文件 | 主题 | 目标架构 | 学到什么 |
|------|------|------|----------|----------|
| 01 | `01_setup_and_first_gemm.cu` | 环境搭建 + 第一个 GEMM | 通用 | clone/include 配置，用 device-level API 跑通一个 GEMM |
| 02 | `02_cute_layout_basics.cu` | CuTe Layout 基础 | 通用 | `cute::Layout`、shape/stride、`print_layout`，理解逻辑坐标→物理坐标 |
| 03 | `03_cute_tensor_tiling.cu` | CuTe Tensor 与分块 | 通用 | `make_tensor`、`local_tile`、把全局内存切成 CTA tile |
| 04 | `04_cute_copy_gemm.cu` | 手写 CuTe GEMM mainloop | sm_80/sm_89 | `cute::copy` / `cute::gemm`，shared memory 流水线 |
| 05 | `05_collective_builder.cu` | CollectiveBuilder 三层 API | sm_80/sm_89 | Device / Kernel / Collective 层次，用 builder 自动选优 |
| 06 | `06_epilogue_fusion.cu` | Epilogue 融合 | sm_80/sm_89 | 把 bias/激活/scale 融进 GEMM 尾声，减少 kernel 启动 |
| 07 | `07_profiling_vs_cublas.cu` | 性能对比 cuBLAS | 通用 | 用 CUTLASS profiler / 自测，对比 cuBLAS 有效算力 |

> 各文件当前为**学习骨架**（含目标、参考链接、TODO），随学习进度逐步填充实现。

---

## 官方与优质参考

- CUTLASS 仓库：https://github.com/NVIDIA/cutlass
- CuTe 稠密 GEMM 教程（官方，**从这里开始**）：https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html
- CUTLASS 3.x GEMM API（Device/Kernel/Collective 分层）：https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html
- CUTLASS 总览：https://docs.nvidia.com/cutlass/latest/overview.html
- Colfax Research — Hopper WGMMA 教程系列（sm_90，进阶了解）：https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/
- Colfax Research — Blackwell Tensor Memory 教程（sm_100，前沿了解）：https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/

## 编译示例

```bash
# 假设 cutlass 克隆在 ~/cutlass
CUTLASS=~/cutlass
nvcc -O2 -std=c++17 -arch=sm_89 \
     -I$CUTLASS/include -I$CUTLASS/tools/util/include \
     cutlass_learning/01_setup_and_first_gemm.cu -o first_gemm
./first_gemm
```
