# cuBLAS 学习模块

> cuBLAS = **CUDA Basic Linear Algebra Subroutines**
> NVIDIA 官方**闭源**线性代数库，提供开箱即用、手工优化到极致的 BLAS 例程（向量/矩阵运算）。
> 与 CUTLASS 互补：**cuBLAS 是成品（直接调），CUTLASS 是可修改开发的模板库（自己造）**。

## cuBLAS vs CUTLASS

| 维度 | cuBLAS | CUTLASS |
|------|--------|---------|
| 本质 | 官方闭源成品库 | 开源可修改的 C++ 模板库 |
| 用法 | 调 API（如 `cublasSgemm`） | 用模板拼自己的 kernel |
| 灵活性 | 固定运算，不可改 | 可定制类型/tile/算子融合 |
| 学习成本 | 低 | 高 |
| 定位 | **性能标尺 + 生产直接用** | 定制 / 研究 / 深入优化 |

> 学习策略：**先用 cuBLAS 当"黄金对照"跑分，再学 CUTLASS 逼近并理解它。**

## 环境（本机 RTX 4090）

cuBLAS 随 CUDA Toolkit 13.0 自带，无需单独安装。编译时链接 `-lcublas`：

```bash
nvcc -O2 -std=c++17 -arch=sm_89 cublas_learning/01_cublas_hello.cu -o cublas_hello -lcublas
./cublas_hello
```

---

## 学习路线（渐进式）

| 阶段 | 文件 | 主题 | 参照函数 | 学到什么 |
|------|------|------|----------|----------|
| 01 | `01_cublas_hello.cu` | 句柄与第一次调用 | `cublasCreate` / `cublasDestroy` | handle 生命周期、列主序约定 |
| 02 | `02_level1_vector.cu` | Level-1 向量运算 | `cublasSaxpy` / `cublasSdot` / `cublasSnrm2` | 向量 axpy、点积、范数 |
| 03 | `03_level2_matvec.cu` | Level-2 矩阵×向量 | `cublasSgemv` | 矩阵-向量乘、转置参数 |
| 04 | `04_level3_gemm.cu` | Level-3 矩阵乘 (核心) | `cublasSgemm` | GEMM、alpha/beta、leading dimension |
| 05 | `05_gemm_ex_mixed.cu` | 混合精度 GEMM | `cublasGemmEx` | FP16/TF32 输入、FP32 累加、算法选择 |
| 06 | `06_batched_gemm.cu` | 批量 GEMM | `cublasSgemmBatched` / `StridedBatched` | 小矩阵批处理（深度学习常用） |
| 07 | `07_benchmark_gemm.cu` | 性能基准 | 上述 + Event 计时 | 测 TFLOPS/有效带宽，供 CUTLASS 对照 |

> ⚠️ **列主序陷阱**：cuBLAS 沿用 Fortran/BLAS 的**列主序 (column-major)**，
> 而 C/C++ 数组是**行主序**。这是新手最常踩的坑。
> 常用技巧：把行主序的 `A*B` 通过 `B^T * A^T` 或调整 `op`/`ld` 参数来算。各文件会具体演示。

## 常用函数速查

| 函数 | 作用 | BLAS 级别 |
|------|------|-----------|
| `cublasSaxpy` | y = a·x + y | 1 |
| `cublasSdot` | 点积 x·y | 1 |
| `cublasSnrm2` | 向量二范数 | 1 |
| `cublasSgemv` | y = α·A·x + β·y | 2 |
| `cublasSgemm` | C = α·A·B + β·C | 3 |
| `cublasGemmEx` | 混合精度 GEMM | 3 |
| `cublasSgemmStridedBatched` | 批量 GEMM | 3 |

> 前缀含义：`S`=float, `D`=double, `C`=complex, `Z`=double complex, `H`=half。

## 参考

- cuBLAS 官方文档：https://docs.nvidia.com/cuda/cublas/
- cuBLAS API 索引（所有函数）：https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference
- CUDA Samples (cuBLAS 示例)：https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS
