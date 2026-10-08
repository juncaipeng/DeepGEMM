# DeepGEMM 仓库系统梳理

> 版本：对应 `main` 分支 commit `78b6900`，`deep_gemm.__version__ = 2.8.0`
>
> 本文目标是"浅显易懂 + 详细"，从整体到模块逐层拆解 DeepGEMM 的设计与实现。
>
> **重要前提**：本 checkout 中 `third-party/deep_jit` 与 `third-party/cutlass` 两个 git submodule 未初始化（目录为空）。因此 DeepJIT 的内部实现（缓存哈希、nvcc 调用、模块加载）无法直接阅读，本文中 DeepJIT 的行为是根据其**调用点、include 路径和 README** 反推的。如需深入，先执行 `git submodule update --init third-party/deep_jit`。

---

## 1. 一句话定位

DeepGEMM 是一个**统一的、高性能的 GPU tensor core 内核库**，把现代大模型训练/推理中的关键算子集中到一个 CUDA 代码库里：

- 稠密 GEMM（FP8 / FP4 / BF16，含分组变体）
- MoE 融合通信的 Mega MoE（EP dispatch + L1/L2 + SwiGLU + combine 融合重叠）
- Lightning Indexer 的 MQA scoring（稠密 / paged / sparse）
- HyperConnection（HC / mHC）
- Mega Gate（router GEMM + top-k 路由融合）

设计特点：

1. **运行时 JIT 编译**（DeepJIT）：安装时不编译任何 CUDA kernel，首次调用时按 shape/config 动态生成并编译，结果落盘缓存。
2. **借鉴 CUTLASS/CuTe 概念但不重度依赖其模板体系**，刻意保持少量核心 kernel，便于学习。
3. 性能对标/超越专家手调库。

---

## 2. 整体架构

### 2.1 分层视图

```
┌─────────────────────────────────────────────────────────────┐
│ Python: deep_gemm/                                           │
│   ├── __init__.py      公开 API（re-export _C + utils）      │
│   ├── mega/            SymmBuffer / 权重变换 / mega 入口      │
│   ├── utils/           layout / math / dist 辅助             │
│   ├── testing/         bench / numeric / utils               │
│   └── legacy/          A100 Triton 老内核                     │
├─────────────────────────────────────────────────────────────┤
│ C++ host: csrc/  (编译进扩展 _C，只有 1 个 TU)                │
│   ├── apis/             pybind 绑定 + 校验 + 架构分发         │
│   ├── jit_kernels/                                              │
│   │   ├── heuristics/  config 搜索（block/stage/cluster）     │
│   │   └── impls/       生成 JIT 源码串 + 建 TMA + launch      │
│   ├── runtime/          jit.hpp (DeepJIT) / runtime.hpp (全局) │
│   ├── utils/            math / layout / exception             │
│   └── indexing/main.cu  仅给 IDE 索引，不参与扩展编译           │
├─────────────────────────────────────────────────────────────┤
│ Device CUDA: deep_gemm/include/deep_gemm/  (.cuh 模板)        │
│   ├── impls/      真正的 kernel 模板 *_impl<...>()             │
│   ├── layout/     smem 布局 / workspace / sym buffer          │
│   ├── scheduler/  tile 调度器 + 部分 metadata 预计算 kernel    │
│   ├── mma/        SM90 wgmma / SM100 tcgen05 封装             │
│   ├── ptx/        wgmma / tcgen05 / tma / ld_st 底层封装       │
│   ├── epilogue/   输出/存储优化                                │
│   └── comm/       grid sync / NVLink barrier                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流（一次调用的生命周期）

以 `deep_gemm.fp8_gemm_nt(a, b, d, ...)` 为例，端到端链路：

1. **Python 调用** → `deep_gemm/__init__.py:46` 导出的 `fp8_gemm_nt`
2. **别名绑定** → 它其实是 `fp8_fp4_gemm_nt` 的别名（`csrc/apis/gemm.hpp:876` 用 `m.attr("fp8_gemm_nt") = m.attr("fp8_fp4_gemm_nt")`）
3. **host 入口校验** → `csrc/apis/gemm.hpp:78-131`：推断 major、检查 shape/dtype、归一化 SF 布局（`transform_sf_pair_into_required_layout`）
4. **架构分发** → `jit->device.get_arch_major()`：SM90 走 `sm90_fp8_gemm_1d1d/1d2d`，SM100 走 `sm100_fp8_fp4_gemm_1d1d`
5. **构造描述 + 选配置** → impl 里建 `GemmDesc`，调 `get_best_config<SM100ArchSpec>(desc)`（`heuristics/common.hpp:18`）得到 `GemmConfig`
6. **建 TMA 描述符** → `impls/runtime_utils.hpp` 的 `make_tma_a_desc/b/cd/sf_desc`
7. **JIT 编译** → impl 里用 `std::format` 拼出一段极小的翻译单元（`#include <...cuh>` + 一个 `__instantiate_kernel()` 取模板实例地址），调 `jit->compile(tag, source)`；返回 `Kernel` 句柄
8. **缓存** → DeepJIT 以 (tag + 完整实例化源码 + toolchain) 为 key，落盘到 `DG_JIT_CACHE_DIR`（默认 `$HOME/.dj`）。命中则跳过 nvcc
9. **launch** → `jit->launch(kernel, options, 参数...)`，透传 raw ptr、dim、`CUtensorMap`、甚至设备端结构体（如 `layout::SymBuffer<>`）
10. **设备端执行** → `deep_gemm/include/deep_gemm/impls/xxx.cuh` 中的模板，用 persistent kernel + warp specialization 跑 tile 循环

这条链路解释了"为什么安装不需要 nvcc"：扩展本体只是一个薄薄的 host 层，真正的 CUDA 代码在运行时才被生成/编译。

---

## 3. 目录结构总览

仓库共约 154 个文件（不含 .git），核心分布：

| 路径 | 作用 |
|---|---|
| `csrc/` | C++ host 层，基本 header-only，唯一 TU 是 `python_api.cpp` |
| `deep_gemm/` | Python 包 + `include/deep_gemm/**`（真正的 `.cuh` kernel 源码） |
| `tests/` | 12 个功能/性能测试 |
| `docs/` | `scaling-factor-format.md`（SF 格式权威说明） |
| `scripts/` | `.pyi` 生成、ncu 脚本、绘图 |
| `third-party/` | `deep_jit`（JIT 框架）、`cutlass`、`tilelang_ops`（测试用参考实现） |

`csrc/` 细分：

- `csrc/apis/` — 9 个 pybind 绑定文件：`config / attention / einsum / hyperconnection / gemm / layout / mega_moe / mega_mhc / mega_gate`
- `csrc/jit_kernels/impls/` — 每个算子家族的 launch 封装（建 TMA + 拼 JIT 源码 + launch）
- `csrc/jit_kernels/heuristics/` — `config.hpp`（数据结构）+ `common.hpp`（搜索驱动）+ `sm90.hpp` / `sm100.hpp` / `mega_moe.hpp` / `mega_gate.hpp`（各架构候选/代价模型）
- `csrc/runtime/` — `jit.hpp` 持有 DeepJIT runtime；`runtime.hpp` 持有 SM 数、cuBLASLt hande 等进程级状态
- `csrc/utils/` — `layout.hpp`（major/dtype/SF 校验）、`math.hpp`、`exception.hpp`、`compatibility.hpp`

---

## 4. 构建与安装

### 4.1 安装流程

```bash
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM && ./develop.sh   # 开发：build + symlink .so 进 deep_gemm/
# 或 ./install.sh             # 打包 wheel 并 pip 安装
```

- `build.sh` → `python setup.py bdist_wheel`
- `install.sh` → build 后 `pip install dist/*.whl --force-reinstall`
- `develop.sh` → `python setup.py build`，再把 `build/` 下的 `.so` symlink 到 `deep_gemm/`；同时把 cutlass 的 `cute`/`cutlass` symlink 到 `deep_gemm/include`

### 4.2 setup.py 关键点

- 唯一源文件：`sources = ['csrc/python_api.cpp']`（`setup.py:32`），扩展名 `deep_gemm._C`
- include 路径（`setup.py:33-39`）：`$CUDA_HOME/include`、`cccl`、`deep_gemm/include`、`third-party/deep_jit/include`、`third-party/cutlass/include`
- `CustomBuildPy.run()`（`setup.py:112-126`）依次做四件事：
  1. `prepare_includes()`：把 cutlass `include/{cute,cutlass}` 拷进 build 树的 `deep_gemm/include`（这样运行时 JIT 的 `#include <cute/...>` 能解析）
  2. `generate_default_envs()`：把 build 时的 `DG_JIT_CACHE_DIR / PRINT_COMPILER_COMMAND / CPP_STANDARD` 烧进 `deep_gemm/envs.py`
  3. `generate_pyi_file()`：解析 `csrc/` 里的 `m.def(...)` 生成 `_C.pyi`
  4. `prepare_agent_files()`：把 `csrc/`、`docs/` 拷进 wheel 供 agent 查错
- `CachedWheelsCommand`（`:176-200`）：默认尝试从 GitHub Releases 下载预编译 wheel，除非设置 `DG_FORCE_BUILD=1` / `DG_USE_LOCAL_VERSION=1`
- 版本号解析自 `deep_gemm/__init__.py` 的 `__version__`

### 4.3 CMakeLists.txt

文件第一行注释即说明：**仅用于 CMake-based IDE（CLion）索引，真正编译由 JIT 完成**。其中 `csrc/indexing/main.cu` 只把所有 `.cuh` include 进来供 IDE 跳转，编成静态库但不进扩展。

---

## 5. DeepJIT 运行时机制（核心）

### 5.1 初始化

`csrc/runtime/jit.hpp`：

```cpp
inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);  // l.12

inline void init_jit(const std::string& library_root_path) {                // l.14
    const auto include_dir = library_root / "include";
    auto config = deep_jit::Config(
        library_root, "DG", "cutlass-" + std::to_string(CUTLASS_VERSION),
        {include_dir}, {"deep_gemm/"});                                     // l.17
    jit = deep_jit::LazyInit<...>([config] {                                 // l.24
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        runtime->default_compiler_options.nvcc_flags->emplace_back(
            "--diag-suppress=39,161,174,177,186,940");                       // l.26-27
        runtime->default_compiler_options.nvcc_flags->emplace_back(
            "--compiler-options=-Wno-deprecated-declarations,-Wno-abi");     // l.28-29
        return runtime;
    });
}
```

- `deep_gemm/__init__.py:111` 调 `_C.init(os.path.dirname(os.path.abspath(__file__)))`，绑定在 `csrc/apis/config.hpp:9-11`
- `Config(library_root, namespace_tag, toolchain_tag, include_dirs, extra_dirs)`：`"DG"` 作为缓存命名空间；toolchain tag 里带 `CUTLASS_VERSION`，避免不同 CUTLASS 版本缓存串味
- `LazyInit<T>`：懒初始化、线程安全的持有器（`runtime`、`heuristics_runtime` 也都用它）

反推出的 DeepJIT API 面：

| API | 用途 |
|---|---|
| `deep_jit::Runtime<deep_jit::CUDA>` | 编译 + 运行核心对象 |
| `.device.get_num_sms()` / `.get_arch_major()` | 设备信息 |
| `.compile(tag, source)` | 编译源码串，返回 `Kernel` 句柄 |
| `.launch(kernel, options, args...)` | 启动 kernel |
| `.default_compiler_options.nvcc_flags` | 默认 nvcc 参数（vector 风格） |
| `.default_launch_options.enable_pdl` | 默认 PDL 开关 |
| `deep_jit::get_env<T>(name, default)` | 环境变量解析 |
| `deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(...)` | 懒建 TMA 描述符 |
| `deep_jit::register_python_api(m, jit)` | 把 JIT 对象暴露到 `_C` |

### 5.2 编译与缓存

JIT 源码串的模板（以 SM100 FP8-FP4 为例）：

```cpp
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
using namespace deep_gemm;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<...模板参数...>);
}
```

- 模板参数（dtype 名、BLOCK_M/N/K、stages、threads、compiled_dims）全部在 host 侧用 `std::format` 代入源码串
- 缓存 key = `tag` + 完整实例化源码 + toolchain；因此不同 dtype/分块/架构各编译一份
- 缓存目录语义（README:164）：`DG_JIT_CACHE_DIR` 是 `:` 分隔的目录列表，**从前到后查找，首个命中生效；miss 时编译进第一个目录**，默认 `$HOME/.dj`
- `DG_JIT_*` 未设置时回退到全局 `DJ_JIT_*`（README:158）

关键环境变量分组（README:160-183）：

- 通用：`DG_JIT_DEBUG`、`DG_PRINT_CONFIGS`
- 缓存：`DG_JIT_CACHE_DIR`
- 编译器：`DG_JIT_NVCC_COMPILER`、`DG_JIT_CPP_STANDARD`（默认 20）
- 编译输出：`DG_JIT_PRINT_COMPILER_COMMAND`、`DG_JIT_PTXAS_VERBOSE`、`DG_JIT_CHECK_NO_SPILLS`、`DG_JIT_CHECK_NO_LOCAL_MEMORY`、`DG_JIT_PRINT_LOAD_TIME`
- 调试/剖析：`DG_JIT_WITH_LINEINFO`、`DG_JIT_DUMP_ASM/PTX/SASS`、`DG_COMM_KERNEL_DEBUG`、`DG_USE_NVIDIA_TOOLS`
- 构建：`DG_SKIP_CUDA_BUILD`、`DG_FORCE_BUILD`

### 5.3 另一个 Runtime（非 JIT）

`csrc/runtime/runtime.hpp:18` 的 `Runtime` 是进程级设备/调优状态，与 DeepJIT 的编译 runtime 分离：

- `num_sms` / `tc_util` 及其 setter（l.71-93），`tc_util` 默认 100
- 自管理 cuBLASLt handle + per-stream workspace（l.24-69），有 `DG_USE_PYTORCH_CUBLASLT_HANDLE`、`DG_USE_TEMP_CUBLASLT_WORKSPACE` 开关
- 单例 `inline auto runtime = deep_jit::LazyInit<Runtime>(...)`（l.97）

Heuristics 侧单例 `heuristics_runtime`（`heuristics/runtime.hpp:67`）持有：`ignore_compile_dims`、`deterministic_algorithms`、`block_m/n_multiple_of`、`mk_alignment_for_contiguous_layout`（l.10-65）。

### 5.4 内核生命周期抽象

| 阶段 | 抽象 | 位置 |
|---|---|---|
| 描述问题 | `GemmDesc` | `heuristics/config.hpp:12` |
| 选择配置 | `get_best_config<ArchSpec>` → `GemmConfig` | `heuristics/common.hpp:18` |
| 编译 | `jit->compile(tag, source)` → `Kernel` | impls |
| 启动 | `jit->launch(kernel, LaunchOptions, args...)` | impls |

`GemmDesc` 字段包含：`gemm_type`、`kernel_type`、`m/n/k/num_groups`、A/B/CD dtypes、`major_a/b`、`with_accumulation`、`num_sms`、`tc_util`、`compiled_dims`、`expected_*`、`ensure_zero_padding`。

`LaunchOptions`（`deep_jit::cuda::LaunchOptions`）含 `num_smem_bytes`、`grid_dim`、`block_dim`、`cluster_dim`、可选 `enable_pdl`。

**`compiled_dims`** 是重要机制：默认 `"nk"`，表示把 N、K 作为编译期常量特化进 kernel（`get_compiled_dim`，`runtime_utils.hpp:30`），未列入的维度传 0 作为运行期参数；可用 `set_ignore_compile_dims` 关闭。

---

## 6. 稠密 GEMM 内核体系

### 6.1 通用骨架

所有 GEMM kernel 共享同一套结构：

- 单个模板函数 `*_impl(...)` 作为 kernel body
- **warp specialization**：不同 warpgroup/warp 承担不同角色，用 mbarrier + named barrier 协作
- persistent tile loop 由 `sched::Scheduler::get_next_block(...)` 驱动
- 异步搬数（TMA）与计算（SM90 WGMMA / SM100 tcgen05 UMMA）解耦

### 6.2 SM90（Hopper）系列

**`sm90_fp8_gemm_1d1d.cuh`（364 行，入口 l.30）**：1D-1D 量化（A/B 每 128 通道一个 scale），纯 FP32 输出。

- 静态约束（l.51-56）：`kNumTMAThreads==128`、`kNumMathThreads%128==0`、`BLOCK_K==128`、`cd_dtype==float`
- MMA：`FP8MMASelector<BLOCK_N>`，即 Hopper `wgmma` FP8，M=64/K=32
- 角色：TMA producer warpgroup（`warp_idx >= kNumMathThreads/32`，l.170-248）+ math warpgroups（l.249-354）
- 数学侧：WGMMA → 寄存器内按 block scale 提升（`final_accum[i] += scale_a*scale_b*accum[i]`，l.318-326）→ TMA store FP32
- PDL：l.158 `cudaGridDependencySynchronize()`，与前置 kernel（如 SF transform）重叠

**`sm90_fp8_gemm_1d2d.cuh`（455 行，入口 l.49）**：`cd_dtype` 必须 bf16。B 的 scale 是 2D（逐 N 块），SFB 无法 TMA 加载，改由 math warp 从 global 直接读进 `smem_sfb`（l.239-250）。用 `stmatrix` 做 epilogue。

**`sm90_bf16_gemm.cuh`（401 行，入口 l.28）**：支持 `float`/`bfloat16` 输出，累积用 `SM90_TMA_REDUCE_ADD_*`。`kDoMergeStages`（l.48-57）：当 stages≥10、Normal、双方 K-major、128 math 线程时，合并 stage 增大 BLOCK_K，减少 `warpgroup_wait<0>` 停顿。

### 6.3 SM100（Blackwell）系列

**`sm100_bf16_gemm.cuh`（433 行，入口 l.22）**：关键编译期配置（l.66-88）：`LAYOUT_AD_M=128`、`UMMA_M=128*kNumMulticast`、`UMMA_K=16`、`BLOCK_K_=64`、`kNumEpilogueStages=2`。三角色：warp0 TMA、warp1 `elect_one_sync` 发 UMMA、epilogue warp 存 D。累加器在 **TMEM**（通过 `Allocator` 分配，l.146-149）而非寄存器。

**`sm100_fp8_fp4_gemm_1d1d.cuh`（580 行，入口 l.24）**：旗舰 block-scaled kernel。

- `kIsMXF4`（A/B 都是 FP4）、`UMMA_K = kIsMXF4 ? 64 : 32`、`kPackFactor = get_smem_pack_factor<dtype>()`（FP4 为 2）
- SF blocking（l.91-104）：`SF_BLOCK_M = align(BLOCK_M,128)`，`SF_BLOCK_N = align(BLOCK_N,128)`，`SF_BLOCK_K = BLOCK_K/128`
- 四类角色：
  1. TMA load（l.213-300）：A/B + 单独的 packed SF tile（`sf_full_barriers`）
  2. MMA-issue leader（l.301-459）：发 UTCCP（smem→TMEM 的 SF 拷贝）和 UMMA，用 `make_runtime_instr_desc_with_sf_id` 动态改 SF id
  3. UTCCP transposer（l.460-496）：`warp_idx==2`（SF_BLOCK_K==2 时再加 3），在 smem 内做 warp transpose 以匹配 UTCCP K-major 128-bit atom
  4. Epilogue（l.497-562）：TMEM→寄存器→smem→TMA store
- MMA：`SM100_MMA_MXF4_SS`（MXF4）或 `SM100_MMA_MXF8F6F4_SS`（l.412-414）

### 6.4 SM90 vs SM100 的本质差异

| 维度 | SM90 Hopper | SM100 Blackwell |
|---|---|---|
| 计算指令 | `wgmma.mma_async`（warpgroup 级） | `tcgen05.mma`/UMMA（单线程发起） |
| 累加器位置 | 寄存器 | Tensor Memory (TMEM) |
| 发起模型 | 128 线程 warpgroup commit | 一个 elected 线程（`elect_one_sync`） |
| SF 通路 | 作为 operand/寄存器 | UTCCP 把 smem SF 拷进 TMEM（block-scaled MMA） |
| 多 CTA | TMA multicast + cluster | 2-CTA 协同 MMA（`cta_group::2`） |
| Warp 角色 | TMA warpgroup + math warpgroups | TMA warp + MMA leader + UTCCP warp + epilogue warp |

### 6.5 支撑代码

- **`layout/gemm.cuh`**：`SM100BF16GemmSharedStorage`（l.11-35，含 `cd[]`/`a[]`/`b[]`、full/empty barriers、`tmem_ptr`）；`SM100FP8FP4GemmSharedStorage`（l.37-65，额外 `sfa[]/sfb[]`、`sf_full_barriers`，A/B stage 按 pack factor 缩小）
- **`scheduler/gemm.cuh`**（315 行）：见第 7 节
- **`mma/sm90.cuh`**（294 行）：`FP8MMA`/`FP8MMASelector`（M=64,K=32，N 从 8 到 256）、`BF16MMA*`、`TF32MMA*`、`make_smem_desc`、`make_gmma_desc<kMajorMode,...>`
- **`mma/sm100.cuh`**（160 行）：`make_smem_desc`（UMMA SmemDescriptor，version=1）、`make_sf_desc`（SWIZZLE_NONE，供 UTCCP）、`to_umma_layout_type`、`make_umma_desc<...>`、`make_runtime_instr_desc_with_sf_id`
- **`ptx/*.cuh`**：`tcgen05.cuh`（220 行，各种 `SM100_MMA_*` 结构）、`tma.cuh`（156 行）、`wgmma.cuh`（26 行）、`ld_st.cuh`（368 行，ldmatrix/stmatrix、cp_async、atomics）
- **`common/`**：`packing.cuh`（`get_smem_pack_factor`，FP4=2）、`math.cuh`（UE8M0 辅助 `get_ue8m0_sf_exp/inv`）、`tma_copy.cuh`（`tma::copy`，同时处理 SM90 multicast 与 SM100 2SM load）

### 6.6 host launch 封装

- `impls/sm90_fp8_gemm_1d1d.hpp`（229 行）：`SM90FP8Gemm1D1DRuntime::compile_and_launch`；host 函数 `sm90_fp8_gemm_1d1d()`（l.77-143）建 desc→选 config→建 TMA（A、SFA MN-major、CD）；另有 `sm90_k_grouped_fp8_gemm_1d1d()`（l.145-226）
- `impls/sm100_fp8_fp4_gemm_1d1d.hpp`（559 行）：6 个 host 函数——普通、M-grouped contiguous、M-grouped masked、K-grouped、`fp8_bmm`、K-grouped FP4
- `impls/runtime_utils.hpp`（285 行）：`get_compiled_dim`、`to_string` 重载、`make_tma_a/b/cd_desc`、`make_tma_sf_desc`
- heuristics：`common.hpp:18` 的 `get_best_config` 遍历候选 layout，用架构代价模型打分取最小 cycles

---

## 7. 分组 GEMM 与 Scaling Factor 布局

### 7.1 GemmType 分类

`common/types.cuh` 定义 `GemmType::{Normal, MGroupedContiguous, MGroupedMasked, KGroupedContiguous, Batched, MGroupedContiguousWithPsumLayout, KGroupedContiguousWithPsumLayout}`。

**M-grouped contiguous**（训练前向/推理 prefill）：每个 expert 的 token 拼成连续 M 段，`grouped_layout[m_block*BLOCK_M]` 给出该行的 group 号；每段需按 GEMM M block 对齐（`get_mk_alignment_for_contiguous_layout()`）。

**M-grouped masked**（推理 decode + CUDA graph）：`grouped_layout[group_idx]` 是该 group 的有效 M 行数（不是 offset）；scheduler 顺序遍历 group，`is_computation_valid` 判断 `m_offset + m_block_idx*BLOCK_M < grouped_layout[current_group_idx]`。

**K-grouped contiguous**：K 维按变长分组。`get_next_k_group()` 维护 `current_k_start/current_shape_k/current_k_end`；`get_next_block` 累积 `current_sf_k_cumsum += ceil_div(align(current_shape_k, kKAlignment), kSFKSpan)` 保持 SF 行索引同步。

**Batched**：`current_group_idx = next_block_idx / num_blocks`。

### 7.2 Scheduler（`scheduler/gemm.cuh`，315 行）

- `IndexType {MN, K, SF_K}`（l.8），`get_global_idx` 据此施加 group offset
- `get_num_1d_blocks_per_group<...>()`（l.14-26）：以 L1/L2 使用代价最小化选 8 或 16 blocks/group
- `Scheduler<GemmType, BLOCK_M/N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs, ...>`（l.30-310）
  - `get_swizzled_block_idx()`（l.108-144）：L2 友好的 (m,n) swizzle；SM90-only 的 odd-multicast 修正用 `#if __CUDA_ARCH__ < 1000` 隔离（SM90 可动态关 multicast，SM100 的 2-CTA 不行）
  - `get_next_block()`（l.185-272）：persistent grid 核心，每个 GemmType 一个分支
  - `is_tma_multicast_valid()` / `is_computation_valid()`：SM90-only 的合法性判断

### 7.3 SF 格式：SM90 FP32 vs SM100 packed UE8M0

详见 `docs/scaling-factor-format.md`。要点：

- **SM90**：FP32，每 128 通道一个（1x128x128），kernel 在寄存器里提升 scale
- **SM100**：packed UE8M0，8-bit exponent，**4 个打包进一个 int32**；SF 张量 MN-major，`stride(-2)==1`、`stride(-1)==align(mn,4)`，`packed_sf_k = ceil_div(k, gran_k*4)`；SF 值必须是 2 的精确幂（断言 `(value & 0x807fffff) == 0`）
- transform kernel：`transpose_and_pack_fp32_into_ue8m0`（`impls/smxx_layout.cuh:78`）、K-grouped 的 `pack_fp32_into_ue8m0`（l.195）
- host 分发：`transform_sf_into_required_layout`（`apis/layout.hpp:11`）；校验 `check_sf_layout()`（`utils/layout.hpp:96-133`）

---

## 8. MQA Logits / Lightning Indexer（`csrc/apis/attention.hpp`）

### 8.1 数学定义

对每个 query 行 q 与每个许可范围内的 KV token j：

```
logits[q,j] = sum_h weights[q,h] * ReLU(dot(q[h], kv[j]))
```

DeepGEMM 把 GEMM + ReLU + head 归约 + 可选清理（非法区间写 `-inf`）融进一个 kernel。输出 `[seq_len, seq_len_kv]`，供外部选 top-k KV。

### 8.2 三个变体

**非 paged / prefill（连续 KV）** — `fp8_fp4_mqa_logits`（`attention.hpp:78`）

- `q [seq_len, num_heads, head_dim]`、`kv [seq_len_kv, head_dim]`、`weights [seq_len, num_heads]`
- 每行 KV 边界来自 `cu_seq_len_k_start/end`
- `block_q = 128/num_heads`，`block_kv = 256`
- `max_seqlen_k == 0` → 全量 logits，清理融合在 kernel 内；否则输出压缩 logits 宽度 `max_seqlen_k`，单独 cleaner
- metadata：`get_mqa_logits_metadata`（l.182）
- 设备：SM100 `sm100_mqa_logits.cuh:552`（核心 l.63，`RingPipeline`+TMEM+三类 warp+cleaner warp）；SM90 `sm90_fp8_mqa_logits.cuh`（WGMMA FP8，寄存器重配 TMA=32/math=112，`epilogue::LogitsCleaner`）

**paged / decode（block-table 间接 KV cache）** — `fp8_fp4_paged_mqa_logits`（l.411）

- `q [batch, next_n, heads, head_dim]`、`fused_kv_cache [num_pages, page_kv, 1, head_dim_with_sf]`（SF 以 float 追加在每页数据后，host 用 `torch::from_blob` 切开）
- `context_lens`、`block_table`、varlen 用的 `indices`；`block_kv ∈ {32,64,128}`（SM100）、SM90 固定 64
- SM100 scheduler（`scheduler/sm100_paged_mqa_logits.cuh:232`）：`RequestInfo` 解析请求；metadata kernel（l.98）按 `ceil_div(context_len, SPLIT_KV)*num_q_tokens` 做前缀划分到各 SM；设备端 chunk-outer / Q-block-inner 遍历；只有持有请求起点的 SM 才能 split K
- SM90 scheduler：`scheduler/sm90_paged_mqa_logits.cuh:141`，单 warp metadata（l.11），varlen "atoms"

**sparse（每行给 sparse KV block 列表）** — `fp8_fp4_sparse_mqa_logits`（l.291）/ `fp8_fp4_paged_sparse_mqa_logits`（l.334）

- 约束：`num_max_sparse_blocks <= 4096` 且 `%4==0`；固定 `kNumHeads=32`、`kHeadDim=128`、`kBlockQ=2`
- JIT：`impls/sm100_sparse_mqa_logits.hpp`，`get_sparse_mqa_split_kv`：FP4→640，FP8→512
- 设备：`sm100_sparse_mqa_logits.cuh`，`KVAccessor` 抽象 Contiguous(TMA)/Paged 两种；把两个 Q token 的 sparse slot 打包进一个 16-bit metadata word
- metadata 调度：`scheduler/sm100_sparse_mqa_logits_metadata.cuh`，`balance_wave_entries`（l.77）按 KV-split 数排序并轮转重条目做负载均衡

### 8.3 类型支持

SM100 统一 FP8/MXFP4/MXFP8（FP8 复用未用的 `sf_q` 描述符槽，`sm100_mqa_logits.hpp:148`）；FP4 head_dim ∈ {64,128}，FP8 ∈ {32,64,128}；SM90 仅 FP8。legacy 包装 `fp8_mqa_logits` / `fp8_paged_mqa_logits` 在 `attention.hpp:549/561`。

---

## 9. HyperConnection（HC / mHC）

用**矩阵值（多路）残差**替代标量残差，并学习混合矩阵。`kNumRoutes=4`，混合函数产生 `kNumHCOutputs = 4*(4+2) = 24` 个输出。

### 9.1 `tf32_hc_prenorm_gemm`

- API（`apis/hyperconnection.hpp`）：`tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits)`
- `a [M,K]` bf16 K-major，`b [N,K]` fp32 K-major，`d [M,N]` fp32 N-major，`sqr_sum [M]` = 每行 `sum(a^2)`
- 可选 `num_splits` → split-K 累积，`d [num_splits,m,n]`
- 设备：`sm90_tf32_hc_prenorm_gemm.cuh`（bf16 A→fp32，边算 sqr_sum 边喂 TF32 WGMMA）；`sm100_tf32_hc_prenorm_gemm.cuh`（独立 MMA warp + cast/reduce warp，TMEM 存 A operand 和累加器）

### 9.2 `mega_mhc`（完整融合算子）

- API（`apis/mega_mhc.hpp:89`）：`x [T,H]`、`residual [T,hc_mult,H]`、`shifted_prev_mix/post_mix/comb_res_mix`、`fn [hc_mult*(hc_mult+2), hc_mult*H]` 及各种输出；可选 `y_bf16`/`y_fp8`（FP8 用 `gran_k=32`、4 UE8M0 packed int32）
- 区分 shifted / normal 两种模式
- 布局（`layout/mega_mhc.cuh`）：`kNumRoutes=4`、`kNumHCOutputs=24`、`BLOCK_M=64`、`BLOCK_N=24`、`BLOCK_K=64`、`kNumThreads=768`（6 warpgroups）、`kDefaultNumSplits=16`、`kNumMaxSplits=64`
- 调度（`scheduler/mega_mhc.cuh`）：`SplitBarrier`（Norm=0, Mix=1），用 `red_async_inc_rel`
- 设备（`sm100_mega_mhc.cuh`，705 行）：6 个静态角色 warpgroup —— WG0 TMA/MMA、WG1-2 Post、WG3 workspace-epilogue、WG4 Mix、WG5 Norm

---

## 10. Mega MoE

### 10.1 目标

传统 EP MoE 是 dispatch → grouped GEMM（L1→SwiGLU→L2）→ combine 三段串行。Mega MoE 把三段融进**一个 persistent kernel**，让后序 wave 的 dispatch 与前序 wave 的 GEMM 重叠、combine 与 GEMM 尾部重叠。跨 rank 传输用 **symmetric memory** + P2P NVLink，靠同一块对称 buffer 里的 flag 协调。

### 10.2 Python / buffer 抽象（`deep_gemm/mega/__init__.py`）

- `SymmBuffer`（l.18）：用 `torch.distributed._symmetric_memory` 分配一整块字节 buffer（`group.size()==1` 时退化到普通 torch），调 `_C.get_symm_buffer_size_for_mega_moe(...)` 拿 `(num_bytes, slice_fn)`，再切出 `x/x_sf/topk_idx/topk_weights/shared_*/l1_acts*/l2_acts*` 等命名视图；`base=` 参数允许小配置复用已有大 buffer
- `_interleave_weights`（l.112）：为 SwiGLU 交错 gate/up 两半；`_transpose_sf_for_utccp`（l.129）：SF 的 4×32 UTCCP 转置
- `transform_weights_for_mega_moe`（l.146）、`fp8_fp4_mega_moe`（l.168）、`bf16_mega_moe`（l.193）

### 10.3 Host API（`csrc/apis/mega_moe.hpp`）

- `get_symm_buffer_size_for_mega_moe`（l.34）：用 `sched::get_num_max_live_pool_blocks` 算最坏 ring 容量，构造 `layout::MegaMoEBuffer`，返回大小与 slice 回调
- `fp8_fp4_mega_moe`（l.154）/ `bf16_mega_moe`（l.287）：校验权重/SF/共享专家形状与 stats 计数后分发
- debug：`DG_COMM_KERNEL_DEBUG` 每次调用后清零对称 buffer

### 10.4 启发式（`heuristics/mega_moe.hpp`）

- `parse_mma_kind`（l.65）：`"bf16xbf16"/"fp8xfp4"/"fp8xfp8"`
- `get_block_config_for_mega_moe`（l.80）：按每专家期望 token 数（`num_tokens*num_ranks*num_topk/num_experts` + 1σ 余量）从 `kCandidateBlockM` 选 `block_m` 最小化每专家 M blocks；2-CTA cluster
- `get_pipeline_config_for_mega_moe`（l.116）：算 dispatch 区、C/D 输出 staging、task payload、SwiGLU amax 归约、TMEM 指针、每 stage 的 A/B/SF tile，选最大 stages

### 10.5 Symmetric buffer / 通信

- `layout/sym_buffer.cuh`：`SymBuffer<kNumRanks>` 存 base ptr + 每 rank offset；`map(ptr, dst_rank)` 把本地指针翻译成远端同 offset 地址（`kNumRanks==1` 时 no-op），`kNumMaxRanks=72`
- `layout/mega_moe.cuh`：`kCandidateBlockM={8,16,32,64,96,128,192,240}`、`kLCMCandidateBlockM=1920`；`MegaMoESignals`（l.53）含 grid-sync 计数、NVLink barrier、各阶段 task 计数、ring flags（`l1_full_count`、`l2_full_mask` 每 L1 N block 一位）；`MegaMoEBuffer`（l.344）顺序布局 workspace/输入/共享专家/路由 ring/combine buffer
- `comm/barrier.cuh`：`wait_until`（60s 超时）、`grid_sync<kNumSMs,...>`（原子计数 + 完成 tag）、`nvlink_barrier<kNumRanks,kNumSMs,...>`（grid sync → rank0 SM 跨 NVLink 交换带符号信号 → 再 grid sync；交替 phase/sign 使计数可复用）

### 10.6 设备 kernel（`sm100_fp8_fp4_mega_moe.cuh`，1484 行）

- A/B 始终 swap 且 K-major；`UMMA_M=256`（2-CTA）、`UMMA_N=BLOCK_M`、`UMMA_K=32`、`UMMA_BLOCK_K=128`
- l.280 `cudaGridDependencySynchronize()`：作为 CDP 的 secondary kernel
- Warp 角色：
  1. **Dispatch warps**（l.333）：读 `input_topk_idx` 统计每专家 token 数 → 原子累加发送计数与全局 offset → 把 (rank,token,topk) 源索引写进目标 rank 的对称 buffer → grid sync → SM0 把本次 grid index 推给所有 peer → NVLink barrier → **pull 循环**（l.424-610）：用 min-peeling 轮转选 rank，把 token 分块 TMA-1D 从远端拉到本地 send buffer 再 TMA-store 进本地 L1 ring，复制 SF（UTCCP token index 变换），写 combine 源 metadata
  2. **A-token TMA load warp**（l.680）：等 `l1_full_count`/`shared_l2_full_count`，2-CTA multicast 发 A+SFA
  3. **B-weight TMA load warp**（l.746）：同模式发权重+SFB
  4. **UMMA issue warp**（l.806，仅 leader CTA）：建 block-scaled instr desc、动态改 UMMA-N、UTCCP 拷 SFA/SFB 进 TMEM、发 `SM100_MMA_MXF8F6F4_2x1SM_SS::fma`
  5. **task scheduler mainloop**（l.927，仅 leader CTA）：`scheduler.mainloop(num_tokens)` 走 L1/L2 task 并广播
  6. **Epilogue warps**（l.934）：写 L1 激活（post-SwiGLU + amax 归约）或 L2 bf16 输出（直接 `*sym_buffer.map(dst_ptr, dst_rank) = packed` 写进 peer combine buffer，l.1307）
  7. **Combine**（l.1321-1476）：读 token 的 topk 槽，等对应专家 rank 的 `combine_ready_grid_idx == grid_idx`，双缓冲 TMA load 累加 fp32 → 转 bf16 → TMA store 到 `y`

### 10.7 Scheduler（`scheduler/mega_moe.cuh`，455 行）

- `get_num_l1_warmup_waves`（l.17）：证明最少 L1 warmup waves，避免 L1↔L2 ring 死锁（L2 的 M block 只在其 L1 任务发出后才可发）
- `get_num_max_live_pool_blocks`（l.48）：ring 容量闭式上界（host 侧用）
- `BlockPhase{Linear1, Linear2, SharedLinear1, SharedLinear2}`；`TaskInfo`（l.91）
- `L2KBlockDependency`（l.143）：64-bit mask，每 L1 N block 一位，L1 完成即 `red_xor_rel` 翻转，使 L2 K block 一旦输入就绪即可开始
- `MegaMoEScheduler`（l.180）：两段式 task-info 流水（producer cluster async store 发布，consumer 用 `task_info_empty_barriers` 释放）；`get_next_task()`（l.346）实现交错：先排空 `num_sched_l1_waves` 个 L1 task，再 L1/L2 交替

---

## 11. Mega Gate

### 11.1 目标

融合 router（gate）线性层 `logits = x @ weightᵀ` 与打分、bias/image-bias、expert map/复制、mask、强制随机、固定路由、top-k 选择、权重归一化。使 `[T,E]` logits 不必完整落 HBM（只经过紧凑 score scratch）。

### 11.2 API（`csrc/apis/mega_gate.hpp:85`）

- `x [T,H]` bf16、`weight [E,H]` bf16、`num_topk`、`use_shared_as_routed`、`num_shared_experts`、`routed_scaling_factor`、`ep_rank`、`scoring_func`（`sigmoid`/`sqrtsoftplus`/`identity`）、可选 `mask/bias/image_bias/image_token_mask/fix_routing_mask/to_physical_map/logical_count/unmapped_topk_idx/force_random`
- 排序用 **score+bias**；输出权重是 **未加 bias 的 score**，在 top-k 内归一化后乘 `routed_scaling_factor`
- 约束：`hidden%256==0`、`E<=512` 且 `E%4==0`、`num_topk<=32`、`num_physical_topk = num_topk+S <= 32`

### 11.3 启发式（`heuristics/mega_gate.hpp`）

`kNumMaxGateWarpgroups=7`、`kNumTopkTokensPerWarpgroup=4`、`kMegaGateMaxSplitK=8`、`kNumMaxRoutedExperts=512`；`select_wave_tile`（l.64）平衡每 wave token 数；`get_mega_gate_candidates`（l.106）枚举 `(num_expert_groups, num_mma_ctas∈{1,2}, num_split_k∈2的幂≤8)`；`compare_mega_gate`（l.157）偏好更多 token 列/更多 SM/更多 MMA CTA/更少 Split-K/更少 wave；确定性模式强制 `num_split_k=1`（l.206）

### 11.4 设备（`sm100_bf16_mega_gate.cuh`，483 行）

- `UMMA_M = kNumExpertsPerGroup`，`UMMA_N = BLOCK_TOKENS`（专家是 M 维，logits 是 `[tokens, experts]` 转置，A/B swap）
- `kNumLogicalCtas = kNumMmaCtas * kNumExpertGroups * kNumSplitK`
- 布局（`layout/mega_gate.cuh`）：`BLOCK_K=64`、`kNumMaxBlockTokens=256`、`kExpertAlignment=128`（专家 padding 到 128 的倍数以用满 UMMA tile）；`ScoringType{Sigmoid=0,SqrtSoftplus=1,Identity=3}`（刻意留空 2）
- Workspace：score scratch 视作 `[num_token_blocks, num_splits, kBlockTokens, kNumExperts]` + 每 token block 的 score barrier
- 流程：TMA 加载（EVICT_FIRST on x）→ UMMA → metadata cache（bias/image-bias/logical-count 拷进 smem）→ `run_gate`：score barrier 跨逻辑 CTA 同步 → TMEM score tile store 到 gmem scratch（`kScoreOnStore` 优化）→ load_scores 累加 Split-K → top-k：masked 写 `-1/0`、force_random 随机、fix_routing 用给定 index、主路径向量化读全部 expert score 加 bias、`-inf` 屏蔽、`select_warp_topk` 用 `__ballot` → 逻辑→物理 expert 重映射并归一化

---

## 12. Einsum（`csrc/apis/einsum.hpp`）

一组硬编码 Einstein 表达式的分发器（显式说明"TODO: 支持任意表达式"）。支持的表达式：

| 表达式 | 说明 | 实现 |
|---|---|---|
| `bmk,bnk->mn` | batched 归约 batch：`a[s,m,k] @ b[s,n,k]ᵀ` | `sm90/sm100_bmn_bnk_mn_gemm` |
| `bhr,hdr->bhd` | 逐 head `a[b,h,r] @ b[h,d,r]ᵀ` | 优先 cuBLASLt（确定性模式走 sm90/sm100 bf16） |
| `bhd,hdr->bhr` | `a[b,h,d] @ b[h,d,r]` | 同上 |
| `bhd,bhr->hdr` | → `[h,d,r]` fp32 | cuBLASLt（可累积） |

FP8 变体：`fp8_bmm`（`[B,M,K] @ [B,N,K]ᵀ`，per-32 packed UE8M0 SF，输出可为 bf16/fp32 或 fp8 `(d,sfd)`）、`fp8_einsum`（映射三个 head 表达式，仅 `bhr,hdr->bhd` 支持 fp8 输出）。

---

## 13. 测试体系

| 文件 | 覆盖内容 |
|---|---|
| `test_attention.py`（816 行） | `test_gemm_skip_head_mid`(l.34)、`test_mqa_logits`(l.125)、`test_paged_mqa_logits`(l.336)、`test_sparse_mqa_logits`(l.574) |
| `test_fp8_fp4.py` | FP8/FP4 稠密与分组 GEMM |
| `test_bf16.py` | BF16 GEMM 系列 |
| `test_layout.py` | SF 布局变换 / layout kernel |
| `test_einsum.py` | einsum 系列 |
| `test_hyperconnection.py`（57 行） | `test_hc_prenorm_gemm`(l.15) |
| `test_mega_moe.py`（455 行） | 分布式；融合 vs DeepEP dispatch + grouped GEMM + TileLang SwiGLU + combine 基线 |
| `test_mega_gate.py`（354 行） | `test_mega_gate`(l.227) 性能扫描 + `test_mega_gate_api_contract`(l.285) 契约/确定性 |
| `test_mega_mhc.py`（266 行） | mHC 正确性 + determinism/CUDA graph/多流 |
| `test_legacy.py` / `test_lazy_init.py` / `test_sanitizer.py` | 老内核 / 懒初始化 / sanitizer |

亮点测试实践：

- **位级一致性**：attention 20 次重跑、mega_gate 100 次、mega_mhc 30 次要 bitwise 相等
- **调度路径等价**：测试 SM100 scheduled 路径（metadata 驱动）与非 scheduled 路径 bitwise 相等
- **边界**：SM 数为 `num_sms-1/num_sms/num_sms+1/2*num_sms+1`；0 token；重复行
- **CUDA graph 语义**：mHC 要求 capture 前先 warmup（split-barrier init 有断言），graph replay 后 bitwise 相等
- 参考实现：`tests/generators.py`、`third-party/tilelang_ops/`（ref_mhc、swiglu、norm）

---

## 14. 跨模块关键机制

1. **CDP 链式启动**：Mega 系列 kernel 早调 `cudaGridDependencySynchronize()`（`sm100_fp8_fp4_mega_moe.cuh:280`、`sm100_bf16_mega_gate.cuh:130`、paged metadata kernel），作为 primary/secondary 链的一部分，在真正干活前做零拷贝设备端初始化/协调
2. **Grid sync + NVLink barrier**（`comm/barrier.cuh`）是 Mega MoE 跨 rank 重叠的骨架；交替 phase/sign 让计数器复用无需 reset
3. **Symmetric memory** 是 offset 在各 rank 一致的扁平字节 buffer，任何 rank 都能算 peer 地址；所有同步 flag 都在同一 buffer
4. **Task-based 调度器 + 交错**：Mega MoE 刻意先 warmup L1 再排 L2 避免死锁；Mega Gate 用 grid-stride 逻辑 token block
5. **MQA 调度三策略**：grid-stride（不切 K）、metadata 平衡（请求/前缀划分）、chunk-outer/Q-block-inner（SM 钳制 split 边界）；只有持有请求起点的 SM 能切 K
6. **UTCCP / UE8M0 无处不在**：128 元素对齐、4×32 转置 index 变换、4 个 UE8M0 打包进 int32
7. **Warp specialization 是统一范式**：producer（TMA）/ MMA issuer / UTCCP transposer / epilogue，用 mbarrier + named barrier 编排

---

## 15. SM90 vs SM100 支持矩阵

| 算子 | SM90 | SM100 |
|---|---|---|
| FP8 GEMM | 1d1d / 1d2d（仅 NT） | fp8_fp4 1d1d（NT/TN/NN/TT） |
| FP4 GEMM | 无 | 支持 |
| BF16 GEMM | 支持 | 支持 |
| MQA / sparse MQA | FP8（paged 固定 block_kv=64，next_n∈{1,2}） | FP8/MXFP4/MXFP8，block_kv∈{32,64,128} |
| HC / mHC | tf32_hc_prenorm | tf32_hc_prenorm + mega_mhc |
| Mega MoE / Mega Gate | 无 | 仅 SM100 |

**SF 布局是最尖锐的 SM90/SM100 分野**：SM90 FP32 per-128-channel vs SM100 packed UE8M0 int32（MN-major、4-per-int32、限 2 的幂）。

---

## 16. 阅读建议（按目的）

- **想理解 JIT 机制**：`csrc/runtime/jit.hpp` → `csrc/apis/config.hpp` → `csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`（看 compile_and_launch）
- **想学 kernel 优化**：先 `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh`（结构最清晰），再 `sm100_fp8_fp4_gemm_1d1d.cuh`（block-scaled 复杂版）
- **想懂配置搜索**：`csrc/jit_kernels/heuristics/common.hpp` → `sm90.hpp` / `sm100.hpp` 的 `get_layout_info`（代价模型）
- **想懂 SF 格式**：`docs/scaling-factor-format.md`（权威）→ `csrc/apis/layout.hpp`
- **想懂分布式融合**：`deep_gemm/mega/__init__.py` → `csrc/apis/mega_moe.hpp` → `include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` → `scheduler/mega_moe.cuh` → `comm/barrier.cuh`

---

## 附录：关键文件/行号速查

| 关注点 | 位置 |
|---|---|
| DeepJIT 初始化 / Config | `csrc/runtime/jit.hpp:12,14-32` |
| `_C.init` 绑定 | `csrc/apis/config.hpp:9-11` |
| 进程级 Runtime（SM/cuBLASLt） | `csrc/runtime/runtime.hpp:18,97` |
| Heuristics 单例 | `csrc/jit_kernels/heuristics/runtime.hpp:10,67` |
| 扩展构建（单 TU） | `setup.py:32,99-108`；`CMakeLists.txt:30` |
| JIT include / wheel 拷贝 | `setup.py:33-39,157-173` |
| 缓存环境烧入 wheel | `setup.py:148-155`；`deep_gemm/__init__.py:5-12` |
| `DG_JIT_CACHE_DIR` 语义 | `README.md:164` |
| `.pyi` 生成 | `setup.py:136-146`；`scripts/generate_pyi.py` |
| pybind 汇总 | `csrc/python_api.cpp:22-41` |
| GEMM 绑定/别名 | `csrc/apis/gemm.hpp:799-940`（别名 `:876`） |
| Desc/config 结构 | `csrc/jit_kernels/heuristics/config.hpp:12-188` |
| 最佳 config 搜索 | `csrc/jit_kernels/heuristics/common.hpp:18-56` |
| SM90/SM100 ArchSpec | `heuristics/sm90.hpp:13`；`heuristics/sm100.hpp:15` |
| compile+launch（SM90） | `impls/sm90_fp8_gemm_1d1d.hpp:31-75` |
| compile+launch（SM100） | `impls/sm100_fp8_fp4_gemm_1d1d.hpp:36-92` |
| TMA 描述符辅助 | `impls/runtime_utils.hpp:118-282` |
| GemmType 定义 | `include/deep_gemm/common/types.cuh` |
| Scheduler | `include/deep_gemm/scheduler/gemm.cuh:30-310` |
| SM90/SM100 MMA 封装 | `include/deep_gemm/mma/sm90.cuh`；`mma/sm100.cuh` |
| Mega MoE 设备 kernel | `include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` |
| Mega MoE 调度 | `include/deep_gemm/scheduler/mega_moe.cuh` |
| Mega Gate 设备 kernel | `include/deep_gemm/impls/sm100_bf16_mega_gate.cuh` |
| 通信 barrier | `include/deep_gemm/comm/barrier.cuh` |
| SymBuffer | `include/deep_gemm/layout/sym_buffer.cuh` |
