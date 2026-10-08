# DeepGEMM 编译与安装原理方案

> 本文目标是**把 "DeepGEMM 是怎么被编译、怎么被安装、安装后 kernel 又是怎么来的" 讲清楚**。
> 阅读顺序建议：先看第 1 章建立整体心智模型，再看第 2~5 章（安装期）和第 6 章（运行期），
> 最后看第 7~10 章（环境变量、端到端流程、排错、设计动机）。

---

## 1. 一句话总览：两阶段构建模型

DeepGEMM 的构建**不是一次性的**，而是分成两个完全独立的阶段：

| 阶段 | 时机 | 编译什么 | 谁来做 | 产物 |
| --- | --- | --- | --- | --- |
| **阶段一：安装期** | `pip install` / `python setup.py build` | **只编译宿主侧（host）的薄扩展**：`csrc/python_api.cpp` 一个翻译单元 + 少量 `.cu`（indexing） | `setup.py`（setuptools + `torch.utils.cpp_extension.CUDAExtension`） | `deep_gemm/_C.cpython-*.so`（Python 扩展）、`.pyi` 存根、打包好的头文件/csrc/docs |
| **阶段二：运行期** | 进程**首次**调用某个算子时（惰性） | **真正的 device kernel**：把 C++ 模板源码拼成一个 `.cu` 字符串，用 `nvcc` 现场编译成 `cubin` | DeepJIT（纯头文件的 C++ JIT 运行时，`third-party/deep_jit`） | `<cache>/cache/<tag>.<digest>/{kernel.cu, kernel.cubin, meta.json, .committed}` |

**核心思想**：

- 安装期**不编译任何 GEMM/MoE kernel**，所以安装非常快（几秒~几十秒，取决于宿主扩展）。
- kernel 是**按需生成**的。同一个算子在不同 `(M, N, K, block 配置, 架构, 编译选项...)` 下是**不同的源码**，首次用到才编译，编译结果按内容哈希落盘缓存，后续命中直接加载。
- 因此仓库里 `deep_gemm/include/deep_gemm/impls/*.cuh` 这些文件**不参与扩展编译**，它们是被运行期 JIT 源码串 `#include` 进去的设备端模板。

**一句话理解**：安装期给的是"编译器 + 模板库 + 调度逻辑"，运行期才用它们"打印"出具体 kernel 并编译。

```
┌────────────────────── 安装期（setup.py，只做一次）──────────────────────┐
│  csrc/python_api.cpp ──nvcc/g++──► deep_gemm/_C...so                     │
│  （pybind11 绑定 + 所有 host 侧调度逻辑；不含任何 GEMM kernel）           │
│  另外打包：deep_gemm/include/**（含 cutlass/cute 头）、csrc/、docs/、.pyi │
└──────────────────────────────────────────────────────────────────────────┘
                                   │ import deep_gemm
                                   ▼
┌────────────────────── 运行期（DeepJIT，每次首次调用）────────────────────┐
│  Python 调用 fp8_gemm_nt(...)                                            │
│    → csrc/apis 校验/dtype 分发                                            │
│    → csrc/jit_kernels/heuristics 选 block/stage/cluster 配置              │
│    → csrc/jit_kernels/impls 拼出 kernel 源码串（#include 模板 .cuh）      │
│    → deep_jit::Runtime::compile(tag, source)                             │
│         ├─ 内存缓存命中？→ 直接返回                                        │
│         ├─ 磁盘缓存命中？→ 直接 load                                       │
│         └─ 否则 nvcc → cubin → 原子提交到缓存                              │
│    → deep_jit::Runtime::launch(kernel, launch_opts, args...)             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 安装期（一）：`setup.py` 的全局开关与版本号

文件：[setup.py](setup.py)

### 2.1 三个环境变量开关

```python
DG_SKIP_CUDA_BUILD   = int(os.getenv('DG_SKIP_CUDA_BUILD', '0')) == 1   # 默认 0
DG_FORCE_BUILD       = int(os.getenv('DG_FORCE_BUILD', '0')) == 1       # 默认 0
DG_USE_LOCAL_VERSION = int(os.getenv('DG_USE_LOCAL_VERSION', '1')) == 1 # 默认 1  ← 注意
```

| 变量 | 作用 |
| --- | --- |
| `DG_SKIP_CUDA_BUILD=1` | `get_ext_modules()` 返回 `[]`，**完全不编译 `_C` 扩展**。用于"只打包纯 Python/头文件"的场景（此时 import 会失败，因为 `from . import _C` 找不到）。 |
| `DG_FORCE_BUILD=1` | 强制走源码构建，**禁止下载预编译 wheel**。 |
| `DG_USE_LOCAL_VERSION=1` | **默认就是 1**。它同时影响两件事：(1) 版本号带本地 git revision；(2) `CachedWheelsCommand` 直接走本地构建、不下载 wheel。 |

> **重要结论**：由于 `DG_USE_LOCAL_VERSION` 默认是 `1`，**从源码 checkout 后直接 `pip install .` / `python setup.py bdist_wheel`，默认行为永远是"本地源码编译"**，不会去 GitHub Releases 下载预编译 wheel。
> 只有显式 `DG_USE_LOCAL_VERSION=0`（且 `DG_FORCE_BUILD=0`）时，才会尝试下载。

### 2.2 版本号生成 `get_package_version()`（[setup.py:51](setup.py#L51)）

1. 从 `deep_gemm/__init__.py` 里正则抓 `__version__`（当前是 `2.8.0`，[deep_gemm/__init__.py:113](deep_gemm/__init__.py#L113)）作为 `public_version`。
2. 若 `DG_USE_LOCAL_VERSION=1`：
   - 先跑 `git status --porcelain`，**工作区不干净则 `assert False`**；
   - 再跑 `git rev-parse --short HEAD`，得到 `+<hash>`；
   - 任何异常（包括上面的 assert）都会被 `except` 吞掉，退化成 `+local`。
3. 最终版本形如 `2.8.0+78b6900` 或 `2.8.0+local`。

这就是为什么在脏工作区里构建会看到 `Warning: Git working directory is not clean`，然后版本号变成 `+local`。

### 2.3 扩展的源文件与 include 路径（[setup.py:32-45](setup.py#L32-L45)）

```python
sources = ['csrc/python_api.cpp']          # ← 扩展本体只有这一个翻译单元
build_include_dirs = [
    f'{CUDA_HOME}/include',
    f'{CUDA_HOME}/include/cccl',
    'deep_gemm/include',                    # 设备端模板 .cuh 所在
    'third-party/deep_jit/include',         # DeepJIT 头文件
    'third-party/cutlass/include',          # CUTLASS
]
third_party_include_dirs = [                 # 这些目录会被"复制"进 wheel，而不是编译进去
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
]
build_libraries = ['cudart']                 # host 侧只链接 cudart（kernel 运行期用 driver API 加载）
```

注意 `sources` 只有 `csrc/python_api.cpp`。**所有 GEMM 逻辑都以 `#include` 头文件的形式被包含进这一个 TU**，编译出来就是 `deep_gemm._C`。这就是"薄扩展"。

### 2.4 `get_ext_modules()`（[setup.py:99](setup.py#L99)）

```python
CUDAExtension(name='deep_gemm._C',
              sources=['csrc/python_api.cpp'],
              include_dirs=build_include_dirs,
              libraries=['cudart'],
              library_dirs=[f'{CUDA_HOME}/lib64'],
              extra_compile_args=cxx_flags)
```

其中 `cxx_flags` 是：

```
-std=c++20  -O3  -fPIC
-Wno-psabi  -Wno-deprecated-declarations
-D_GLIBCXX_USE_CXX11_ABI=<torch 编译时用的值>
```

- `-std=c++20`：因为用了 C++20 的 `std::format`、concepts 等。
- `-D_GLIBCXX_USE_CXX11_ABI=...`：与 torch 的 ABI 对齐，避免 `std::string` ABI 不匹配导致的符号问题。
- 若 `DG_SKIP_CUDA_BUILD=1`，这里直接返回 `[]`。

---

## 3. 安装期（二）：`CustomBuildPy` 的四步定制

文件：[setup.py:111-173](setup.py#L111-L173)

`build_py` 被替换成 `CustomBuildPy`，在标准 Python 打包之前插入 4 个动作：

```python
class CustomBuildPy(build_py):
    def run(self):
        self.prepare_includes()       # 1. 复制 cutlass/cute 头到 build 目录
        self.generate_default_envs()  # 2. 生成 deep_gemm/envs.py
        self.generate_pyi_file()      # 3. 生成并复制 _C.pyi 存根
        self.prepare_agent_files()    # 4. 复制 csrc/ 和 docs/ 到 wheel 里
        build_py.run(self)            # 5. 标准打包
```

### 3.1 `prepare_includes()`：把 CUTLASS 头复制进 wheel（[setup.py:157](setup.py#L157)）

把 `third-party/cutlass/include/cute` 和 `.../cutlass` 两个目录，复制到 `build_lib/deep_gemm/include/{cute,cutlass}`。

- 目的：**运行期 JIT 需要这些头文件**。JIT 编译时 `nvcc` 会加 `--include-path <deep_gemm>/include`，而模板 `.cuh` 里 `#include <cute/...>`、`#include <cutlass/...>` 就能找到。
- 所以虽然安装期不编译 kernel，但**必须把整套 CUTLASS 头文件随 wheel 发出去**（这就是 wheel 体积的主要来源）。
- 复制前会 `shutil.rmtree` 目标，保证幂等。

包内最终布局（`package_data`，[setup.py:209-215](setup.py#L209-L215)）：

```
deep_gemm/
  include/deep_gemm/**/*     # 设备端模板 + host 侧头
  include/cute/**/*
  include/cutlass/**/*
```

### 3.2 `generate_default_envs()`：把安装时的环境变量固化（[setup.py:148](setup.py#L148)）

在 `build_lib/deep_gemm/envs.py` 写一个很小的文件：

```python
# Pre-installed environment variables
persistent_envs = dict()
persistent_envs['DG_JIT_CACHE_DIR'] = '...'          # 仅当构建时该变量存在
persistent_envs['DG_JIT_PRINT_COMPILER_COMMAND'] = '...'
persistent_envs['DG_JIT_CPP_STANDARD'] = '...'
```

只有这三个键（`DG_JIT_CACHE_DIR`、`DG_JIT_PRINT_COMPILER_COMMAND`、`DG_JIT_CPP_STANDARD`）会被固化。

这些值在 import 时被回填（[deep_gemm/__init__.py:5-12](deep_gemm/__init__.py#L5-L12)）：

```python
from .envs import persistent_envs
for key, value in persistent_envs.items():
    if key not in os.environ:      # 运行时的显式设置优先
        os.environ[key] = value
```

**意义**：允许打包者把"构建环境里配置好的" JIT 行为带到用户环境，同时用户仍可用运行时环境变量覆盖。`try/except ImportError` 保证源码直接使用时（没有 envs.py）不报错。

### 3.3 `generate_pyi_file()`：生成类型存根（[setup.py:136](setup.py#L136)）

调用 [scripts/generate_pyi.py](scripts/generate_pyi.py) 的 `generate_pyi_file(name='_C', root='./csrc', output_dir='./stubs')`：

- 扫描 `csrc/` 下的 `PYBIND11_MODULE` / `m.def(...)` 语句；
- 用自带的括号跟踪器解析 C++ 函数签名，把类型映射成 Python 类型、默认值映射成 Python 默认值；
- 输出 `stubs/_C.pyi`，再复制到 `build_lib/deep_gemm/_C.pyi`。

`_C.pyi` 让 IDE / 类型检查器能对 `deep_gemm.xxx(...)` 给出补全和签名提示（因为 `_C` 是 C++ 扩展，没有 Python 源码可分析）。**它纯粹是开发体验产物，不影响运行。**

### 3.4 `prepare_agent_files()`：把源码塞进 wheel（[setup.py:128](setup.py#L128)）

把仓库根目录的 `csrc/`、`docs/` 整个复制到 `build_lib/deep_gemm/` 下：

```python
for name in ('csrc', 'docs'):
    dst = os.path.join(package_dir, name)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(os.path.join(current_dir, name), dst)
```

**为什么要把 C++ 源码和文档打进 wheel？** 注释写得很直白：*"Copy csrc and docs into the wheel for agent-side error lookup"*——方便 AI/agent 在报错时拿到 C++ 源码和文档进行定位。这是为智能排错场景专门设计的。

---

## 4. 安装期（三）：`setuptools.setup()` 与"下载 or 构建"

### 4.1 主入口（[setup.py:203-222](setup.py#L203-L222)）

```python
setuptools.setup(
    name='deep_gemm',
    version=get_package_version(),
    packages=find_packages('.'),
    package_data={'deep_gemm': ['include/deep_gemm/**/*',
                                'include/cute/**/*',
                                'include/cutlass/**/*']},
    ext_modules=get_ext_modules(),
    zip_safe=False,
    cmdclass={'build_py': CustomBuildPy,
              'bdist_wheel': CachedWheelsCommand},
)
```

### 4.2 `CachedWheelsCommand`：预编译 wheel 下载逻辑（[setup.py:176-200](setup.py#L176-L200)）

```python
class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        if DG_FORCE_BUILD or DG_USE_LOCAL_VERSION:
            return super().run()          # ← 默认走这里：本地构建

        wheel_url, wheel_filename = get_wheel_url()
        try:
            # 从 GitHub Releases 下载对应 wheel
            ...
        except (urllib.error.HTTPError, urllib.error.URLError):
            super().run()                 # 下载失败则回退到源码构建
```

`get_wheel_url()` 拼出的文件名格式（[setup.py:80](setup.py#L80)）：

```
deep_gemm-{version}+cu{cuda_major}-torch{torch_maj.min}-cxx11abi{0|1}-cp{py}{-linux_x86_64}.whl
```

并到 `https://github.com/DeepSeek-AI/DeepGEMM/releases/download/v{version}/{wheel_name}` 下载。

**注意 CUDA 版本取自 `torch.version.cuda`**（即"torch 是用哪个 CUDA 编的"），而不是当前环境装的 CUDA。这是一种"wheel 匹配以 PyTorch 为准"的策略。

**风险提示**：下载路径用的 `urllib.request.urlopen(..., timeout=1)` 超时极短（1 秒），仅捕获 `HTTPError/URLError`，其它异常（如超时抛出的 `socket.timeout`/`TimeoutError`）不会被捕获——即外网不佳时可能直接异常而非回退。默认路径（本地构建）不经过这里，所以正常开发不受影响。

### 4.3 wheel 内部最终长什么样

```
deep_gemm/
  __init__.py, envs.py, _C.pyi, _C.cpython-*.so
  include/deep_gemm/**        # 设备端模板 + host 头
  include/cute/**, include/cutlass/**   # JIT 需要的 CUTLASS 头
  csrc/**                     # 为 agent 排错附带
  docs/**                     # 同上
  mega/, testing/, utils/, legacy/      # 纯 Python 子模块
```

---

## 5. 三个 shell 脚本的区别，以及 `CMakeLists.txt` 的真实用途

### 5.1 `develop.sh`：开发模式（不装进 site-packages）

[develop.sh](develop.sh) 做的事：

1. `ln -sf $script_dir/third-party/cutlass/include/cutlass deep_gemm/include`（`cute` 同理）——把 CUTLASS 头**软链**到包目录，避免复制。
2. `rm -rf build dist`、`rm -rf *.egg-info`。
3. `python setup.py build`。
4. 在 `build/` 里找到 `_C*.so`，**软链回 `deep_gemm/`**。

效果：**原地就能 `import deep_gemm`**，改 Python/host 代码后重新 build 即可，不需要 pip 安装。这是日常开发最常用的方式。

### 5.2 `build.sh`：只出 wheel

[build.sh](build.sh)：`rm -rf build dist` → `python setup.py bdist_wheel`。产物在 `dist/*.whl`，用于分发。

### 5.3 `install.sh`：构建并安装

[install.sh](install.sh)：`rm -rf build dist` → `python setup.py bdist_wheel` → `pip install dist/*.whl --force-reinstall`。一步到位装进当前 Python 环境。

### 5.4 `CMakeLists.txt` 只是给 IDE 用的

[CMakeLists.txt](CMakeLists.txt) 第一行注释已经说明：

> *"current just for CMake-based IDE indexing, the real compilation is done via JIT"*

里面虽然声明了 `pybind11_add_module(_C csrc/python_api.cpp)` 和 `cuda_add_library(deep_gemm_indexing_cuda STATIC csrc/indexing/main.cu)`，但**真正的构建流程完全不经过 CMake**。它的唯一目的是让 CLion/VS Code 等基于 CMake 的 IDE 能正确解析头文件路径、跳转、补全。`csrc/indexing/main.cu` 本身也不参与扩展编译（`sources` 里没有它）。

> 一句话：**CMake 是"假的"，setuptools 才是真的。**

---

## 6. 运行期：DeepJIT 编译机制（重点）

DeepJIT 是仓库 `third-party/deep_jit` 里的一个**纯头文件、C++20 的 JIT 运行时**，同时支持 CUDA 和 Ascend 后端。DeepGEMM 只用它的 CUDA 后端。

### 6.1 初始化：`init_jit`

入口在 [csrc/runtime/jit.hpp](csrc/runtime/jit.hpp)。`deep_gemm/__init__.py` 的最后一行调用 `_C.init(<包目录>)`（[deep_gemm/__init__.py:111](deep_gemm/__init__.py#L111)），进入：

```cpp
inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir  = library_root / "include";
    const auto config = deep_jit::Config(
        library_root,                                  // python_library_root
        "DG",                                          // env_prefix
        "cutlass-" + std::to_string(CUTLASS_VERSION),  // extra_signature
        {include_dir},                                 // include_dirs
        {"deep_gemm/"}                                 // include_prefixes
    );

    jit = deep_jit::LazyInit<...>([config] {
        auto runtime = std::make_shared<...>(config);
        runtime->default_compiler_options.nvcc_flags->emplace_back(
            "--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->emplace_back(
            "--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        return runtime;
    });
}
```

四个 `Config` 参数（定义见 [config.hpp](third-party/deep_jit/include/deep_jit/runtime/config.hpp)）逐一解释：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `python_library_root` | `deep_gemm/` 包目录（绝对路径） | JIT 的"包根"，用于定位 `include/`、解析 post_hook 相对路径 |
| `env_prefix` | `"DG"` | 环境变量前缀，决定读 `DG_JIT_*` 还是 `DJ_JIT_*` |
| `extra_signature` | `"cutlass-<CUTLASS_VERSION>"` | 参与缓存 key 的额外签名。**CUTLASS 版本一变，所有缓存自动失效** |
| `include_dirs` | `{<pkg>/include}` | JIT 编译时的 `--include-path` |
| `include_prefixes` | `{"deep_gemm/"}` | **只有文件名以 `deep_gemm/` 开头的 `#include <...>` 才会被追踪并纳入源码哈希**（见 6.4） |

`LazyInit` 表示 runtime **第一次被解引用时才真正构造**（延迟初始化）。`Runtime` 构造函数里会去探测设备、nvcc 版本、缓存目录等。

另外 [csrc/runtime/runtime.hpp](csrc/runtime/runtime.hpp) 维护**进程级 host 状态**：`num_sms`、`tc_util`、cuBLASLt handle 与 workspace。它是 `LazyInit<Runtime>`，与 JIT 的 runtime 是两回事（同名易混）。

### 6.2 `Runtime` 的成员构成

文件：[runtime.hpp](third-party/deep_jit/include/deep_jit/runtime/runtime.hpp)

```cpp
template <typename Backend> class Runtime {
    Config config;
    Env env;                          // 环境变量解析（DG_* → DJ_*）
    Device device;                    // 设备查询
    CompilerOptions default_compiler_options;
    LaunchOptions default_launch_options;
    DiskCache disk_cache;             // 磁盘缓存
    Backend backend;                  // CUDA 后端（nvcc 调用、cubin 加载）
    Parser parser;                    // #include 解析 + 哈希
    MemCache<std::string, Kernel> mem_cache;  // 进程内缓存
    hash::FNV1a hash_base;            // 缓存 key 的"基底"哈希
};
```

构造函数中：

```cpp
hash_base.update(config.extra_signature);              // 1. cutlass-<ver>
hash_base.update(backend.compiler_info.get_hash());    // 2. nvcc --version 的哈希
```

### 6.3 编译一个 kernel 的完整流程

对外 API 有两个：

```cpp
// 编译 + 加载，返回可 launch 的 Kernel（带进程内缓存）
std::shared_ptr<Kernel> compile(name, source, override_options = {});

// 只编译落盘、不加载（用于离线预热），返回 cubin 路径
std::filesystem::path compile_without_load(name, source, override_options = {});
```

`compile` 的实际步骤（[runtime.hpp:56-98](third-party/deep_jit/include/deep_jit/runtime/runtime.hpp#L56-L98)）：

```
1. options = default_compiler_options.override_with(override_options)   // 合并覆盖
2. key = cache_key(source, options)        // 算内容哈希
3. mem_cache.get_or_create(key, ...)       // 进程内命中则直接返回
4. compile(name, source, key, options):
     a. disk_cache.entry(name, key)
          - 遍历所有缓存根，若存在 <root>/cache/<name>.<key>/.committed → 命中，返回其路径
          - 未命中 → 在 <root[0]>/tmp/<uuid>/ 建临时目录
     b. 命中 → 直接返回路径
     c. 未命中 → backend.compile(...)：写 kernel.cu、调 nvcc 生成 kernel.cubin、
                 按需 dump ptx/sass、写 meta.json
     d. 释放 GIL，entry.commit()：写 .committed → fsync 目录 → 原子 rename 到最终路径
5. Backend::load(path, env)   // 用 CUDA driver API 把 cubin 加载成 Kernel
```

**关键设计点**：

- **内存缓存（`mem_cache`）**：同一进程内相同 key 只编译/加载一次，后续零开销。
- **磁盘缓存**：跨进程、跨运行复用。
- **原子提交**：先在 `tmp/<uuid>` 里构建完整目录，最后 `rename` 到 `cache/<tag>.<digest>`。这是**目录级原子性**——并发进程要么看不到、要么看到完整条目。
- **分布式文件系统安全**：若 `rename` 因别的 rank 已创建而失败，认为别人赢了，清掉自己的临时目录即可。注释特别说明**避免用 `std::filesystem::remove_all`**（在 NFS 等分布式文件系统上并发操作同一父目录可能 segfault 并产生陈旧目录项），改用 `safe_remove_all`。
- **异常路径清理**：`DiskCacheEntry` 析构时会删除未提交的临时目录（best-effort）。
- **tag 命名约束**：`tag` 只能含字母/数字/下划线（`entry()` 里断言），因为它会拼进目录名。

### 6.4 缓存 key 的五个来源（决定"什么变了要重编"）

`cache_key` 的顺序（[runtime.hpp:92](third-party/deep_jit/include/deep_jit/runtime/runtime.hpp#L92)）：

```cpp
std::string cache_key(source, options) {
    auto hash = hash_base;                                   // ① + ②
    options.update_hash(hash);                               // ③ 编译选项（flags 串接后哈希）
    return hash.update(options.get_post_hook_hash(config))   // ④ post_hook 路径 + 内容
               .update(parser.parse_into_hash(source))       // ⑤ 源码 + 被追踪的 include
               .get_hex_digest();                            // 32 位十六进制
}
```

| # | 来源 | 说明 |
| --- | --- | --- |
| ① | `extra_signature` | 这里是 `cutlass-<CUTLASS_VERSION>`，CUTLASS 版本变则全部失效 |
| ② | nvcc 版本哈希 | `nvcc --version` 全文的哈希。换编译器/升级 CUDA 则全部失效 |
| ③ | 有效编译选项 | `CompilerOptions::get_flags()` 拼成的字符串的哈希，含 arch、`-O`、ptxas 选项、nvcc flags 等 |
| ④ | post_hook | 路径字符串 + 该 Python 脚本文件内容的哈希（无则空串）。hook 改了也会失效 |
| ⑤ | 源码 | 源码正文 + 递归追踪到的所有 include 文件的哈希 |

哈希算法是 **FNV-1a**，每段更新前会带上长度前缀（防止歧义拼接），输出 32 字符 hex。

**关键推论**：**修改你自己写的 kernel 源码会重编；但如果你只改了没被 `include_prefixes` 追踪的头文件（比如某些 CUTLASS 头），缓存不会失效**——因为 ⑤ 只追踪 `#include <deep_gemm/...>`。

### 6.5 include 解析规则（容易踩坑）

文件：[parser.hpp](third-party/deep_jit/include/deep_jit/utils/parser.hpp)

`parse_include()` 的规则很严格：

1. 必须是以 `#` 开头、紧跟 `include` 的行；
2. 必须是 **尖括号形式** `#include <...>`；**双引号形式 `#include "..."` 会直接 `DJ_PANIC` 报错**；
3. 文件名**必须以 `include_prefixes` 里的某项开头**（DeepGEMM 只配置了 `deep_gemm/`），否则**不追踪**（返回空，不参与哈希、不加依赖）；
4. 递归追踪时：
   - 用 `include_dirs` 依次查找文件；
   - 有**缓存**（同一文件只哈希一次）；
   - **禁止循环 include**（用 `visiting` 集合检测，发现环则报错）。

所以：JIT 源码里写 `#include <deep_gemm/impls/...cuh>` 会被追踪；写 `#include <cute/...>` 不会（这也解释了为什么 CUTLASS 版本要单独进 `extra_signature`——否则升级 CUTLASS 不会让缓存失效）。

### 6.6 nvcc 是怎么被调用的

文件：[backend/cuda/backend.hpp](third-party/deep_jit/include/deep_jit/backend/cuda/backend.hpp)

**找 toolchain**（`find_cuda_toolkit`）按优先级：

1. 环境变量 `CUDA_HOME`，其次 `CUDA_PATH`；
2. `which nvcc`；
3. `/usr/local/cuda`。

然后 nvcc 路径：若设了 `DG_JIT_NVCC_COMPILER` / `DJ_JIT_NVCC_COMPILER` 就用它，否则 `<cuda_home>/bin/nvcc`。同时尝试找 `cuobjdump`（dump SASS 用）。

**版本检查**：解析 `nvcc --version` 里的 `release X.Y`，**要求 ≥ 12.9**，否则断言失败。

**编译命令**（[backend.hpp:96-115](third-party/deep_jit/include/deep_jit/backend/cuda/backend.hpp#L96-L115)）：

```
cd <workdir> && <nvcc> <workdir>/kernel.cu --cubin --output-file <workdir>/kernel.cubin \
    <options.get_flags()...> \
    --include-path <include_dir> ...
```

- `cd` 到工作目录再执行，注释说明是**为了防止同名 include 文件被误用**。
- 生成物固定为 `kernel.cubin`（+ 可选 `kernel.ptx`、`kernel.sass`）。
- 编译后校验 cubin 存在且非空；若开启 `check_no_spills` / `check_no_local_memory`，会用正则扫 ptxas 输出并断言。
- 若配置了 `post_hook`，编译后额外执行 `python <hook> <cubin>`（可对 cubin 做后处理，如重写）。
- 最后写 `meta.json`：`command`、`config`、`compiler_info`、`compiler_options`。
- 完成后释放 GIL，允许其它 Python 线程并行。

**默认编译选项**（[options.hpp:39-62](third-party/deep_jit/include/deep_jit/backend/cuda/options.hpp#L39-L62)）：

```cpp
optimize_level = "3";           fast_math = false;
ptxas_register_usage_level=10;  ptxas_verbose = false;
check_no_spills=false; check_no_local_memory=false; with_line_info=false;
dump_ptx=false; dump_sass=false;
arch = device.get_arch();
nvcc_flags = {"-std=c++<DG_JIT_CPP_STANDARD|20>",
              "--compiler-options=-fPIC",
              "--compiler-options=-fconcepts",
              "--expt-relaxed-constexpr",
              "--expt-extended-lambda"};
```

`get_flags()` 展开后核心是：`--gpu-architecture=sm_<arch>`、`-O3`、`--compiler-options=-O3`、`--ptxas-options=--register-usage-level=10`，再加上述 flags。

`override_with()` 支持按字段覆盖；`extra_nvcc_flags` 是**追加**语义，其余字段是**覆盖**语义。大 kernel 常通过 `override_options` 调高 `ptxas_register_usage_level` 或加 `extra_nvcc_flags`。

### 6.7 缓存目录布局与多根查找

文件：[cache/disk.hpp](third-party/deep_jit/include/deep_jit/cache/disk.hpp)

```
<cache_root_0>/                 # 唯一可写根
  cache/<tag>.<digest>/
      kernel.cu                 # 源码
      kernel.cubin              # 编译产物
      meta.json                 # 命令 + config + 编译器信息 + 选项
      .committed                # 提交标记（命中判定依据）
      kernel.ptx / kernel.sass  # 可选
  tmp/<uuid>/                   # 构建中的临时目录

<cache_root_1..n>/              # 只读查找根（可选）
  cache/<tag>.<digest>/...
```

- `DG_JIT_CACHE_DIR` 支持**冒号分隔多路径**：解析成"第一个可写根 + 后续只读查找根"。
- 查找命中条件是 `<dir>/cache/<name>.<digest>/.committed` **存在**；命中时会 `try_update_mtime`（LRU 友好）。
- 未设置时默认 `$HOME/.dj`。
- 空路径项会断言报错（避免 `a::b` 这种歧义）。

### 6.8 加载与启动

- 编译产物是 **cubin**，通过 CUDA **driver API** 加载成 `Kernel`（不是 runtime API 的 `cudaModuleLoad`），所以 host 扩展只链接 `cudart` 就够。
- `launch(kernel, override_options, args...)` 用 `LaunchOptions`：`stream`、`num_smem_bytes`、`grid_dim`、`block_dim`、`cluster_dim`、`cooperative`、`enable_pdl`、`nonportable_cluster_size_allowed`。未设置的字段继承默认值（stream 用当前流、cluster 为 `(1,1,1)` 等）。
- 每次 launch 前会 `DJ_HOST_ASSERT(kernel != nullptr)`。

---

## 7. 环境变量体系

DeepJIT 的 `Env` 类（[utils/env.hpp](third-party/deep_jit/include/deep_jit/utils/env.hpp)）实现**两级回退**：

```
<env_prefix>_<SUFFIX>   →   DJ_<SUFFIX>   →   代码内默认值
```

DeepGEMM 的 `env_prefix = "DG"`，所以是 `DG_<SUFFIX>` → `DJ_<SUFFIX>` → 默认。

### 7.1 与缓存/编译相关的变量

| 变量（`DG_` 前缀，可回退 `DJ_`） | 默认 | 作用 |
| --- | --- | --- |
| `DG_JIT_CACHE_DIR` | `$HOME/.dj` | 缓存根，冒号分隔多路径（首个可写） |
| `DG_JIT_CPP_STANDARD` | `20` | JIT 的 `-std=c++<N>` |
| `DG_JIT_PRINT_COMPILER_COMMAND` | `false` | 打印 nvcc 命令 |
| `DG_JIT_DEBUG` | `false` | 打开后等价于同时开启下面多个 dump/verbose 开关 |
| `DG_JIT_DUMP_ASM` | `false` | 同时 dump PTX 与 SASS |
| `DG_JIT_DUMP_PTX` | `false` | dump PTX |
| `DG_JIT_DUMP_SASS` | `false` | dump SASS |
| `DG_JIT_PTXAS_VERBOSE` | `false` | ptxas 详细输出（含寄存器/溢出信息） |
| `DG_JIT_CHECK_NO_SPILLS` | `false` | 出现寄存器溢出即断言失败 |
| `DG_JIT_CHECK_NO_LOCAL_MEMORY` | `false` | 使用 local memory 即断言失败 |
| `DG_JIT_WITH_LINEINFO` | `false` | nvcc 生成行号信息（便于 profiling 定位） |
| `DG_JIT_NVCC_COMPILER` | 空 | 指定 nvcc 绝对路径 |
| `DJ_*`（无 DG_ 对应） | — | 全局 DeepJIT 变量；注意 `DJ` 是保留前缀，不能作为 `env_prefix` |

> 注意：`DG_USE_PYTORCH_CUBLASLT_HANDLE`、`DG_USE_TEMP_CUBLASLT_WORKSPACE` 属于 [csrc/runtime/runtime.hpp](csrc/runtime/runtime.hpp) 里的 cuBLASLt 行为开关，与 JIT 缓存无关。

### 7.2 环境变量回溯到 setup

安装时可以设置 `DG_JIT_CACHE_DIR` / `DG_JIT_PRINT_COMPILER_COMMAND` / `DG_JIT_CPP_STANDARD`，它们会被 `generate_default_envs()` 写进 `envs.py`，从而"固化"到安装产物中（见 3.2）。运行时同名变量优先级更高。

---

## 8. 端到端：从源码到跑通一个算子

```bash
# 0) 准备：确保 submodule 就位（否则 deep_jit/cutlass 头缺失，编译失败）
git submodule status

# 1) 开发模式：原地构建
./develop.sh
#    - symlink cutlass/cute 头到 deep_gemm/include
#    - python setup.py build（编译 _C）
#    - symlink build 里的 _C*.so 回 deep_gemm/

# 2) 验证 import 与初始化（会触发 init_jit → 构造 Runtime → 探测设备/nvcc/缓存目录）
python -c "import deep_gemm; print(deep_gemm.__version__)"

# 3) 首次调用某算子（触发真正的 JIT 编译）
python tests/test_fp8_gemm.py      # 首次会很慢（nvcc 编译），之后走缓存很快

# 4) 观察缓存
ls -R "${DG_JIT_CACHE_DIR:-$HOME/.dj}/cache" | head
```

**首次运行慢是正常的**：每个新 config 都要现场 nvcc 编译，可能几秒到几十秒。第二次起命中磁盘缓存，几乎瞬时。

打包分发则用：

```bash
./build.sh     # 生成 dist/*.whl
# 或
./install.sh   # 生成 wheel 并 pip install --force-reinstall
```

---

## 9. 常见问题与排错

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `from . import _C` 导入失败 | 没编译扩展（如 `DG_SKIP_CUDA_BUILD=1`）或没跑 `develop.sh` | 跑 `./develop.sh` 或安装 wheel |
| `NVCC version must be at least 12.9` | 运行环境 nvcc < 12.9 | 升级 CUDA Toolkit，或设 `DG_JIT_NVCC_COMPILER` 指向新 nvcc |
| 找不到 `cute/...`、`cutlass/...` | submodule 未初始化，或 wheel 缺 include | `git submodule update --init`；确认 `deep_gemm/include/{cute,cutlass}` 存在 |
| 改了自己的 kernel 却没生效 | 改了未被追踪的头（不含 `deep_gemm/` 前缀），或命中了旧缓存 | 把路径改成 `deep_gemm/...` 前缀，或清 `DG_JIT_CACHE_DIR` |
| `non-standard include` / `circular include` 报错 | JIT 源码用了 `#include "..."` 或存在循环 include | 改为 `#include <deep_gemm/...>`，消除循环 |
| `Git working directory is not clean` + 版本变 `+local` | 工作区有未提交改动 | 提交/暂存后重建，或用 `DG_USE_LOCAL_VERSION=0` |
| 首次调用极慢 | 正常运行期 JIT 行为 | 预热/复用 `DG_JIT_CACHE_DIR`；可离线 `compile_without_load` 预热 |
| 缓存目录膨胀 | 每次 config/选项变化产生新条目 | 定期清理；用「冒号分隔多根」把共享缓存挂只读 |
| 外网差时 wheel 下载卡住/报错 | `CachedWheelsCommand` 只在 `DG_USE_LOCAL_VERSION=0` 时走下载 | 保持默认（本地构建），或设 `DG_FORCE_BUILD=1` |

**调试 JIT 的推荐组合**：

```bash
DG_JIT_DEBUG=1 DG_JIT_CACHE_DIR=/tmp/dg_cache python tests/test_fp8_gemm.py
# 会打印 nvcc 命令、dump PTX/SASS、ptxas verbose
```

---

## 10. 为什么这样设计？（设计动机）

1. **编译时间从"安装期"挪到"运行期 + 缓存"**
   如果把所有 shape/config 组合都预先编译，组合数量爆炸且安装极慢。JIT 只编译实际用到的组合。

2. **模板化 + 运行期常量特化带来极致性能**
   把 `N`/`K`、block 配置、stage 数等作为**编译期常量**特化进 kernel（`compiled_dims` 默认 `"nk"`），换来更好的寄存器分配与指令调度。这正是"按内容哈希做缓存"的理由——源码串本身就编码了这些常量。

3. **缓存 key 覆盖所有影响产物的因素**
   CUTLASS 版本、nvcc 版本、编译选项、源码与追踪的 include、post_hook，任一变化都会生成新 key，**保证正确性不回退**；而只改无关文件不会误失效。

4. **面向分布式训练场景**
   缓存条目用目录 + `.committed` + 原子 rename，能在 NFS/并行文件系统上被多 rank 安全共享；多根查找支持"共享只读缓存 + 本地可写缓存"。

5. **薄扩展 + 头文件分发**
   安装期只编一个 TU，速度快、失败面小；真正的模板库以头文件随 wheel 分发，由运行期 JIT 使用。代价是 wheel 较大（带整套 CUTLASS 头）。

6. **为 agent 排错优化**
   把 `csrc/` 与 `docs/` 打进 wheel，报错时可直接定位到 C++ 源码与文档，服务智能排错场景。

---

## 11. 关键文件与行号索引

| 主题 | 位置 |
| --- | --- |
| 开关与版本号 | [setup.py:22-24](setup.py#L22-L24)、[setup.py:51](setup.py#L51) |
| 编译源与 include | [setup.py:32-45](setup.py#L32-L45) |
| 扩展定义 | [setup.py:99-108](setup.py#L99-L108) |
| `CustomBuildPy` 四步 | [setup.py:111-173](setup.py#L111-L173) |
| wheel 下载/回退 | [setup.py:176-200](setup.py#L176-L200) |
| `setuptools.setup` | [setup.py:203-222](setup.py#L203-L222) |
| 开发/构建/安装脚本 | [develop.sh](develop.sh)、[build.sh](build.sh)、[install.sh](install.sh) |
| CMake（仅 IDE） | [CMakeLists.txt](CMakeLists.txt) |
| pyi 生成 | [scripts/generate_pyi.py](scripts/generate_pyi.py) |
| envs 回填 | [deep_gemm/__init__.py:5-12](deep_gemm/__init__.py#L5-L12) |
| `_C.init` 调用 | [deep_gemm/__init__.py:111](deep_gemm/__init__.py#L111) |
| JIT 初始化与 Config | [csrc/runtime/jit.hpp](csrc/runtime/jit.hpp) |
| 进程级 host runtime | [csrc/runtime/runtime.hpp](csrc/runtime/runtime.hpp) |
| Config 定义 | [deep_jit/runtime/config.hpp](third-party/deep_jit/include/deep_jit/runtime/config.hpp) |
| Runtime / cache_key | [deep_jit/runtime/runtime.hpp](third-party/deep_jit/include/deep_jit/runtime/runtime.hpp) |
| 磁盘缓存与原子提交 | [deep_jit/cache/disk.hpp](third-party/deep_jit/include/deep_jit/cache/disk.hpp) |
| include 解析 | [deep_jit/utils/parser.hpp](third-party/deep_jit/include/deep_jit/utils/parser.hpp) |
| CUDA 编译选项 | [deep_jit/backend/cuda/options.hpp](third-party/deep_jit/include/deep_jit/backend/cuda/options.hpp) |
| nvcc 调用与工具链查找 | [deep_jit/backend/cuda/backend.hpp](third-party/deep_jit/include/deep_jit/backend/cuda/backend.hpp) |
| 环境变量两级回退 | [deep_jit/utils/env.hpp](third-party/deep_jit/include/deep_jit/utils/env.hpp) |
| DeepJIT 官方说明 | [third-party/deep_jit/README.md](third-party/deep_jit/README.md) |
| 架构总览（配套文档） | [docs/pjc/deepgemm-architecture-review.md](docs/pjc/deepgemm-architecture-review.md) |
| 接口参考（配套文档） | [docs/pjc/deepgemm-interfaces.md](docs/pjc/deepgemm-interfaces.md) |
