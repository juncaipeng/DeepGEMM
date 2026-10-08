# DeepGEMM 接口参考（入门指南 + 输入格式与示例）

> 配套文档：[deepgemm-architecture-review.md](deepgemm-architecture-review.md)（架构梳理）、[deepgemm-build-install.md](deepgemm-build-install.md)（编译安装）
>
> 版本：`main` @ `78b6900`，`deep_gemm.__version__ = 2.8.0`
>
> 所有接口均通过 `import deep_gemm` 暴露（`_C` 无子模块，C++ 命名空间不体现在 Python 层）。
>
> **本文分两部分**：第一部分（§1–§6）面向第一次接触 DeepGEMM 的读者，只讲概念和上手；第二部分（§7 起）是逐接口的参考手册，供查阅。

---

# 第一部分：入门基础（第一次用必读）

## 1. DeepGEMM 是什么

DeepGEMM 是一个**只做矩阵乘法（GEMM）及相关算子**的 GPU 加速库，专门服务大模型训练/推理：

- 输入两个矩阵 A、B，算出 `D = C + A @ B`（C 是可选的累加项）。
- 支持 Hopper（SM90，如 H800/H100/H20）和 Blackwell（SM100，如 B200）。
- 核心卖点：**FP8/FP4 低精度矩阵乘**，比 cuBLAS 更快，为 DeepSeek-V3 式的 MoE 训练设计。
- 内核是**运行时 JIT 编译**的：安装时不编译任何 kernel，第一次按 shape 调用时现编现用并落盘缓存（默认 `$HOME/.dj`）。所以**第一次调用慢是正常的**，之后直接复用缓存。

## 2. 三个必备概念

### 2.1 矩阵乘法与 M、N、K

GEMM 的世界只有三个字母。对 `D[M,N] = A[M,K] @ B[K,N]`：

```
            K 列                        N 列
        ┌──────────────┐          ┌──────────────┐
   M 行 │      A       │    ×   K │      B       │  =   D[M,N]
        └──────────────┘          └──────────────┘
                                     M 行
                                ┌──────────────┐
                                │      D       │
                                └──────────────┘
```

- **M**：A 的行数（通常是 token 数）
- **N**：B 的列数 / D 的列数（通常是输出特征维度）
- **K**：A 的列数 = B 的行数（**收缩维**，乘完就"消失"的维度，通常是输入特征维度）

DeepGEMM 的函数名/参数全用这三个字母，看到 `M=4096, N=7168, K=2048` 要立刻能对上形状。

### 2.2 内存布局：row-major 与 K-major/N-major

PyTorch 张量是**行主序**（row-major）：`x[i, j]` 中让 `j` 连续存储的那一维叫"连续维"。

- 一个 `[M, K]` 张量如果 `x.stride(-1) == 1`（最后一维 K 连续），就叫 **K-major**；
- 反之如果 M 维连续（比如 `x.T` 的视图），就叫 **MN-major**（或 N-major / M-major）。

```
K-major（正常连续张量）          MN-major（转置视图）
j →  连续                        i ↓  连续
┌────────────┐                  ┌──┬──┬──┬──┐
│ 按行一行行存 │                  │列│列│列│列│
└────────────┘                  └──┴──┴──┴──┘
```

函数名后缀 `nt/nn/tn/tt` 描述的就是 A、B 各自是不是 K-major（详见 §7.1）。**为什么重要**：Tensor Core（TMA 指令）对内存布局有硬性要求，布局不对会直接触发断言失败。

### 2.3 低精度与缩放因子（SF）

FP32 占 4 字节，BF16 占 2 字节，**FP8 只占 1 字节、FP4 半字节**——低精度换来 2–8 倍的带宽和吞吐。但数值范围太小，直接存会溢出/丢精度。解法是**分块缩放（block-wise scaling）**：

```
A [M, K]，按 K 方向每 128 列一块：
┌────────────┬────────────┬────────────┐
│  块 0      │  块 1      │  块 2      │    A_fp8 [M, K]   （数据，除以块缩放后存 FP8）
│  × s[0]    │  × s[1]    │  × s[2]    │    A_sf  [M, K/128]（每块一个 FP32 缩放值）
└────────────┴────────────┴────────────┘
真实值 ≈ A_fp8[m, k] × A_sf[m, k/128]
```

- 这个"每块一个缩放值"就是 **SF（scaling factor）**。
- **recipe 描述 SF 怎么分块**：`(gran_mn, gran_k)` 表示"MN 方向每 gran_mn 行、K 方向每 gran_k 列共享一个缩放值"。
  - `(1, 128)`：每行每 128 列一块（per-token，最常用）
  - `(1, 1)`：整列共享（per-channel，权重常用，配合 `(1,1,128)` 三元组）
  - `(128, 128)`：整块共享（per-block，旧式权重）
  - K 方向 gran 32 是 FP4 的标准配置（MX 格式）
- 调用接口时传的是 `(data, sf)` **二元 tuple**，例如 `a = (a_fp8_tensor, a_sf_tensor)`。
- **SM90 的 SF 是 FP32**；**SM100 的 SF 是打包 UE8M0**（4 个缩放值打包进 1 个 int32，必须是 2 的幂）。别慌：传 FP32 的 SF，库会自动转换（`transform_sf_into_required_layout`，见 §10）。
- FP4 数据是**两个 FP4 打包进 1 个字节**，用 `torch.int8` 张量承载，所以数据张量的最后一维是 `K/2`。

## 3. 五分钟上手

### 3.1 准备

```python
import torch
import deep_gemm   # 需要 SM90/SM100 的 GPU + CUDA >= 12.9
print(deep_gemm.get_num_sms())   # 能打印出数字说明库加载成功
```

### 3.2 第一个例子：BF16（最简单，没有 SF）

BF16 不需要量化，是最平滑的入口：

```python
import torch, deep_gemm

m, n, k = 1024, 512, 7168
a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)  # A [M,K]，K-major
b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)  # B [N,K]，K-major
d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)  # D [M,N]，输出

deep_gemm.bf16_gemm_nt(a, b, d)     # D = A @ B.T

# 验证
ref = (a.float() @ b.float().T).to(torch.bfloat16)
assert (d - ref).abs().max() < 1e-2
```

注意 `nt` 的含义：A `[M,K]`（K-major）、B `[N,K]`（K-major）、数学上算的是 `A @ B.T`。**d 是 in-place 写入**，必须预先分配。

### 3.3 第二个例子：FP8（量化 + SF 处理）

FP8 流程三步：**量化 A → 量化 B → 调用**。量化用库自带的 helper：

```python
import torch, deep_gemm

m, n, k = 4096, 7168, 2048
a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)

# 第 1 步：量化。per-token = 每行每 128 列一个缩放值，返回 (数据, SF) 二元组
a_fp8, a_sf = deep_gemm.per_token_cast_to_fp8(a, use_ue8m0=True,  gran_k=128)
b_fp8, b_sf = deep_gemm.per_token_cast_to_fp8(b, use_ue8m0=True,  gran_k=128)
#    ↑ a_fp8: [m, k] float8_e4m3fn      a_sf: [m, k/128] fp32（或打包 int32）

# 第 2 步：SF 布局调整（SM100 必需；SM90 传 None 更快——见下）
a_sf = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(a_sf)
b_sf = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(b_sf)

# 第 3 步：调用。a、b 都是 (数据, SF) tuple
d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d)
```

关于第 2 步的细节：
- **SM90**：SF 保持 FP32 即可，`per_token_cast_to_fp8(..., use_ue8m0=False)` 产出的 FP32 SF **传 None 也可以**——GEMM 接口内部会自动调用 `transform_sf_into_required_layout` 做布局转换（额外开销小）。
- **SM100**：需要打包 UE8M0。同样可以让接口内部自动转，但**每次调用都转一遍有开销**；生产环境建议像上面那样在量化后立即手动转好、缓存复用。
- 想要累积（残差）：`deep_gemm.fp8_gemm_nt(a, b, d, c=c)`，通常 `c` 与 `d` 是同一块内存（`c = d`）。

### 3.4 第一次调用慢、显存、验证

- **首次调用会触发 JIT 编译**（几秒到几十秒），同 shape 的第二次调用直接命中缓存。生产环境先 warmup。
- 输出 `d` 只支持 **BF16 或 FP32**（FP8 输出仅 `fp8_einsum` 的一个表达式和 mega 系列支持）。
- 验证精度：FP8 参考实现是 `ref = (a.float() @ b.float().T)`（注意 SF 要乘回去，直接用 `cast_back_from_fp8` 反量化更省事）。

## 4. 我该用哪个函数（决策树）

```
你的输入是什么？
│
├─ BF16 ────────────────────────────── 稠密：bf16_gemm_nt（§8.2）
│                                      分组：m_grouped_bf16_gemm_*（§8.5）
│
├─ FP8/FP4，A、B 是普通矩阵 ────────── 稠密：fp8_gemm_nt（§8.1）
│                                      训练反向 wgrad：k_grouped_fp8_gemm_*（§8.4）
│
├─ FP8/FP4，MoE 多专家 ─────────────── 训练前向（输入已按专家拼好）：m_grouped_*_contiguous（§8.3）
│                                      推理解码（CUDA graph）：m_grouped_*_masked（§8.3）
│                                      SM100 想要一步到位（dispatch+GEMM+combine 全融合）：Mega MoE（§12）
│
├─ 注意力 logits（MQA） ─────────────── fp8_fp4_mqa_logits 系列（§11）
│
└─ Router 选专家 ───────────────────── bf16_mega_gate（§13，仅 SM100）
```

recipe 怎么选（新手直接抄）：

| 场景 | recipe | 等价写法 |
|---|---|---|
| FP8 激活 × FP8 权重（per-token × per-128-channel） | `recipe_a=(1,128), recipe_b=(1,1)` 语义；惯例写 `recipe=None`（fp32 SF 时内部按 (1,128)/(128,128) 分派）或 `recipe_a=(1,128), recipe_b=(128,128)` | — |
| FP8 × FP4 权重（SM100） | `recipe_a=(1,128), recipe_b=(1,32)` | — |
| FP4 × FP4（SM100） | `recipe_a=(1,32), recipe_b=(1,32)` | — |
| 训练 wgrad（per-channel） | `recipe=(1,1,128)` 三元组 | — |

## 5. 常见报错速查（新手 90% 的坑都在这）

| 症状 | 原因 | 修法 |
|---|---|---|
| 第一次调用卡住几十秒 | JIT 在编译 | 正常，等缓存；生产先 warmup |
| `DG_HOST_ASSERT` 提到 major / K-major | 内存布局不对 | 检查 A/B 是否 K-major（`x.stride(-1)==1`）；FP4 在任何架构都强制双 K-major |
| `d` 的断言失败 | 输出 dtype 不对 | `d` 只能 bf16/fp32（分组 GEMM 只能 bf16）；`d` 必须 N-major |
| SF 相关断言 | SF shape/dtype 不符 | 用 `per_token_cast_to_fp8` 产出原生格式；或 `transform_sf_into_required_layout` 归一化 |
| SM100 上 SF 断言"power of 2" | UE8M0 编码要求 | SF 值必须能对齐到 2 的幂，用 `use_ue8m0=True` 的 helper 产出 |
| 分组 GEMM 结果错乱 | 没设对齐 | 先 `set_mk_alignment_for_contiguous_layout(get_theoretical_mk_alignment_for_contiguous_layout())`，再构造数据 |
| `fp8_gemm_nt` 传入 `m_grouped` 的 3D 张量 | 接口用错 | 3D 的 A/B 要用 `m_grouped_*` 系列 |
| 传了关键字参数报 TypeError | 部分 config 函数 pybind 没写 `py::arg` | 改用位置参数，如 `set_num_sms(132)` |
| SM90 上调用 nn/tn/tt 失败 | SM90 的 FP8 内核只支持双 K-major | 布局转成 NT（`x = x.t().contiguous().t()` 之类）或换 `fp8_gemm_nt` |

## 6. 三个黄金法则

1. **先跑 BF16，再上 FP8**：BF16 验证 shape/布局正确，FP8 验证量化正确，问题隔离最省时间。
2. **数据构造抄测试**：`tests/generators.py` 里每个场景都有生成函数（§18），别自己从零拼。
3. **`d` 永远预分配、in-place 写**；累加传 `c`（通常 `c is d`）。

---

# 第二部分：接口参考手册

## 7. 通用约定

### 7.1 命名与数学含义

`D = C + A @ B`。后缀表示 A/B 的"主维"（major）：

| 后缀 | 含义 | A | B | C/D |
|---|---|---|---|---|
| `nt` | A 非转置、B 转置 | `[M,K]` K-major | `[N,K]` K-major | N-major |
| `nn` | 二者都非转置 | `[M,K]` K-major | `[K,N]` N-major | N-major |
| `tn` | A 转置、B 非转置 | `[K,M]` MN-major | `[K,N]` N-major | M-major |
| `tt` | 二者都转置 | `[K,M]` | `[N,K]` | M-major |

- **SM90 的 FP8/FP4 内核只支持双 K-major 输入（即只有 `nt` 组合实际可用）**；SM100 支持 NT/TN/NN/TT 全部。BF16 无此限制：`nn/tn/tt` 是薄 transpose 包装（转成 `nt` 调用），且 cuBLASLt 路径任意布局都支持。
- 文档常说 `fp8_gemm_nt` 做 `D = C + A @ B.T`。
- `cd` 必须 N-major（`check_major_type_cd`）。

### 7.2 数据类型与打包

| 概念 | Python/Torch 表示 |
|---|---|
| FP8 | `torch.float8_e4m3fn` |
| FP4 | packed，**2 个 FP4/字节**，用 `torch.int8` 承载（C++ 别名 `kPackedFP4 = torch::kInt8`），故逻辑 K 维数据 shape 的最后一维是 `K/2` |
| BF16 | `torch.bfloat16` |
| SF（scale factor） | SM90：`torch.float32`；SM100：packed UE8M0，**4 个打包进 1 个 `torch.int32`** |

`(data, sf)` 以 **Python tuple** 形式传入（`a=(a_fp8, a_sf)`）。

### 7.3 两种 SF recipe 写法

- **3 元组** `recipe=(gran_m, gran_n, gran_k)` —— 传给 `recipe=` 参数，隐含 `is_sfa` 由内部判断（一般 SFA 用 `gran_m`，SFB 用 `gran_n`）。
- **2 元组** `recipe_a=(gran_mn, gran_k)` / `recipe_b=(gran_mn, gran_k)` —— 分别给 A/B，**不可再传 `recipe`**。

常见取值：
- 旧式 `(128,128,False,False)` 配置 → forward 传 `recipe=None`，wgrad 传 `(1,1,128)`。
- `(1,128,128)`：per-token 1x128。
- `(1,1,128)`：per-128-channel（K 方向）。
- `(1,1,32)`：FP4 常用，K 方向 gran 32。
- `(1,32)`（2 元组）：FP4 权重。

### 7.4 环境/全局配置

```python
import deep_gemm

deep_gemm.set_num_sms(132)          # 限制使用的 SM 数
deep_gemm.get_num_sms()
deep_gemm.set_tc_util(100)          # 近似 tensor core 利用率(%)
deep_gemm.get_tc_util()
deep_gemm.set_pdl(True)             # 开/关 Programmatic Dependent Launch
deep_gemm.get_pdl()
deep_gemm.use_deterministic_algorithms(True)   # 确定性算法（会禁 cuBLASLt 快路径、固定 split-K=1）
deep_gemm.set_ignore_compile_dims(True)        # 忽略编译期维度特化
deep_gemm.set_block_size_multiple_of(64)       # 约束 block size 为某值的倍数；也可传 (bm, bn)
```

`set_num_sms / get_num_sms / set_pdl` 等部分函数在 pybind 未写 `py::arg`，因此**只能用位置参数**。

### 7.5 对齐（分组 GEMM 的关键）

```python
align_m = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()          # 理论最小对齐
align_m = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m) # 带 M 期望值
deep_gemm.set_mk_alignment_for_contiguous_layout(align_m)   # 进程级设置
deep_gemm.get_mk_alignment_for_contiguous_layout()
```

**这是进程级全局状态**，测试里都先设置再生成数据。别名（`deep_gemm/utils/layout.py`）：
`get_m_alignment_for_contiguous_layout` / `get_k_alignment_for_contiguous_layout` = 同一个 getter。

---

## 8. 稠密与分组 GEMM

### 8.1 FP8/FP4 稠密：`fp8_fp4_gemm_{nt,nn,tn,tt}`

```python
deep_gemm.fp8_fp4_gemm_nt(
    a,            # tuple[Tensor, Tensor] = (data, sfa)
    b,            # tuple[Tensor, Tensor] = (data, sfb)
    d,            # Tensor, BF16 或 FP32（输出，in-place 写入）
    c=None,       # Tensor 或 None；累积时传（可与 d 同一块内存）
    recipe=None,          # tuple[int,int,int]
    recipe_a=None,        # tuple[int,int]
    recipe_b=None,        # tuple[int,int]
    compiled_dims='nk',   # 编译期特化维度（tn/tt 为 'mn'）
    disable_ue8m0_cast=False,
    alpha=None,           # float 缩放；仅 SM100 JIT 路径支持
)
```

**输入格式**

| 参数 | 形状 | dtype | 备注 |
|---|---|---|---|
| `a[0]` | `[M,K]` | fp8 或 packed fp4(int8) | SM90/双 FP4 时强制 K-major |
| `a[1]` | SF | fp32(SM90) / int32(SM100) | 布局需 TMA-aligned、转置（MN-major） |
| `b[0]` | `[N,K]` | 同上 | |
| `b[1]` | SF | 同上 | |
| `d` | `[M,N]` | bf16/fp8-e4m3/fp32 | 必须 N-major；FP8 D ⟺ 传 `sfd`（此处 `fp8_fp4_gemm_*` 不含 sfd，FP8 输出走 einsum 的 `fp8_bmm`） |
| `c` | `[M,N]` | 同 `d` | 量化 D 不可累积 |

**别名**（对象完全相同）：
`fp8_gemm_nt/nn/tn/tt`、`fp4_gemm_nt` 都指向 `fp8_fp4_gemm_*`。

**示例**（来自 `tests/test_fp8_fp4.py::test_gemm`）

```python
a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
c = d if accumulate else None

# 量化（helper 见 §15.2）
a = (per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=128),
     ...)   # -> (data, sf)
b = per_token_cast_to_fp8(b, use_ue8m0=True, gran_k=128)

deep_gemm.fp8_fp4_gemm_nt(a, b, d,
                          c=c,
                          disable_ue8m0_cast=not use_ue8m0,
                          recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                          alpha=alpha)
```

**三种量化组合（SM100）**

```python
# FP8 x FP8，gran_k=128
QuantConfig((128, 128, False, False))
# FP8 A(gran 128) x FP4 B(gran 32)
QuantConfig((128, 32, False, True))   # recipe_a=(1,128), recipe_b=(1,32)
# FP4 x FP4，gran_k=32
QuantConfig((32, 32, True, True))
```

误差容忍：FP8xFP8 `1e-3`，混合 `1e-2`，FP4xFP4 `2e-2`。

### 8.2 BF16 稠密：`bf16_gemm_{nt,nn,tn,tt}`

```python
deep_gemm.bf16_gemm_nt(a, b, d, c=None, compiled_dims='nk', alpha=None)
# nn 同 nt 默认 'nk'；tn/tt 默认 'mn'，如：
# deep_gemm.bf16_gemm_tn(a, b, d, c=None, compiled_dims='mn', alpha=None)
```
- `a,b`: `[M,K]` / `[N,K]` BF16；`d`: `[M,N]` BF16 或 FP32。
- **分派顺序**：若 cuBLASLt 可用且未开确定性模式 → cuBLASLt 快路径（任意架构、任意布局都支持 `alpha`）；否则走自研 JIT 内核（SM90 禁 `alpha`，SM100 支持）。
- 示例：`deep_gemm.bf16_gemm_nt(a, b, d, c=c, alpha=alpha)`（`tests/test_bf16.py:43`）。

### 8.3 M 分组：`m_grouped_*_contiguous` / `m_grouped_*_masked`

**M-grouped contiguous（前缀 / 训练前向）**

```python
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a, b, d, grouped_layout,
    recipe=None, recipe_a=None, recipe_b=None,
    compiled_dims='nk', disable_ue8m0_cast=False,
    use_psum_layout=False, ensure_zero_padding=True,
    expected_m_for_psum_layout=None,   # nn 版无此参数
)
# nn 版：m_grouped_fp8_fp4_gemm_nn_contiguous(a, b, d, grouped_layout,
#     recipe=..., recipe_a=..., recipe_b=..., compiled_dims='nk',
#     disable_ue8m0_cast=False, use_psum_layout=False, ensure_zero_padding=True)
```
- `a`: `[M,K]`，`b`: `[G,N,K]`，`d`: `[M,N]`（**仅 BF16**）。A 必须 K-major。
- **DeepGEMM 只对 M 分组**，N/K 固定（面向 MoE 中同形状的专家）。
- `grouped_layout`（非 PSUM）：长度 `M` 的 `int32`，`grouped_layout[m]` = 该行所属 group 号；**padding 行填 `-1`**；各 group 起点需按 M block 对齐。
- `grouped_layout`（PSUM，`use_psum_layout=True`）：长度 `num_groups`，存**累积行边界**（非对齐），此时还需要 `expected_m_for_psum_layout`。
- 别名：`m_grouped_fp8_gemm_nt_contiguous` / `m_grouped_fp4_gemm_nt_contiguous`（nn 同理）。

**示例**（`tests/generators.py::generate_m_grouped_contiguous`）

```python
alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

m = sum(align(actual_m, alignment) for actual_m in actual_ms)
a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

grouped_layout = torch.empty(m, device='cuda', dtype=torch.int32)
for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
    grouped_layout[start:actual_end] = i
    grouped_layout[actual_end:aligned_end] = -1   # padding
    a[actual_end:aligned_end] = 0                 # padding 置零（保证量化 SF 规则）
    start = aligned_end

deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a, b, d, grouped_layout, use_psum_layout=..., ensure_zero_padding=...)
```

**M-grouped masked（解码 + CUDA graph）**

```python
deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
    a, b, d, masked_m, expected_m,
    recipe=None, recipe_a=None, recipe_b=None,
    compiled_dims='nk', disable_ue8m0_cast=False)
```
- `a`: `[G,M,K]`，`b`: `[G,N,K]`，`d`: `[G,M,N]`（**仅 BF16**）。A/B 都必须 K-major。
- `masked_m`：长度 `num_groups` 的 `int32`，**每 group 的有效行数**（不是 offset）。`expected_m` 是必填标量（启发式提示）。
- 场景：CUDA graph 下 CPU 不知道每专家收到多少 token，用 mask 只算有效部分；典型输入来自 DeepEP 低延迟 kernel。
- 别名：`m_grouped_fp8_gemm_nt_masked` / `m_grouped_fp4_gemm_nt_masked`。

**示例**

```python
a = torch.randn((num_groups, max_m, k), ...)
b = torch.randn((num_groups, n, k), ...)
d = torch.empty((num_groups, max_m, n), ...)
masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a, b, d, masked_m, expected_m)
```

### 8.4 K-grouped contiguous（MoE 权重反向）

```python
deep_gemm.k_grouped_fp8_gemm_tn_contiguous(          # SM100
    a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1,1,128),
    compiled_dims='mn', use_psum_layout=False)
deep_gemm.k_grouped_fp8_gemm_nt_contiguous(          # SM90，c 必填，且不支持 PSUM
    a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1,1,128),
    compiled_dims='mn', use_psum_layout=False)
deep_gemm.k_grouped_fp4_gemm_nt_contiguous(          # SM100，recipe=(1,1,32)
    a, b, d, ks_cpu, grouped_layout, c=None, recipe=(1,1,32),
    compiled_dims='mn', use_psum_layout=False)
```
- **K 轴分组，M/N 固定**。`a`: `[total_k, M]`，`b`: `[total_k, N]`（都是 K 维在首维、MN-major）。
- `ks_cpu`: `Optional[list[int]]`，每组 **K 大小（须已按 k_alignment 对齐）**；`ks_cpu` 只能为 None/空当 `use_psum_layout=True`（此时 sum_k 取 `a.size(0)`）。
- `grouped_layout`（非 PSUM）是**对齐后的每组 K 大小**；PSUM 时是每组的**逻辑 K 终点**（起点为 `align(前一组终点, k_alignment)`）。
- `k_alignment % 128 == 0`（FP8/BF16），FP4 要求 `% 256 == 0`；且每个 `ks[i] % k_alignment == 0`。
- `recipe` 的前两维强制为 1（仅 `(1,1,128)` / `(1,1,32)`）。
- BF16 版：`k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks_cpu, grouped_layout, c=None, compiled_dims='mn', use_psum_layout=False)`；SM90 上 `c` 必填且不支持 PSUM。

**示例**

```python
k_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
host_ks_cpu = [align(k, k_alignment) for k in logical_ks_cpu]
a = torch.zeros((total_k, m), device='cuda', dtype=torch.bfloat16)
b = torch.zeros((total_k, n), device='cuda', dtype=torch.bfloat16)
gemm(a, b, equivalent_d, host_ks_cpu, grouped_layout,
     equivalent_d if accumulate else None, recipe=(1,1,gran_k))
```

### 8.5 BF16 分组（补充，此前文档遗漏）

```python
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
    a, b, d, grouped_layout, compiled_dims='nk',
    use_psum_layout=False, ensure_zero_padding=True, expected_m_for_psum_layout=None)
deep_gemm.m_grouped_bf16_gemm_nn_contiguous(       # 无 expected_m 参数
    a, b, d, grouped_layout, compiled_dims='nk',
    use_psum_layout=False, ensure_zero_padding=True)
deep_gemm.m_grouped_bf16_gemm_nt_masked(
    a, b, d, masked_m, expected_m, compiled_dims='nk')
```
- 语义与 FP8/FP4 的对应版本一致（`a` `[M,K]`/`[G,M,K]`、`b` `[G,N,K]`、`d` `[M,N]`/`[G,M,N]` 均 BF16；masked 版 A/B 双 K-major，contiguous 版 A 必须 K-major）。
- 用法示例见 `tests/test_bf16.py:62-148`（含 PSUM 与 `expected_m_for_psum_layout` 的搭配）。

### 8.6 跳过 head 中段：`fp8_gemm_nt_skip_head_mid`（attention 用）

```python
deep_gemm.fp8_gemm_nt_skip_head_mid(
    a, b, d, head_splits, recipe=None, compiled_dims='nk', disable_ue8m0_cast=False)
```
- `head_splits=(left, mid, right)`，`n % (left+right) == 0`，`d` 形状 `[M, n + n/(left+right)*mid]`。
- 例：`head_splits=(128,64,128)`；`d` 由 `[m,n]` 经 `apply_skip_head_mid` 零填充中段扩展得到。
- SM90 走 1d2d，SM100 走 1d1d（仅 FP8，gran 128）。`recipe=None` 时内部会填充默认 recipe，无需显式传。

### 8.7 cuBLASLt 包装（对照/兜底）

```python
deep_gemm.cublaslt_gemm_nt(a, b, d, c=None)   # 也有 nn/tn/tt
deep_gemm.cublaslt_nvfp4_gemm_nt(a, b, d, c=None)  # a,b 为 packed fp4 tuple
deep_gemm.batched_syrk(a, d)     # D = A @ A.mT；a 可 2D 或 3D
deep_gemm.batched_symm(a, b, d)  # D = A @ B，A 对称；2D 或 3D
```
`cublaslt_nvfp4_gemm_nt` 要求：a/b 为 packed FP4（int8，K/2 末维）且 K-major 连续，`k % 32 == 0`，SF 为 **UE4M3 字节**、cuBLASLt tiled layout `[ceil(mn/128), ceil(k/64), 32, 4, 4]`（16 元素一块），dtype uint8/int8。

---

## 9. Einsum（硬编码表达式）

```python
deep_gemm.einsum(expr: str, a, b, d, c=None)          # BF16
deep_gemm.fp8_einsum(expr: str, a, b, d, c=None, recipe=(1,128,128))  # FP8
```

| 表达式 | 形状 | 输出 | 说明 |
|---|---|---|---|
| `bmk,bnk->mn` | `a=(s,m,k)`, `b=(s,n,k)` | `(m,n)` fp32/bf16 | batch 归约；fp32 输出必须 C 别名 D（累积），bf16 输出不能给 C |
| `bhr,hdr->bhd` | `a=(b,h,r)`, `b=(h,d,r)` | `(b,h,d)` bf16 | 非确定性模式且 cuBLASLt 可用时优先 cuBLASLt，否则回落自研 kernel；`c` 必须 None；所有 `stride(2)==1` |
| `bhd,hdr->bhr` | `a=(b,h,d)`, `b=(h,d,r)` | `(b,h,r)` | 同上 |
| `bhd,bhr->hdr` | `a=(b,h,d)`, `b=(b,h,r)` | `(h,d,r)` fp32 | **仅 cuBLASLt**，可给 C 累积 |

`fp8_einsum`：`a,b` 为 `(data, sf)` 对，**SF 为 FP32**（SM100 上会被自动打包成 UE8M0 int32，也可预先传打包好的 int32；绝不能是 fp8 e4m3）。

- `bhr,hdr->bhd`：**唯一支持 FP8 输出** `(d, sfd)`；SF 需 `d.stride(0)==n` 且 `n % 128 == 0`、`c=None`、仅 SM100；输出为 FP8 tuple 时 gran_k=32。该表达式在 SM90 也可用（强制 gran_k=128）。
- 另两个 fp8 表达式仅 SM100（arch 10）。

**示例**

```python
deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c)

x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=use_ue8m0)
x_fp8 = x_fp8[0].view(b, h, r), x_fp8[1].view(b, h, ceil_div(r, 128))
y_fp8 = per_block_cast_to_fp8(...)
output = torch.empty_like(z, dtype=output_dtype)
sfd = torch.empty_strided((b, ceil_div(h*d, 32*4)), (1, align(b, 4)), dtype=torch.int32)
deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, (output, sfd))
```

---

## 10. SF / Layout 工具

```python
deep_gemm.get_tma_aligned_size(x: int, element_size: int) -> int
deep_gemm.get_mn_major_tma_aligned_tensor(sf: Tensor) -> Tensor
deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf: Tensor, psum_layout=None) -> Tensor
deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf, grouped_layout, ks_cpu, gran_k, k_alignment, use_psum_layout=False) -> Tensor
deep_gemm.transform_sf_into_required_layout(
    sf, mn, k, recipe,
    num_groups=None, is_sfa=None, disable_ue8m0_cast=False, psum_layout=None) -> Tensor
```

**`transform_sf_into_required_layout` 产出规则**（`csrc/apis/layout.hpp:37-57`）：

| 输入 | 条件 | 输出 |
|---|---|---|
| FP32 `gran_mn=1, gran_k=128` | SM90 或 `disable_ue8m0_cast` | TMA-aligned、MN-major 的 FP32 |
| FP32 `gran_mn=128, gran_k=128` | SM90 或 `disable_ue8m0_cast` | 不做变换（校验 SM90 SFB 连续性） |
| FP32 `gran_k∈{32,128}` | SM100 | 先广播 `gran_mn>1`，再打包成 **int32 UE8M0**、TMA-aligned、MN-major |
| INT `gran_mn=1, gran_k∈{32,128}` | SM100 | 已是 packed UE8M0，仅校验 |

- 2 元组 recipe**不能**传 `is_sfa`；3 元组**必须**传 `is_sfa`。
- `get_mn_major_tma_aligned_tensor` 输出 strides `{aligned_mn*sf_k, 1, aligned_mn}`（2D 输入→2D 输出；已是目标布局时原样返回）。
- `get_...packed_ue8m0_tensor` 输出 int32 `{num_batches, mn, ceil_div(sf_k,4)}`，strides `{packed_sf_k*aligned_mn, 1, aligned_mn}`。`psum_layout` 需满足：单 batch、SF 连续、psum 为非空连续 int32。
- `get_k_grouped_...` 仅 SM100，输出 **contiguous** int32 `{packed_sf_k, mn}`。约束：SF 2D 连续、`mn % 4 == 0`、`num_groups <= 128`、每组 `k % k_alignment == 0`；`ks_cpu` 可为 None/空但需 `use_psum_layout=True`（此时 packed_sf_k 按上界估计）。
- 注意：`transform_sf_into_required_layout` 从 `deep_gemm` 顶层导出，但**不在 `deep_gemm.utils` 里**。

**示例（权重 SF 预处理，Mega MoE）**

```python
w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
```

**示例（layout kernel）**

```python
packed_sf = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf)
transposed_sf = deep_gemm.get_mn_major_tma_aligned_tensor(fp32_sf)
packed_sf = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    fp32_sf, grouped_layout, ks_cpu, gran_k, k_alignment)
```

---

## 11. Attention / Lightning Indexer（MQA logits）

数学定义：

```
logits[q,j] = sum_h weights[q,h] * ReLU(dot(q[h], kv[j]))
```

### 11.1 非 paged / prefill

```python
deep_gemm.fp8_fp4_mqa_logits(
    q,                        # tuple[Tensor, Optional[Tensor]] = (q_fp, q_sf)，q_sf 可选（FP4 必填）
    kv,                       # tuple[Tensor, Tensor] = (kv_fp, kv_sf)，kv_sf 必填
    weights,
    cu_seq_len_k_start, cu_seq_len_k_end,
    clean_logits=True, max_seqlen_k=0,
    logits_dtype=torch.float32, schedule_meta=None) -> Tensor
```

| 参数 | 形状 | dtype | 备注 |
|---|---|---|---|
| `q_fp` | `[seq_len, num_heads, head_dim]` | fp8 / packed fp4 | contiguous；`head_dim ∈ {32(fp8),64,128}` |
| `q_sf` | `[seq_len, num_heads]` | int32 | 仅 MX（SM100）；FP4 必填 |
| `kv_fp` | `[seq_len_kv, head_dim]` | 同 q | contiguous |
| `kv_sf` | `[seq_len_kv]` | int32(MX) / fp32(非 MX) | 必填 |
| `weights` | `[seq_len, num_heads]` | fp32；SM100 可 bf16 | `stride(1)==1`；bf16 时 `logits_dtype` 也要 bf16 |
| `cu_seq_len_k_start/end` | `[seq_len]` | int32 | 每行的 KV 起止 |
| 输出 | `max_seqlen_k==0` → `[seq_len, seq_len_kv]`；否则 `[seq_len, max_seqlen_k]` | fp32/bf16 | **由库内部分配**（返回视图），行 stride 自动 padding 到 1024B 对齐，无需调用方处理 |

约束：`num_heads > 0 and <= 128 and num_heads % 4 == 0`；`block_kv=256`，`block_q=128/num_heads`（内部常量）。压缩输出（`max_seqlen_k>0`）要求 `clean_logits=False`。

**SM100** 支持 fp8+MXFP4+MXFP8；**SM90** 仅 fp8、`num_heads ∈ {32,64}`、weights fp32、禁 `schedule_meta`。

**metadata（可选，SM100 only）**

```python
deep_gemm.get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end,
                                  num_kv_tokens, num_heads) -> Tensor  # 1D int32
```

**示例**（`tests/test_attention.py::test_mqa_logits`）

```python
q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float32)
ks = torch.zeros(seq_len, dtype=torch.int, device='cuda')
ke = torch.arange(seq_len, dtype=torch.int, device='cuda') + (seq_len_kv - seq_len)

# MX 路径量化
q_q = per_token_cast_to_fp4(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
q_in = (q_q[0].view(seq_len, num_heads, head_dim // 2), q_q[1].view(seq_len, num_heads))
kv_q = per_token_cast_to_fp8(kv.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
kv_in = (kv_q[0].view(seq_len_kv, head_dim), kv_q[1].view(seq_len_kv))

logits = deep_gemm.fp8_fp4_mqa_logits(
    q=q_in, kv=kv_in, weights=weights,
    cu_seq_len_k_start=ks, cu_seq_len_k_end=ke,
    clean_logits=True, max_seqlen_k=0, logits_dtype=torch.float32)

# 调度路径（SM100）
workspace = deep_gemm.get_mqa_logits_metadata(ks, ke, seq_len_kv, num_heads)
scheduled = deep_gemm.fp8_fp4_mqa_logits(..., schedule_meta=workspace)
```

### 11.2 Paged / decode

```python
deep_gemm.fp8_fp4_paged_mqa_logits(
    q, kv_cache, weights, context_lens, block_table, schedule_meta, max_context_len,
    clean_logits=False, logits_dtype=torch.float32, indices=None) -> Tensor
```

| 参数 | 形状 | dtype | 备注 |
|---|---|---|---|
| `q_fp` | `[batch_size, next_n, num_heads, head_dim]` | fp8/packed fp4 | `head_dim ∈ {32,64,128}` |
| `q_sf` | `[batch_size, next_n, num_heads]` | int32 | 仅 MX |
| `fused_kv_cache` | `[num_kv_blocks, block_kv, num_heads_kv=1, head_dim_with_sf]` | uint8 | **KV 数据 + 页尾 SF 融合在一条 uint8 里**；`head_dim_with_sf = kv_dim + 4`（4 字节 SF）；要求 `stride(1)==head_dim_with_sf`、`stride(3)==1`、`stride(0) % 4 == 0` |
| `weights` | `[batch_size*next_n, num_heads]` | fp32/bf16 | |
| `context_lens` | `[batch_size, next_n]` | int32 | **仅支持 2D** |
| `block_table` | `[batch_size, max_block_len]` | int32 | `stride(1)==1` |
| `schedule_meta` | `[num_sms+1, 2]` | int32 | 来自下面的 metadata 函数；num_sms 必须与当前 `get_num_sms()` 一致 |
| `indices` | `[batch_size]` | int32 | varlen，仅 SM100，`next_n==1` |
| 输出 | `[batch_size*next_n, max_context_len]` | fp32/bf16 | 同样由库内部分配 |

`block_kv ∈ {32,64,128}`（SM100），SM90 固定 64。paged 不支持 `clean_logits`。

```python
deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None) -> Tensor
```

**示例**（`tests/test_attention.py::test_paged_mqa_logits`）

```python
context_lens_nextn = ((context_lens.unsqueeze(1) + 1) * torch.rand(batch_size, next_n)).int()
context_lens_nextn[:, -1] = context_lens

metadata = deep_gemm.get_paged_mqa_logits_metadata(
    context_lens=context_lens_nextn, block_kv=block_kv,
    num_sms=deep_gemm.get_num_sms(), indices=indices)

logits = deep_gemm.fp8_fp4_paged_mqa_logits(
    q=q_in, kv_cache=kv_in, weights=kernel_weights,
    context_lens=context_lens_nextn, block_table=block_table,
    schedule_meta=metadata, max_context_len=max_model_len,
    clean_logits=False, logits_dtype=logits_dtype, indices=indices)
```

### 11.3 Sparse MQA logits（仅 SM100）

固定 `num_heads=32`、`head_dim=128`、`block_q=2`。

```python
deep_gemm.fp8_fp4_sparse_mqa_logits(
    q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv, use_unaligned_ks=False)
deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
    q, kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv)
```
- `num_max_sparse_blocks ∈ (0, 4096]` 且 `% 4 == 0`；`sparse_block_kv ∈ {8,16}`。
- `q_sf` 必填；`q_sf`/`kv_sf` 均 int32 contiguous；`weights` **必须 bf16**（`stride(1)==1`，比非 sparse 版更严格）。
- 输出 bf16 `[num_q_tokens, num_max_sparse_blocks * sparse_block_kv]`。
- 语义：每个 `sparse_kv_block_indices` 行以其 KV 长度推出的有效块**开头**，该前缀必须是**唯一、严格递增**的绝对块索引；`use_unaligned_ks` 时块 i 起点为 `i*sparse_block_kv + ks % sparse_block_kv`。
- paged 版额外要求：`next_n==1`、q 4D、`page_kv % sparse_block_kv == 0`、`fused_kv_cache.stride(0) % 512 == 0`。

metadata：

```python
deep_gemm.get_sparse_mqa_logits_metadata(
    cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens,
    sparse_kv_block_indices, qk_dtype, sparse_block_kv, use_unaligned_ks=False) -> Tensor(uint8)
deep_gemm.get_paged_sparse_mqa_logits_metadata(
    context_lens, block_table, indices, page_kv,
    sparse_kv_block_indices, qk_dtype, sparse_block_kv) -> Tensor(uint8)
```
Paged 版要求：同一请求的 query 连续；配对 query 的 block-table 行必须一致；`page_kv % sparse_block_kv == 0`。

**示例（contiguous）**

```python
metadata = deep_gemm.get_sparse_mqa_logits_metadata(
    cu_seq_len_k_start=starts, cu_seq_len_k_end=ends, num_kv_tokens=num_kv_tokens,
    sparse_kv_block_indices=sparse_indices, qk_dtype=q[0].dtype, sparse_block_kv=8)
out = deep_gemm.fp8_fp4_sparse_mqa_logits(
    q=q, kv=(kv_fp, kv_sf), weights=weights, metadata=metadata,
    num_max_sparse_blocks=2048, sparse_block_kv=8)
```

### 11.4 旧接口（legacy）

```python
deep_gemm.fp8_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                         clean_logits=True, max_seqlen_k=0)
deep_gemm.fp8_paged_mqa_logits(q, kv_cache, weights, context_lens, block_table,
                               schedule_meta, max_context_len, clean_logits=False, indices=None)
```
`q` 是**普通 Tensor（无 SF）**，强制 fp8、`logits_dtype=fp32`。

---

## 12.（原 §6）Mega MoE（仅 SM100）

### 12.1 分配对称 buffer

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden,
    num_shared_experts=0, mma_type='fp8xfp4', activation='swiglu',
    use_fp8_dispatch=None) -> SymmBuffer   # use_fp8_dispatch 已废弃，仅为兼容保留
```
- `group`：`torch.distributed` ProcessGroup；`group.size()==1` 时退化为普通分配。需要 PyTorch 的 symmetric memory（README 建议 >= 2.9）。
- `mma_type ∈ {'fp8xfp4', 'fp8xfp8', 'bf16xbf16'}`。
- `num_max_tokens_per_rank` 会被**自动向上对齐到 `get_token_alignment_for_mega_moe()` = 1920**。
- 该函数标了 `# TODO: remove`，但仍是主要入口。
- `SymmBuffer` 构造还有 `base` 参数，可复用已分配的底层 buffer（省显存/避免反复 rendezvous）。

`SymmBuffer` 字段与形状（所有 view 建在同一块底层 buffer 上）：

| 字段 | 形状 | dtype |
|---|---|---|
| `x` | `[num_max_tokens_per_rank, hidden]` | fp8（fp8xfp4/fp8xfp8）或 bf16 |
| `x_sf` | `[num_max_tokens_per_rank, hidden/128]` | int32（K-major） |
| `topk_idx` | `[num_max_tokens_per_rank, num_topk]` | int64 |
| `topk_weights` | `[num_max_tokens_per_rank, num_topk]` | fp32 |
| `shared_l1_acts` | **就是 `x` 的别名**（非独立 buffer） | |
| `shared_l1_acts_sf` | `[num_max_shared_sf_tokens, hidden/128]` | int32（列主序） |
| `shared_l2_acts` | `[num_max_tokens_per_rank, intermediate_hidden * num_shared_experts]` | fp8/bf16 |
| `shared_l2_acts_sf` | `[nnn, shared_intermediate_hidden/128]` | int32 |
| `l1_acts` | `[num_ring_tokens, hidden]` | fp8/bf16 |
| `l1_acts_sf` | `[num_sf_ring_tokens, hidden/128]` | int32（M-major） |
| `l2_acts` | `[num_ring_tokens, intermediate_hidden]` | fp8/bf16 |
| `l2_acts_sf` | `[num_sf_ring_tokens, intermediate_hidden/128]` | int32（M-major） |

`activation` 目前**只支持 `'swiglu'**。

### 12.2 权重变换

```python
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
    l1_weights, l2_weights, activation='swiglu')
```
- 输入可为 `(data, sf)` tuple（FP8/FP4）或普通 Tensor（BF16）。
- tuple 路径：L1 做 gate/up **交错**（`_interleave_weights`，gran=8）+ SF 做 4×32 UTCCP 转置（`_transpose_sf_for_utccp`，要求 `mn % 128 == 0`）；L2 只做 SF 转置。
- BF16 路径：L1 只 interleave，L2 完全不变。

### 12.3 执行

```python
deep_gemm.fp8_fp4_mega_moe(
    y, l1_weights, l2_weights, sym_buffer,
    shared_l1_weights=None, shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    recipe=(1,1,32), activation='swiglu',
    activation_clamp=None, fast_math=True)

deep_gemm.bf16_mega_moe(
    y, l1_weights, l2_weights, sym_buffer,
    shared_l1_weights=None, shared_l2_weights=None,
    cumulative_local_expert_recv_stats=None,
    activation='swiglu', activation_clamp=None, fast_math=True)
```
- `y`: `[num_tokens, hidden]` BF16 输出；`num_tokens <= num_max_tokens_per_rank`。
- `recipe` 必须恰好 `(1,1,32)`；`activation_clamp` 默认 `+inf`，需 `>= 0`。
- **权重第一维 E 是"每 rank 专家数"**（num_experts_per_rank），不是全局专家数。
- FP8/FP4 权重：K-major、`(data, sf)` tuple，SFA/SFB 为 `gran_k=32`、UE8M0 packed、MN-major、TMA-aligned int32。L1 权重 `[E, 2*intermediate, hidden]`，L2 `[E, hidden, intermediate]`。fp8xfp8 与 fp8xfp4 由**权重 dtype 自动区分**（FP8 vs packed FP4）；fp8 路径的共享专家权重必须也是 FP8。
- BF16 权重：3D `[E, 2*intermediate, hidden]` / `[E, hidden, intermediate]`，K-major 连续。
- 共享专家权重为 2D（无 group 维），`shared_l1`/`shared_l2` 必须成对给。
- `cumulative_local_expert_recv_stats`：int、`numel == num_experts_per_rank`、连续。
- 辅助：`deep_gemm.get_block_m_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type)`、`deep_gemm.get_token_alignment_for_mega_moe()`（返回 1920）。
- 调试：`DG_COMM_KERNEL_DEBUG=1` 时每次调用后清空对称 buffer（需重新拷贝输入）。

### 12.4 完整示例（`tests/test_mega_moe.py`）

```python
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank, num_topk,
    hidden, intermediate_hidden, num_shared_experts=num_shared_experts,
    mma_type=args.mma_type)

transformed_l1_weights, transformed_l2_weights = (
    deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights))

# 拷贝输入到 buffer 视图（可 fuse 进上游 kernel）
buffer.x[:num_tokens].copy_(x[0])
buffer.x_sf[:num_tokens].copy_(x[1])
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(
    y=y, l1_weights=transformed_l1_weights, l2_weights=transformed_l2_weights,
    sym_buffer=buffer, activation_clamp=args.activation_clamp, fast_math=True)
# 结束
buffer.destroy()
```
多进程需 `torchrun`；测试还提供 DeepEP dispatch + grouped GEMM + TileLang SwiGLU + combine 的非重叠基线对照。

---

## 13.（原 §7）Mega Gate（仅 SM100）

融合 router 线性层、打分、bias/image-bias、专家映射、mask、top-k、权重归一化。

```python
topk_idx, topk_weights = deep_gemm.bf16_mega_gate(
    x,                # [T,H]  BF16，连续
    weight,           # [E,H]  BF16，连续
    num_topk,
    use_shared_as_routed,
    num_shared_experts,
    routed_scaling_factor,
    ep_rank,
    scoring_func='identity',        # 'sigmoid' | 'sqrtsoftplus' | 'identity'
    mask=None,                      # [T] bool
    bias=None,                      # [E] fp32
    image_bias=None,                # [E] fp32（与 image_token_mask 必须同时给）
    image_token_mask=None,          # [T] bool
    fix_routing_mask=None,          # [T] bool（需同时给 unmapped_topk_idx）
    to_physical_map=None,           # [E+S, dup] int32（与 logical_count 同时给）
    logical_count=None,             # [E+S] int32
    unmapped_topk_idx=None,         # [T,K] int64，可非连续，stride(1)==1
    force_random=None,              # [T] bool
    out=None)                       # (topk_idx[T,K'] int64, topk_weights[T,K'] fp32)
```
- `K' = num_topk + (num_shared_experts if use_shared_as_routed else 0)`。
- **排序用 score+bias；输出权重是未加 bias 的 score，在 top-k 内归一化后乘 `routed_scaling_factor`**。
- 约束：`hidden % 256 == 0`；`0 < E <= 512` 且 `E % 4 == 0`；`0 < num_topk <= E` 且 `<= 32`；`ep_rank >= 0`；`K' <= 32`；`num_tokens <= 1<<20`；`routed_scaling_factor` 有限。
- `use_shared_as_routed=True` 时：`num_shared_experts ∈ {1,2}`、`num_topk % num_shared_experts == 0`，且 `num_routed_experts % (num_topk / num_shared_experts) == 0`。
- `T == 0` 直接返回。

```python
deep_gemm.get_bf16_mega_gate_config(num_tokens, hidden, num_routed_experts, num_topk) -> dict
# keys: block_tokens, num_mma_ctas, num_split_k, num_expert_groups, num_gate_warpgroups, num_sms
```

**示例**

```python
topk_idx = torch.empty((T, num_topk), dtype=torch.int64, device='cuda')
topk_weights = torch.empty((T, num_topk), dtype=torch.float32, device='cuda')

deep_gemm.bf16_mega_gate(
    x, weight, num_topk=6, use_shared_as_routed=True, num_shared_experts=1,
    routed_scaling_factor=1.5, ep_rank=0,
    scoring_func='sqrtsoftplus',
    bias=bias, image_bias=image_bias, image_token_mask=image_token_mask,
    to_physical_map=to_physical_map, logical_count=logical_count,
    out=(topk_idx, topk_weights))
```

确定性模式会强制 `num_split_k == 1`：

```python
deep_gemm.use_deterministic_algorithms(True)
assert deep_gemm.get_bf16_mega_gate_config(16, 4096, 256, 6)['num_split_k'] == 1
```

---

## 14.（原 §8）HyperConnection（HC / mHC）

### 14.1 `tf32_hc_prenorm_gemm`

```python
deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None)
```
- `a`: `[M,K]` BF16（K-major）；`b`: `[N,K]` FP32（K-major）；`d`: `[M,N]` FP32（N-major）；`sqr_sum`: `[M]` FP32 = 每行 `sum(a^2)`。
- `num_splits` 给定时 `d` 为 `[num_splits,M,N]`、`sqr_sum` 为 `[num_splits,M]`（split-K 累积）。
- 支持 SM90 / SM100。

**示例**

```python
a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
b = torch.randn((n, k), dtype=torch.float, device='cuda')
d = torch.empty((m, n), dtype=torch.float, device='cuda')
s = torch.empty((m,), dtype=torch.float, device='cuda')
deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=None)
# 参考: ref_d = a.float() @ b.T; ref_s = a.float().square().sum(-1)
```

### 14.2 `mega_mhc`（仅 SM100）

```python
deep_gemm.mega_mhc(
    x, residual, shifted_prev_mix, post_mix, comb_res_mix, fn,
    mix_scales, mix_bases, hc_mult,
    hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, num_sinkhorn_iters,
    rmsnorm_weight, rmsnorm_eps, rmsnorm_scale,
    new_residual, new_prev_mix, new_post_mix, new_comb_res_mix,
    y_bf16=None, y_fp8=None, y_gemm_sf=None, y_routed_sf=None,
    y_shared_sf=None, shared_sf_block_m=0)
```

| 参数 | 形状 | dtype |
|---|---|---|
| `x` | `[T,H]` | bf16 |
| `residual` | `[T,hc_mult,H]` | bf16 |
| `shifted_prev_mix` | `[T,hc_mult,1]` 或 None | fp32 |
| `post_mix` | `[T,hc_mult,1]` | fp32 |
| `comb_res_mix` | `[T,hc_mult,hc_mult]` | fp32 |
| `fn` | `[hc_mult*(hc_mult+2), hc_mult*H]` | fp32 |
| `mix_scales` | `[3]` | fp32 |
| `mix_bases` | `[hc_mult*(hc_mult+2)]` | fp32 |
| `rmsnorm_weight` | `[H]` | bf16 |
| `new_*` 输出 | 与输入对应 | bf16/fp32 |
| `y_bf16` | `[T,H]` | bf16 |
| `y_fp8` | `[T,H]` | e4m3 |
| `y_gemm_sf` | `[T,H/128]` | int32（TMA-aligned 列主序） |
| `y_routed_sf` | `[T,H/128]` | int32（连续行主序） |
| `y_shared_sf` | `[rows,H/128]` | int32（需 `shared_sf_block_m > 0`） |

- `hc_mult` 必须等于 `kNumRoutes == 4`；`num_hc_outputs = hc_mult*(hc_mult+2) = 24`。
- 约束：仅 SM100；`hidden % 1024 == 0`；`num_tokens <= 1<<20`；`num_sinkhorn_iters >= 1`。
- shifted 状态 **全有或全无**（`shifted_prev_mix` / `new_prev_mix` 同时给或同时 None）。
- `y_bf16` / `y_fp8` 至少给一个；给 `y_fp8` 时要么只给 `y_gemm_sf`，要么给 `y_routed_sf`+`y_shared_sf`（`y_gemm_sf` 与 `y_routed_sf` 互斥）。
- 确定性模式下 split-K 固定 16（`kDefaultNumSplits`）。

**示例**

```python
hc_mult = 4
inputs = dict(
    x=torch.randn((T, hidden), dtype=torch.bfloat16, device='cuda'),
    residual=torch.randn((T, hc_mult, hidden), dtype=torch.bfloat16, device='cuda'),
    post_mix=torch.randn((T, hc_mult, 1), dtype=torch.float, device='cuda').sigmoid(),
    comb_res_mix=torch.rand((T, hc_mult, hc_mult), dtype=torch.float, device='cuda'),
    shifted_prev_mix=None,
    fn=torch.randn((24, hc_mult * hidden), dtype=torch.float, device='cuda').mul_(1e-2),
    mix_scales=torch.randn((3,), dtype=torch.float, device='cuda').mul_(0.1),
    mix_bases=torch.randn((24,), dtype=torch.float, device='cuda').mul_(0.1),
    rmsnorm_weight=torch.randn((hidden,), dtype=torch.float, device='cuda').mul_(0.1).add_(1).bfloat16(),
    hc_mult=hc_mult, hc_norm_eps=2e-5, hc_pre_eps=3e-4, hc_post_scale=1.25,
    sinkhorn_eps=2e-6, num_sinkhorn_iters=10, rmsnorm_eps=7e-6, rmsnorm_scale=1.25,
)
outputs = dict(
    new_residual=torch.empty_like(inputs['residual']),
    new_prev_mix=None,
    new_post_mix=torch.empty_like(inputs['post_mix']),
    new_comb_res_mix=torch.empty_like(inputs['comb_res_mix']),
    y_bf16=torch.empty_like(inputs['x']),
)
deep_gemm.mega_mhc(**inputs, **outputs)
```
CUDA graph 场景需先 warmup（split-barrier 初始化有断言，capture 前必须跑过至少一次；`tests/test_mega_mhc.py:150-174` 有完整示范）。

---

## 15.（原 §9）分布式 / 数学工具

### 15.1 `deep_gemm.init_dist` / `uneven_all_gather`

```python
rank, num_ranks, group = deep_gemm.init_dist(local_rank: int, num_local_ranks: int)
out = deep_gemm.uneven_all_gather(tensor, dim=0, group=None)
```
- `init_dist` 读环境变量 `MASTER_ADDR`（默认 `127.0.0.1`）、`MASTER_PORT`（默认 `8361`）、`WORLD_SIZE`、`RANK`，后端 nccl。
- `uneven_all_gather`：交换各 rank 的 dim 大小 → pad 到最大 → all-gather → 裁剪。
- 另有 `deep_gemm.utils.dist.dist_print`（未从 utils 顶层导出）。

### 15.2 量化 / 数学辅助（`deep_gemm.utils.math`，`from deep_gemm import *` 可用）

```python
ceil_div(x: int, y: int) -> int
align(x: int, y: int) -> int
ceil_to_ue8m0(x: Tensor) -> Tensor
pack_ue8m0_to_int(x: Tensor) -> Tensor

per_token_cast_to_fp8(x, use_ue8m0, gran_k=128, use_packed_ue8m0=False) -> (Tensor, Tensor)
per_channel_cast_to_fp8(x, use_ue8m0, gran_k=128) -> (Tensor, Tensor)
per_block_cast_to_fp8(x, use_ue8m0, gran_k=128) -> (Tensor, Tensor)
per_custom_dims_cast_to_fp8(x, dims: Tuple, use_ue8m0) -> (Tensor, Tensor)
per_token_cast_to_fp4(x, use_ue8m0, gran_k=128, use_packed_ue8m0=False) -> (Tensor, Tensor)
transpose_packed_fp4(a: Tensor) -> Tensor
unpack_ue8m0_from_int(packed_sf) -> Tensor
cast_back_from_fp4(packed, sf, gran_k=128, use_packed_ue8m0=False) -> Tensor
cast_back_from_fp8(x_fp8, sf, gran_k=128, use_packed_ue8m0=False) -> Tensor
```

产出 SF 形状：

| 函数 | data | sf |
|---|---|---|
| `per_token_cast_to_fp8` | `(m,n)` fp8 | `(m, ceil_div(n,gran_k))` fp32 / int32(packed) |
| `per_channel_cast_to_fp8` | `(m,n)` fp8 | `(m/gran_k, n)`（要求 m % gran_k == 0） |
| `per_block_cast_to_fp8` | `(m,n)` fp8 | `(ceil_div(m,gran_k), ceil_div(n,gran_k))` |
| `per_token_cast_to_fp4` | `(m, n//2)` int8 | `(m, ceil_div(n,gran_k))` |

注：`use_packed_ue8m0=True` 时 per_token 的 SF 列数会补齐到 4 的倍数（打包需要）。

---

## 16.（原 §10）接口速查表（按架构支持）

| 接口族 | SM90 | SM100 | fp8 | fp4 | bf16 |
|---|---|---|---|---|---|
| `fp8_fp4_gemm_{nt,nn,tn,tt}` | 仅 NT（双 K-major） | NT/TN/NN/TT | ✓ | ✓(SM100) | ✗ |
| `bf16_gemm_{nt,nn,tn,tt}` | ✓ | ✓ | ✗ | ✗ | ✓ |
| `m_grouped_fp8_fp4_gemm_{nt,nn}_contiguous` | NT | NT/NN | ✓ | ✓(SM100) | ✗ |
| `m_grouped_fp8_fp4_gemm_nt_masked` | ✓ | ✓ | ✓ | ✓(SM100) | ✗ |
| `m_grouped_bf16_gemm_{nt,nn}_contiguous` / `_nt_masked` | ✓ | ✓ | ✗ | ✗ | ✓ |
| `k_grouped_fp8_gemm_nt_contiguous` | ✓（c 必填） | ✗ | ✓ | ✗ | ✗ |
| `k_grouped_fp8_gemm_tn_contiguous` | ✗ | ✓ | ✓ | ✗ | ✗ |
| `k_grouped_fp4_gemm_nt_contiguous` | ✗ | ✓ | ✗ | ✓ | ✗ |
| `k_grouped_bf16_gemm_tn_contiguous` | ✓（c 必填） | ✓ | ✗ | ✗ | ✓ |
| `einsum` | ✓ | ✓ | ✗ | ✗ | ✓ |
| `fp8_einsum` | 部分 | ✓ | ✓ | ✗ | ✗ |
| `fp8_fp4_mqa_logits` | fp8 only | ✓ | ✓ | ✓(SM100) | ✗ |
| `fp8_fp4_paged_mqa_logits` | fp8, block_kv=64 | ✓ | ✓ | ✓(SM100) | ✗ |
| sparse MQA 系列 | ✗ | ✓ | ✓ | ✓ | ✗ |
| `tf32_hc_prenorm_gemm` | ✓ | ✓ | ✗ | ✗ | ✓(a) |
| `mega_mhc` | ✗ | ✓ | ✓(out) | ✗ | ✓ |
| Mega MoE | ✗ | ✓ | ✓ | ✓ | ✓ |
| Mega Gate | ✗ | ✓ | ✗ | ✗ | ✓ |

**SM90 与 SM100 的 SF 不兼容**：SM90 要 fp32，SM100 要 packed UE8M0 int32，SF 不能跨架构复用。

---

## 17.（原 §11）常见坑

1. **`fp8_gemm_nt` 不是独立函数**，而是 `fp8_fp4_gemm_nt` 的对象别名——签名/默认值完全一致。
2. **部分 config 函数只能用位置参数**（pybind 未声明 `py::arg`），例如 `set_num_sms(132)`。
3. **分组 GEMM 的对齐是进程级全局状态**：先 `set_mk_alignment_for_contiguous_layout(...)`，再构造 `grouped_layout`。
4. **`d` 是 in-place 写入**；累积时 `c` 常与 `d` 同对象。量化 D 不能累积。
5. **masked 的 `masked_m` 是每 group 有效行数**，不是累积 offset；`grouped_layout` 在 contiguous non-PSUM 下是每行的 group 号，padding 行为 `-1`。
6. **FP4 数据最后一维是 `K/2`**（2 个 FP4/字节，用 int8 承载）。
7. **SM100 的 SF 必须是 2 的幂**（packed UE8M0），否则断言失败。
8. `fp8_gemm_nt_skip_head_mid` 的 SM90 分支内部会先经 `transform_sf_pair_into_required_layout` 填充默认 recipe（fp32 SF 时为 `(1,128,128)`，gran_n=128 ≠ 1 → 走 1d2d），因此 **`recipe=None` 是安全的**，不需要显式传。
9. `mega_mhc` 的 `shifted_prev_mix` / `new_prev_mix` 是位置必填但可传 `None`（可选语义）。
10. Mega MoE / Mega Gate / mHC **仅 SM100**；Mega MoE 需多进程 + symmetric memory + `torchrun`。
11. mqa_logits 系列的**输出由库内部分配并返回**，1024B 行对齐是内部保证，调用方不要自己预分配输出。
12. 第一次调用任何新 shape 都会触发 JIT 编译（秒级），生产环境务必 warmup；缓存目录可用 `DG_JIT_CACHE_DIR` 控制。

---

## 18.（原 §12）测试中可复用的输入构造 helper

来自 `tests/generators.py`（可直接借鉴到自己的代码）：

| helper | 作用 |
|---|---|
| `cast_fp8_fp4_with_major(x, major, gran_k, is_fp4, use_ue8m0, use_block_cast_for_fp8=False)` | 产出按 major 定向的 `(data, sf)` |
| `grouped_cast_fp8_fp4_with_major(...)` | 3D 分组版 |
| `k_grouped_cast_fp8_fp4_with_major(x, ks_cpu, major, use_ue8m0, gran_k, is_fp4, ...)` | K-grouped 版 |
| `generate_normal(m,n,k, major_a,major_b, accumulate, out_dtype, kernel_type, ...)` | 稠密 `(a,b,c,d,ref_d)` |
| `generate_m_grouped_contiguous(...)` | contiguous 分组 `(m,a,b,grouped_layout,d,ref_d,valid_mask)` |
| `generate_m_grouped_masked(...)` | masked 分组 `(a,b,masked_m,d,ref_d,valid_mask)` |
| `generate_k_grouped_contiguous(...)` | K-grouped `(total_k,a,b,c,d,ref_d,grouped_layout,host_ks_cpu)` |
| `QuantConfig((gran_k_a, gran_k_b, is_fp4_a, is_fp4_b))` | 量化配置与误差容忍 `max_diff()` |

`tests/utils.py`：`assert_psum_zero_padding`、`assert_direct_output_matches_fp32_accumulation`、`convert_to_fp8`、`to_cublaslt_vec16_sf_layout`。

---

## 19.（原 §13）最小可运行示例集合

```python
import torch, deep_gemm

# 1) 稠密 FP8 GEMM
m, n, k = 1024, 1024, 1024
a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
a_fp8, a_sf = deep_gemm.per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=128)
b_fp8, b_sf = deep_gemm.per_token_cast_to_fp8(b, use_ue8m0=True, gran_k=128)
d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
a_sf = deep_gemm.get_mn_major_tma_aligned_tensor(a_sf)
b_sf = deep_gemm.get_mn_major_tma_aligned_tensor(b_sf)
deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (b_fp8, b_sf), d)   # 等价于 fp8_fp4_gemm_nt

# 2) BF16 GEMM
d2 = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
deep_gemm.bf16_gemm_nt(a, b, d2)

# 3) 分组（contiguous）
G = 4
deep_gemm.set_mk_alignment_for_contiguous_layout(
    deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout())
gg = torch.repeat_interleave(torch.arange(G, device='cuda', dtype=torch.int32), m // G)
b3 = torch.randn((G, n, k), device='cuda', dtype=torch.bfloat16)
d3 = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b3, d3, gg)

# 4) einsum
s = 8
a4 = torch.randn((s, m, k), device='cuda', dtype=torch.bfloat16)
b4 = torch.randn((s, n, k), device='cuda', dtype=torch.bfloat16)
d4 = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
deep_gemm.einsum('bmk,bnk->mn', a4, b4, d4)
```

---

## 附：本次修订记录（相对上一版）

1. **修正**：删除"skip_head_mid 默认 recipe 抛 bad_optional_access"的坑（不属实，内部会填默认值）。
2. **修正**：`fp8_einsum` 的 SF dtype 是 FP32（SM100 自动/可预先打包为 int32 UE8M0），不是 fp8 e4m3。
3. **修正**："SM90 只支持 NT"限定为 FP8/FP4 内核；BF16 的 nn/tn/tt 在 SM90 可用（transpose 包装 + cuBLASLt）。
4. **修正**：einsum/bf16 "优先 cuBLASLt" 补充条件（非确定性模式且 cuBLASLt 可用）；bf16 的 `alpha` 在 cuBLASLt 路径任意架构可用。
5. **修正**：mqa_logits 输出的 1024B 对齐是库内部保证（输出内部分配），非调用方约束。
6. **补充**：`m_grouped_bf16_gemm_{nt,nn}_contiguous`、`m_grouped_bf16_gemm_nt_masked`、`bf16_gemm_{nn,tn,tt}` 完整签名（§8.2/§8.5）。
7. **补充**：Mega MoE `use_fp8_dispatch` 废弃参数、token 对齐 1920、E 为 per-rank 专家数、SymmBuffer `base` 复用、fp8xfp8/fp8xfp4 按权重 dtype 区分。
8. **补充**：Mega Gate `num_routed_experts % (num_topk/num_shared_experts) == 0`、`num_tokens <= 1<<20`。
9. **补充**：mega_mhc 仅 SM100、`hidden % 1024 == 0`、`y_gemm_sf`/`y_routed_sf` 互斥。
10. **新增**：第一部分入门指南（§1–§6：概念图解、上手示例、决策树、报错速查）。
