# small-exp2：Qwen3-8B/vLLM 上的 K 与 S 容忍度标定

## 1. 任务定位

本实验使用本地 Qwen3-8B 和 vLLM，回答三个最基础的问题：

1. 本次 attention score $S$ 出现多大误差时，模型结果仍然可接受？
2. KV cache 中的 K 出现存储误差并被后续 query 重复读取时，模型能够容忍到什么程度？
3. K 误差经过真实 query 后，会形成怎样的 S 误差分布？

第一阶段只做上述三项，不开展以下系统级实验：

- 不判断或优化 K block 大小；
- 不开展完整 BER campaign；
- 不实现 ABFT、DMR、checksum 或纠错流程；
- 不标定 ABFT 残差阈值 $\tau_0,\tau_1$；
- 不要求复现 `small-exp1` 的 64-token 硬件组织。

vLLM 内部的 page/block 只用于定位和修改 KV cache，不作为实验变量或正确性判据。后续再把本实验得到的模型容忍度与 `small-exp1` 的 block、BER 和 ABFT 结果连接。

## 2. 核心误差模型

对第 $t$ 个 query，clean score 为

$$
S_t^{\mathrm{clean}}=Kq_t.
$$

设 K 存储错误为 $E_K$，本次计算产生的附加 score 误差为 $E_{C,t}$，则

$$
\widetilde K=K+E_K,
$$

$$
\widetilde S_t=(K+E_K)q_t+E_{C,t}.
$$

总 score 误差为

$$
\boxed{
E_{S,t}
=\widetilde S_t-S_t^{\mathrm{clean}}
=E_Kq_t+E_{C,t}
}.
$$

因此需要分开标定：

- **S 层容忍度**：将 $E_{S,t}$ 作为本次 score 的总误差，不区分它来自存储还是计算；
- **K 层容忍度**：单独研究持久 $E_K$ 被未来多个 $q_t$ 读取后的累计影响；
- **K→S 传播**：测量 $E_Kq_t$ 随 layer、head、query 和时间变化的分布。

## 3. 两层正确性定义

### 3.1 S 层：本次 query 是否可接受

记绝对容差为 $a_S=\mathtt{atol}_S$，相对容差为
$r_S=\mathtt{rtol}_S$。对于所有有效、未被 causal mask 排除的 score：

$$
\left|
\widetilde S_{t,i}-S_{t,i}^{\mathrm{clean}}
\right|
\le
a_S+r_S\left|S_{t,i}^{\mathrm{clean}}\right|.
$$

整个 score 向量的通过条件为

$$
\operatorname{Pass}_S(t;a_S,r_S)
=
\bigwedge_{i\in\mathcal V_t}
\left[
\left|\widetilde S_{t,i}-S_{t,i}^{\mathrm{clean}}\right|
\le a_S+r_S\left|S_{t,i}^{\mathrm{clean}}\right|
\right].
$$

其中 $\mathcal V_t$ 是当前 query 的有效 score 下标集合。mask 产生的 `-inf` 不参与比较；额外 NaN/Inf 一律判失败。

S 层最终需要给出可直接使用的：

$$
\boxed{
\mathtt{rtol}_S,\qquad \mathtt{atol}_S
}.
$$

### 3.2 K 层：存储错误是否在生命周期内有害

K 是否发生物理错误定义为

$$
F_K=\mathbf 1[\widetilde K\ne K].
$$

但 K 错误是否影响模型依赖未来 query。设错误 K 在 cache 中继续服务 $L$ 次 query，定义生命周期有害事件：

$$
H_K^{(L)}
=
\mathbf 1
\left[
\exists u\in\{t,\ldots,t+L-1\}:
\text{模型层指标超过允许预算}
\right].
$$

K 层主要输出风险曲线：

$$
P\left(
H_K^{(L)}=1
\mid
\text{K 错误类型、幅度、位置和 }L
\right).
$$

第一阶段不要强行把 K 容忍度压缩成唯一一组 `K_rtol/K_atol`。如果必须给出工程阈值，应同时注明错误稀疏度、持久时间 $L$ 和允许风险。

## 4. 实验基准

### 4.1 模型与推理框架

- 模型：本地 Qwen3-8B，模型具体 revision 以本地文件为准；
- 框架：vLLM；
- tokenizer、权重、dtype、KV-cache dtype 和 attention backend 必须固定并记录；
- q、V、模型权重默认不注错；
- generation 使用 greedy decoding，或固定所有 sampling seed；
- 同一条样本的 clean 与 faulted 运行必须配对。

不得假定 head dimension。必须从模型配置读取 $d_h$。

### 4.2 Clean reference

所有错误影响都相对于同一 vLLM 配置下的无故障结果：

$$
S_t^{\mathrm{clean}},\quad
P_t^{\mathrm{clean}},\quad
Y_t^{\mathrm{clean}},\quad
\operatorname{logits}_t^{\mathrm{clean}}.
$$

其中：

- $S_t$：softmax 前 attention score；
- $P_t=\operatorname{softmax}(S_t)$：attention probability；
- $Y_t=P_tV$：attention output。

不得把“读取错误 K 后正确重算得到的 score”作为 clean reference。

### 4.3 Raw 与 scaled score

同时记录：

$$
\begin{aligned}
S_{\mathrm{raw}}&=QK^\top,\\
S_{\mathrm{scaled}}&=\frac{QK^\top}{\sqrt{d_h}}.
\end{aligned}
$$

模型影响以实际送入 softmax 的 scaled score 为主。若不存在其他缩放，则换算为：

$$
\boxed{
\begin{aligned}
r_{S,\mathrm{raw}}&=r_{S,\mathrm{scaled}},\\
a_{S,\mathrm{raw}}&=\sqrt{d_h}\,a_{S,\mathrm{scaled}}.
\end{aligned}}
$$

## 5. 第一阶段必须完成的实验

第一阶段只要求下面三个实验。先完成 smoke，再扩大必要样本，不进行大范围系统扫描。

### 实验 A：S 直接扰动与容忍度

在 softmax 前对 scaled score 注入可控误差：

$$
\widetilde S_t=S_t^{\mathrm{clean}}+E_{S,t}.
$$

只保留三类最重要的误差形态：

1. **单 score 随机位置误差**：代表稀疏计算或存储传播错误；
2. **稀疏随机误差**：少量 score 同时受影响；
3. **top-1/top-2 定向误差**：使两个最大 score 的间隔缩小，用作敏感性压力测试。

初始幅度采用粗网格：

$$
\gamma_S\in
\{10^{-6},10^{-5},10^{-4},10^{-3},10^{-2},10^{-1}\}.
$$

分别测试：

- 绝对误差 $|E_{S,t,i}|=\gamma_S$；
- 相对误差 $|E_{S,t,i}|=\gamma_S|S_{t,i}^{\mathrm{clean}}|$。

根据模型质量开始变化的位置增加局部点，不需要一开始运行完整二维密集网格。

候选 S 容差先使用：

$$
\begin{aligned}
r_S&\in\{10^{-6},10^{-5},10^{-4},10^{-3},10^{-2}\},\\
a_S&\in\{10^{-7},10^{-6},10^{-5},10^{-4},10^{-3}\}.
\end{aligned}
$$

必须包含当前临时值：

$$
(r_S,a_S)=(10^{-5},10^{-6}).
$$

### 实验 B：持久 K-cache 误差

直接修改 vLLM 中已经写入的 K cache，并让错误在后续 query 中持续存在。不要每个 query 后恢复 clean K。

第一阶段只测试：

1. 单个 K 元素的可控数值误差；
2. 单个 FP16/BF16 bit flip，按实际 KV-cache dtype 分为 sign、exponent、mantissa；
3. 每个样本只设置一个 K 故障，避免第一阶段被多错误组合淹没。

对数值误差定义：

$$
\widetilde K_{p,j}
=K_{p,j}+\delta_K.
$$

粗扫描：

$$
\frac{|\delta_K|}{|K_{p,j}|+\epsilon}
\in
\{10^{-6},10^{-5},10^{-4},10^{-3},10^{-2},10^{-1}\},
$$

其中 $\epsilon$ 用于处理接近 0 的 K 元素，并同时保存实际绝对误差。

在故障注入后连续记录后续 query，至少区分：

- 当前/下一次 query；
- 后续 16 次 query；
- 直到样本结束的完整剩余生命周期。

### 实验 C：K→S 传播

实验 B 的每个 K 错误都必须记录它在后续 query 中产生的 score 误差：

$$
E_{S,u}^{(K)}
=\widetilde S_u-S_u^{\mathrm{clean}}
=E_Kq_u.
$$

至少输出：

- K 误差幅度与 $\max_i|E_{S,u,i}^{(K)}|$ 的关系；
- 当前 q 未放大、未来 q 放大的比例；
- K 错误在未来至少一次超过 S 容差的概率；
- sign/exponent/mantissa 三类 bit flip 的传播差异；
- early/middle/late layer 的差异。

核心统计为：

$$
P\left(
\exists u\le L:
\neg\operatorname{Pass}_S(u;a_S,r_S)
\mid E_K
\right).
$$

## 6. 最小采样设计

### 6.1 数据

第一阶段只要求一个可复现的通用语言建模验证集，同时保留少量固定生成 prompt。

必须保存：

- 数据集名称、版本和 split；
- tokenizer 和预处理方式；
- 样本 ID；
- 数据摘要哈希；
- calibration/test 划分。

不得在 test split 上选择容差。

### 6.2 模型位置采样

第一阶段不遍历所有 layer/head，只选择：

- early layer：靠近模型前部的一层；
- middle layer：中间一层；
- late layer：靠近输出的一层；
- 每层随机选择至少 2 个有效 head；
- query 位置覆盖短上下文和长上下文位置。

具体层号必须根据本地 Qwen3-8B 配置自动确定并记录。

### 6.3 规模

建议：

- smoke：16 条序列，每条至少 256 tokens，1 个 seed；
- 第一阶段正式实验：不少于 50,000 个 clean 有效 token；
- 每个主要扰动条件不少于 500 个独立注入位置；
- 至少 3 个独立随机 seed；
- 参数选择完成后，在独立 test split 上复核一次。

如果算力不足，优先保证：配对 clean/faulted、三个层位、三个 seed 和独立 test；不要优先扩展错误类型。

## 7. 第一阶段指标

### 7.1 主要模型指标

第一阶段必须报告：

- clean/faulted token-level NLL；
- PPL 及相对 PPL 变化；
- greedy next-token top-1 改变率；
- logits 最大绝对误差和相对 L2 误差；
- attention probability 的 total variation distance；
- attention output 的相对 L2 误差；
- NaN/Inf 数量；
- 生成序列首次与 clean 结果分叉的位置。

PPL 不能作为唯一判据，因为平均值可能掩盖少数严重错误。

### 7.2 S 容差的风险指标

将模型层结果标为“可接受”或“有害”后，对每个 $(a_S,r_S)$ 统计：

$$
R_{\mathrm{harmful\ pass}}
=P\left(
\operatorname{Pass}_S
\mid
\text{模型层结果有害}
\right),
$$

$$
R_{\mathrm{benign\ reject}}
=P\left(
\neg\operatorname{Pass}_S
\mid
\text{模型层结果可接受}
\right).
$$

优先降低 $R_{\mathrm{harmful\ pass}}$，再考虑降低误拒绝。

### 7.3 暂定模型质量预算

在项目负责人没有给出新的应用预算前，第一阶段暂时使用：

- 相对 PPL 上升不超过 `0.1%`；
- 相对 PPL 上升的 95% CI 上界不超过 `0.2%`；
- greedy top-1 改变率不超过 `0.1%`；
- 不出现额外 NaN/Inf；
- 必须检查 p95、p99 和最坏样本。

这些只是初始选择口径，不是论文的最终科学结论。必须保存完整曲线，便于后续调整预算。

## 8. vLLM 实现要求

vLLM 可能使用 fused attention，无法直接暴露完整 S。允许两种实现：

1. 在 vLLM attention/backend 中加入仅用于实验的 score/K 注入与观测接口；
2. 对选定层使用可观测的 reference attention 路径。

如果使用 reference 路径，必须先验证无注错时它与原 vLLM 的一致性，至少比较：

- attention output；
- 最终 logits；
- greedy token；
- PPL。

K 错误必须写入实际会被后续 query 继续读取的 cache 页面，不能只修改临时张量。

所有实验必须固定并记录：

- model/tokenizer revision；
- vLLM、PyTorch、CUDA 和 GPU 版本；
- weights、activation 和 KV-cache dtype；
- attention backend；
- seed 和确定性设置。

## 9. Conda、接口与输出

使用 Conda 管理环境，环境名建议为：

```text
pim-kv-small-exp2
```

至少提供：

```text
small-exp2/
├── README.md
├── TASK_SPEC.md
├── environment.yml
├── run_experiment.py
├── configs/
├── smallexp2/
├── tests/
└── results/
```

CLI 至少支持：

```text
--model-path
--dataset-path / --dataset-name
--profile smoke|phase1
--seed
--layers
--heads
--output
--resume
```

结果目录至少包含：

```text
results/<run>/
├── config.json
├── environment.json
├── raw/
├── tables/
│   ├── baseline_metrics.csv
│   ├── s_tolerance.csv
│   ├── k_tolerance.csv
│   ├── k_to_s_transfer.csv
│   └── recommendation.json
├── figures/
└── REPORT.md
```

所有比例必须保存 numerator、denominator、estimate 和 95% CI。NLL/PPL 使用按序列 paired bootstrap；比例使用 Wilson 95% CI。

## 10. 第一阶段图表

只要求最重要的图：

1. S 误差幅度—相对 PPL/top-1 改变率曲线；
2. `rtol × atol` 的 harmful-pass 风险热图；
3. K 误差幅度—生命周期有害概率曲线；
4. K 误差—未来最大 S 误差的传播散点/分位数图；
5. sign/exponent/mantissa 分层对比；
6. early/middle/late layer 对比。

图表同时导出 PDF 和 300-DPI PNG。

## 11. 第一阶段交付结论

`recommendation.json` 和 `REPORT.md` 必须回答：

### S 层

```text
strict_s_rtol_scaled
strict_s_atol_scaled
balanced_s_rtol_scaled
balanced_s_atol_scaled
recommended_s_rtol_raw
recommended_s_atol_raw
```

其中 raw 值可以直接作为 `small-exp1` 的候选：

```text
result_rtol = <recommended_s_rtol_raw>
result_atol = <recommended_s_atol_raw>
```

### K 层

- 不同 K 数值误差的生命周期有害概率；
- sign/exponent/mantissa bit flip 的风险；
- 当前 q 无害但未来 q 变为有害的比例；
- 风险随剩余 query 数 $L$ 的变化；
- 是否有证据支持给出单一 K 容差；若没有，应明确使用风险曲线。

### K→S

- K 误差映射到 S 误差的经验分布；
- K 错误导致 S 超过推荐容差的条件概率；
- layer、head、query 位置的主要差异。

## 12. 第一阶段完成标准

满足以下条件才算完成：

1. Conda 环境能够由 `environment.yml` 创建；
2. vLLM clean baseline 可复现；
3. 完成 S 的三类关键扰动；
4. 完成单 K 数值错误和 bit flip 的持久 cache 实验；
5. 同时记录当前 query 和后续 query 的 K→S 传播；
6. 给出 S 的严格值、平衡值及置信区间；
7. 给出 K 生命周期风险曲线，不用单次 q 代替；
8. 独立 test split 未参与参数选择；
9. 所有原始计数、配置、seed 和环境信息完整保存；
10. 报告明确写出限制，不对 block、BER 或完整 ABFT 可靠性做超出实验范围的结论。

## 13. 后续扩展（第一阶段不做）

第一阶段完成后，再根据结果决定是否扩展：

- 全部 layer/head；
- 更多模型规模和架构；
- 更多语言建模与下游任务数据集；
- 多 K 错误和不同稀疏度；
- 按真实 BER 的错误注入；
- 不同 KV-cache dtype/量化格式；
- 64-token 或其他硬件 block 映射；
- 与 `small-exp1` 的 ABFT 阈值和纠错策略联合优化。

这些扩展用于论文完整性和泛化性，但不应阻塞第一阶段得到 K/S 容忍度基线。
