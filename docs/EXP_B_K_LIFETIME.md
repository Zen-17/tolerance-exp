# 实验 B：持久 K-cache 误差（含实验 C：K→S 传播）

对照 `TASK_SPEC.md` 第 3.2、4、5（实验 B）、6、7、8、10.3、11（K 层）。

每一次 K 注入都记录后续 query 的 \(E_S^{(K)}=E_K q\)，供实验 C 使用。K→S 的交付见 `docs/EXP_C_K_TO_S.md`。

## 问题

1. 已写入 KV cache 的单个 K 元素出错后，被后续多次 query 重复读取，生命周期有害概率是多少？
2. \(E_K\) 经真实 \(q_u\) 形成怎样的 \(S\) 误差？何时超过实验 A 的 S 容差？

**不**把 K 容忍度压成单一 `K_rtol/K_atol`。输出风险曲线，并注明稀疏度（单元素）、\(L\) 和允许风险。

## 设计

1. 选定层：`reshape_and_cache_flash` 把当前步 K/V 写入**真实 paged cache**。
2. **立刻、只一次**修改该 cache 中一个已存在的 K 元素；后续 query **不恢复**。
3. q、V、权重不注错；每个样本只设一个 K 故障。
4. 用 reference attention 从（可能已损坏的）cache gather K，算
   \(S^{\mathrm{clean}}\)（把该元素在 **gather 副本** 上还原）与
   \(\widetilde S\)（真实损坏 K），得到 \(E_S=\widetilde S-S^{\mathrm{clean}}=E_K q\)。
   不得把“读了错误 K 再正确重算”当作 clean。
5. clean 与 faulted 配对、greedy；faulted 与 clean 走同一 reference 层。

### 注入时机与生命周期 \(L\)

Prefill 写完 prompt 的全部 K 后、softmax 前注入，目标为 prompt 中部已缓存 token \(p\)。

| 窗口 | 含义 |
|------|------|
| \(L=1\) | 当前 query（prefill 最后一行） |
| \(L=2\) | 当前 + 下一次 decode |
| \(L=16\) | 注入后至多 16 次 query |
| \(L=\mathrm{full}\) | 直到本次生成结束 |

\[
H_K^{(L)}=\mathbf{1}[\exists u<L:\text{该 query 模型指标超预算}]
\]

暂定预算与实验 A 相同（相对 PPL 升 \(>0.1\%\)、top-1 改变、NaN/Inf）。

### 两类故障（规格第 5 节，第一阶段只做这些）

| 类型 | 定义 | 扫描 |
|------|------|------|
| numeric | \(\widetilde K_{p,j}=K_{p,j}+\delta_K\) | \(\|\delta_K\|/(\|K\|+\epsilon)\in\{10^{-6},\ldots,10^{-1}\}\)，同时存绝对 \(\delta_K\) |
| bitflip | 按 **实际 KV-cache dtype**（本机 Qwen3-8B 为 BF16）翻 1 bit | sign / exponent / mantissa 各抽 1 bit |

BF16：bit15=sign，bit7–14=exponent，bit0–6=mantissa。smoke 取 15、14、0。

### K→S（实验 C，随 B 记录）

对注入后每个 query \(u\) 记录 \(\max_i|E_{S,u,i}|\)、是否 \(\mathrm{Pass}_S(u;a_S,r_S)\)。
\(a_S,r_S\) 优先用实验 A `recommendation.json` 的 balanced scaled，否则 \((10^{-5},10^{-6})\)。

至少统计：

- \(|E_K|\) vs 未来 \(\max|E_S|\)
- 当前 q 未放大、未来 q 放大的比例（\(\max|E_{S,u>0}| > \max|E_{S,0}|\)）
- \(P(\exists u\le L:\neg\mathrm{Pass}_S\mid E_K)\)
- sign / exponent / mantissa 差异
- early / middle / late 差异

### 采样

与实验 A 同一语料 `smallexp2_synthetic_lm_v1`。smoke：16 条 \(\ge 256\) token 序列、1 seed、层 4/18/31 轮换、每层 2 个 **KV head**、短 64 / 长 256 上下文。test 不参与选参。

## 代码路径

入口：`run_exp_b.py --profile smoke|phase1`。`--heads` 是 **KV head**（Qwen3-8B 共 8 个）。`--s-rec` 默认读实验 A `results/expA_smoke_spec/tables/recommendation.json` 的 balanced scaled；缺文件则用 \((10^{-5},10^{-6})\)。

```text
tolerance-exp/
├── docs/EXP_B_K_LIFETIME.md      # 本文件
├── run_exp_b.py                  # 入口 smoke|phase1
├── smallexp2/
│   ├── k_faults.py               # numeric / BF16 bitflip
│   ├── hooks.py                  # 持久写 cache + 观测 E_S
│   ├── analyze_b.py              # 风险曲线与 K→S 表
│   ├── data.py                   # 复用 A
│   ├── geometry.py               # 复用 A
│   ├── metrics.py                # 复用 A
│   └── env_info.py               # 复用 A
├── tests/test_exp_b_k_faults.py
└── results/expB_<profile>_<stamp>/
    ├── config.json
    ├── environment.json
    ├── raw/trials.jsonl
    ├── tables/
    │   ├── baseline_metrics.csv
    │   ├── k_tolerance.csv
    │   ├── k_to_s_transfer.csv
    │   └── recommendation.json
    ├── figures/                  # 规格图 3–6，PDF+300DPI PNG
    └── REPORT.md
```

实验 A 文件不改入口；B 只扩展 `hooks.py` 的 K 注入分支，避免双 patch。
