# 实验 C：K→S 传播

对照 `TASK_SPEC.md` 第 3.2、5（实验 C）、7.2、8、10.4–10.6、11（K→S）。

C **不引入新的故障类型**。它回答：实验 B 写入 paged K-cache 的单个 \(E_K\)，经真实 \(q_u\) 变成怎样的 \(E_S\)，何时超过实验 A 的 \(\mathrm{Pass}_S\)。

注入物理与 B 相同；B 的 `trials.jsonl` 是 C 的主数据。C 另做一次 **恒等式探针**，核对
\(\widetilde S-S^{\mathrm{clean}}=E_K q/\sqrt{d_h}\) 是否在选定层的 reference 路径上成立。

## 问题

1. \(|E_K|\) 如何映射到 \(\max_i|E_{S,u,i}^{(K)}|\)？（经验分布 + 分位数）
2. 当前 query 未放大、未来 query 放大的比例？
3. \(P(\exists u\le L:\neg\mathrm{Pass}_S(u;a_S,r_S)\mid E_K)\) 随 \(|E_K|\)、\(L\)、bit 类、layer、head、上下文如何变？
4. 实测 \(E_S\) 是否等于 \(E_K q_{\cdot,j}\cdot d_h^{-1/2}\)（只出现在损坏 key 所在列、以及共享该 KV head 的 GQA query heads）？

**不**把传播压成单一 `K_rtol`。输出条件概率与增益曲线。test split 不参与选择。

## 设计

### 数据来源

| 来源 | 作用 |
|------|------|
| `results/expB_<profile>/raw/trials.jsonl` | 主分析：每个 K 故障后 16 次 query 的 \(\max|E_S|\)、\(\mathrm{Pass}_S\) |
| 恒等式探针（本实验小规模重跑） | 记录损坏 key 列上的 \(E_S\)、对应 \(q_{\cdot,j}\)、预测值 |

\(a_S,r_S\) 与 B 相同：优先实验 A balanced scaled，否则 \((10^{-5},10^{-6})\)。

### 传播恒等式

单元素 \(K_{p,h_{\mathrm{kv}},j}\leftarrow K+\delta_K\) 时，scaled score 只在 key 位置 \(p\)、且 query head 属于该 KV group 时改变：

\[
E_{S,u,h,p}
=\frac{\delta_K\,q_{u,h,j}}{\sqrt{d_h}},
\quad h\in\mathrm{GQA}(h_{\mathrm{kv}}).
\]

探针在 gather 副本上还原该元素得到 \(S^{\mathrm{clean}}\)，与损坏 cache 的 \(\widetilde S\) 相减；预测值为 \(|\delta_K|\cdot\max_{h\in\mathrm{GQA}}|q_{u,h,j}|\cdot d_h^{-1/2}\)。
不得把“读了错误 K 再正确重算”当作 clean。\(d_h\) 只从本地 `config.json` 读。

### 核心统计

- \(|E_K|\) 分箱上的 \(\max|E_S|\) 的 p50/p90，以及 \(P(\neg\mathrm{Pass}_S\mid\mathrm{bin},L)\)，\(L\in\{1,2,16,\mathrm{full}\}\)
- 增益 \(g_u=\max|E_{S,u}|/(|E_K|+\epsilon)\) 的分布（有限值；exponent 溢出单独计数）
- 未来放大：\(\max_{u>0}|E_S|>\max|E_{S,0}|\)
- sign / exponent / mantissa；early / middle / late；KV head；短 64 / 长 256；query 下标 \(u\)

### 采样

主分析沿用 B 的 calibration 试验。探针：同一语料前若干 calibration 序列、一层、短上下文、数值相对误差 \(10^{-4},10^{-3},10^{-2}\) 加 mantissa bit，用于核对线性关系。test 不用。

## 代码路径

```text
tolerance-exp/
├── docs/EXP_C_K_TO_S.md
├── run_exp_c.py                 # --from-b 分析；默认再跑恒等式探针
├── smallexp2/
│   ├── analyze_c.py
│   ├── hooks.py                 # 记录 q、预测 E_S、损坏 key 列上的 E_S
│   └── k_faults.py              # 复用 B
├── tests/test_exp_c_transfer.py
└── results/expC_<profile>/
    ├── config.json
    ├── environment.json
    ├── raw/{b_source.json, identity_trials.jsonl}
    ├── tables/
    │   ├── k_to_s_transfer.csv
    │   ├── k_to_s_by_ek_bin.csv
    │   ├── k_to_s_by_query.csv
    │   ├── k_to_s_by_layer.csv
    │   ├── k_to_s_by_head.csv
    │   ├── k_to_s_by_bitclass.csv
    │   ├── identity_check.csv
    │   └── recommendation.json
    ├── figures/                 # 规格图 4–6 + 增益/query 曲线
    └── REPORT.md
```

```text
PYTHONUNBUFFERED=1 python run_exp_c.py --profile smoke --from-b results/expB_smoke
PYTHONUNBUFFERED=1 python run_exp_c.py --profile smoke --from-b results/expB_smoke --skip-identity
```
