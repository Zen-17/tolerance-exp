# 实验 A：S 直接扰动与容忍度

对照 `TASK_SPEC.md` 第 3.1、4、5（实验 A）、6、7.1–7.3、8、10.1–10.2、11（S 层）。

## 问题

softmax 前 scaled score 允许多大误差时模型结果仍可接受？给出
`strict/balanced` 的 scaled `rtol_S, atol_S` 及 raw 换算（`result_rtol/result_atol`）。

## 设计

1. 选定层走 reference attention：先 `reshape_and_cache_flash` 写真实 paged KV，再算
   \(S_{\mathrm{scaled}}=QK^\top/\sqrt{d_h}\)（\(d_h\) 从本地 Qwen3-8B `config.json` 读取）。
2. \(S_{\mathrm{raw}}=QK^\top=S_{\mathrm{scaled}}\sqrt{d_h}\)。模型影响以 scaled 为准。
3. 只在**本次 query**（prefill 最后一行，对应下一个 token）注入
   \(\widetilde S=S^{\mathrm{clean}}+E_S\)，decode 不再注错。q、V、权重不注错。
4. clean 与 faulted 配对、同一 reference 路径、greedy decoding。
5. 无注错时对比 flash vs reference：attention output、logits、greedy token、PPL。

### 三类 \(E_S\)（规格第 5 节）

| 形态 | 含义 |
|------|------|
| single | 选定 head 上 1 个有效 score |
| sparse | 少量有效 score（约 1%，上限 8） |
| top2_gap | 缩小当前 query 最大两个 score 的间隔 |

\(\gamma_S\in\{10^{-6},\ldots,10^{-1}\}\)；绝对 \(|E|=\gamma\) 与相对 \(|E|=\gamma|S|\) 都做。
质量拐点处再加密。候选容差含 \((r_S,a_S)=(10^{-5},10^{-6})\)。

### 采样（规格第 6 节）

- 可复现 LM 集 `smallexp2_synthetic_lm_v1`，calibration/test 划分；不在 test 上选容差。
- smoke：16 条序列，每条 \(\ge 256\) tokens，1 个 seed。
- 层号由配置自动取 early/middle/late；每层 \(\ge 2\) 个 head。
- query 覆盖短上下文（64）与长上下文（256）。

### 有害预算（规格 7.3）

相对 PPL 升 \(\le 0.1\%\)（95% CI 上界 \(\le 0.2\%\)）；greedy top-1 改变率 \(\le 0.1\%\)；无额外 NaN/Inf；看 p95/p99/最坏样本。优先降 \(R_{\mathrm{harmful\ pass}}\)。

## 代码路径

```text
run_exp_a.py          # smoke | phase1
smallexp2/data.py     # 256-token 序列、split、哈希
smallexp2/hooks.py    # reference S、注入、Y/logits
smallexp2/faults.py   # 三类 E_S 与 Pass_S
smallexp2/analyze_a.py
```
