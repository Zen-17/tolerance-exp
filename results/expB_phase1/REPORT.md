# Experiment B report (K lifetime + K→S)

## K layer

No single K_rtol/K_atol. Use lifetime risk curves in k_tolerance.csv.
Sparsity: one K element per sample. S-tol used: {'s_rtol_scaled': 1e-06, 's_atol_scaled': 1e-05, 'source': 'expA_balanced_scaled', 'path': '/opt/data/data/tolerance-exp/results/expA_phase1/tables/recommendation.json'}.

- H_K L=1: 0.1854 [0.1744, 0.1970]
- H_K L=16: 0.3201 [0.3067, 0.3338]
- H_K full: 0.3201 [0.3067, 0.3338]
- delayed harm (current q OK, later q harmful): 0.1347
- rel PPL p95/p99/worst: 0.01042 / 0.04441 / 0.534

## K→S

- future-q amplified: 0.4852 [0.4707, 0.4998]
- P(exists u: not Pass_S) full lifetime: 0.5516
- by bit class: {"sign": {"n": 504, "p_fail_s_full": {"numerator": 504, "denominator": 504, "estimate": 1.0, "ci95_low": 0.9924354365613216, "ci95_high": 1.0}, "mean_max_es_future_finite": 0.582397399632822, "n_finite": 504, "n_inf": 0, "n_nan": 0}, "exponent": {"n": 504, "p_fail_s_full": {"numerator": 504, "denominator": 504, "estimate": 1.0, "ci95_low": 0.9924354365613216, "ci95_high": 1.0}, "mean_max_es_future_finite": 9.88525617037669e+36, "n_finite": 300, "n_inf": 103, "n_nan": 101}, "mantissa": {"n": 504, "p_fail_s_full": {"numerator": 494, "denominator": 504, "estimate": 0.9801587301587301, "ci95_low": 0.9638653632818068, "ci95_high": 0.9891877146858115}, "mean_max_es_future_finite": 0.0015922982498840798, "n_finite": 504, "n_inf": 0, "n_nan": 0}}
- by layer: {"4": {"n": 1512, "p_fail_s_full": {"numerator": 835, "denominator": 1512, "estimate": 0.5522486772486772, "ci95_low": 0.527082795419569, "ci95_high": 0.5771497312507637}, "mean_max_es_future_finite": 7.06926757272522e+35, "n_finite": 1439, "n_inf": 44, "n_nan": 29}, "18": {"n": 1512, "p_fail_s_full": {"numerator": 833, "denominator": 1512, "estimate": 0.5509259259259259, "ci95_low": 0.5257565067077646, "ci95_high": 0.5758372218190154}, "mean_max_es_future_finite": 6.53955201585825e+35, "n_finite": 1445, "n_inf": 29, "n_nan": 38}, "31": {"n": 1512, "p_fail_s_full": {"numerator": 834, "denominator": 1512, "estimate": 0.5515873015873016, "ci95_low": 0.5264196287445115, "ci95_high": 0.5764934988540451}, "mean_max_es_future_finite": 6.92917114023709e+35, "n_finite": 1448, "n_inf": 30, "n_nan": 34}}

## Consistency

{
  "layer": 4,
  "ctx": 256,
  "first_divergence": null,
  "top1": {
    "numerator": 0,
    "denominator": 16,
    "estimate": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.19361341827271994,
    "n_compared": 16
  },
  "ppl_flash": 1.0006373621467188,
  "ppl_ref": 1.0006558297294483,
  "logits": {
    "max_abs": 0.171875,
    "rel_l2": 0.007527731947920375,
    "has_nan_inf": false
  },
  "attn_output_max_abs": 0.0019509196281433105,
  "n_flash_tokens": 16,
  "n_ref_tokens": 16
}

## Limits

Does not study K-block size, BER, or ABFT. Test split was not used for selection.
