# Experiment B report (K lifetime + K→S)

## K layer

No single K_rtol/K_atol. Use lifetime risk curves in k_tolerance.csv.
Sparsity: one K element per sample. S-tol used: {'s_rtol_scaled': 1e-05, 's_atol_scaled': 1e-05, 'source': 'expA_balanced_scaled', 'path': '/opt/data/data/tolerance-exp/results/expA_smoke_spec/tables/recommendation.json'}.

- H_K L=1: 0.1771 [0.1373, 0.2253]
- H_K L=16: 0.3125 [0.2617, 0.3682]
- H_K full: 0.3125 [0.2617, 0.3682]
- delayed harm (current q OK, later q harmful): 0.1354
- rel PPL p95/p99/worst: 0.01042 / 0.03517 / 0.1154

## K→S

- future-q amplified: 0.4826 [0.4255, 0.5402]
- P(exists u: not Pass_S) full lifetime: 0.5451
- by bit class: {"sign": {"n": 32, "p_fail_s_full": {"numerator": 32, "denominator": 32, "estimate": 1.0, "ci95_low": 0.8928172849426366, "ci95_high": 1.0}, "mean_max_es_future_finite": 0.5623486433178186, "n_finite": 32, "n_inf": 0, "n_nan": 0}, "exponent": {"n": 32, "p_fail_s_full": {"numerator": 32, "denominator": 32, "estimate": 1.0, "ci95_low": 0.8928172849426366, "ci95_high": 1.0}, "mean_max_es_future_finite": 7.657176350084124e+36, "n_finite": 21, "n_inf": 7, "n_nan": 4}, "mantissa": {"n": 32, "p_fail_s_full": {"numerator": 30, "denominator": 32, "estimate": 0.9375, "ci95_low": 0.7985253275199642, "ci95_high": 0.9826897968048427}, "mean_max_es_future_finite": 0.0016058990731835365, "n_finite": 32, "n_inf": 0, "n_nan": 0}}
- by layer: {"4": {"n": 108, "p_fail_s_full": {"numerator": 58, "denominator": 108, "estimate": 0.5370370370370371, "ci95_low": 0.44334384333440474, "ci95_high": 0.6281858915763978}, "mean_max_es_future_finite": 7.446586519490824e+35, "n_finite": 104, "n_inf": 2, "n_nan": 2}, "18": {"n": 90, "p_fail_s_full": {"numerator": 49, "denominator": 90, "estimate": 0.5444444444444444, "ci95_low": 0.4418444949284148, "ci95_high": 0.6434055434341024}, "mean_max_es_future_finite": 4.349750040406831e+35, "n_finite": 86, "n_inf": 2, "n_nan": 2}, "31": {"n": 90, "p_fail_s_full": {"numerator": 50, "denominator": 90, "estimate": 0.5555555555555556, "ci95_low": 0.4527174112548652, "ci95_high": 0.6538451366982815}, "mean_max_es_future_finite": 5.281419908225666e+35, "n_finite": 87, "n_inf": 3, "n_nan": 0}}

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
