# Experiment A report

## S-layer recommendation

- strict scaled (rtol, atol) = (1e-06, 1e-07)
- balanced scaled (rtol, atol) = (1e-06, 1e-05)
- recommended raw rtol = 1e-06
- recommended raw atol = 0.00011313708498984762
- result_rtol = 1e-06
- result_atol = 0.00011313708498984762

strict harmful-pass = 0.0092 [0.0066, 0.0128]
balanced benign-reject = 0.6878 [0.6802, 0.6953]

Trials used for selection: 18144 (calibration/smoke only).

## Flash vs reference consistency

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

Parameter choice used the calibration split only. This experiment does not study K-block size, BER, or ABFT. Tolerances were not chosen on the test split.
