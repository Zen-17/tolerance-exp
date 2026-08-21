# Experiment A report

## S-layer recommendation

- strict scaled (rtol, atol) = (1e-06, 1e-07)
- balanced scaled (rtol, atol) = (1e-05, 1e-06)
- recommended raw rtol = 1e-05
- recommended raw atol = 1.1313708498984761e-05
- result_rtol = 1e-05
- result_atol = 1.1313708498984761e-05

strict harmful-pass = 0.0000 [0.0000, 0.0741]
balanced benign-reject = 1.0000 [0.9740, 1.0000]

Trials used for selection: 12 (calibration/smoke only).

## Flash vs reference consistency

{
  "flash_seconds": 0.7570961080491543,
  "ref_seconds": 1.1833747886121273,
  "first_divergence": null,
  "top1": {
    "numerator": 0,
    "denominator": 16,
    "estimate": 0.0,
    "ci95_low": 0.0,
    "ci95_high": 0.19361341827271994,
    "n_compared": 16
  },
  "ppl_flash": 1.1866366050249373,
  "ppl_ref": 1.1805674055337834,
  "logits": {
    "max_abs": 0.625,
    "rel_l2": 0.018434430288708326,
    "has_nan_inf": false
  },
  "n_flash_tokens": 16,
  "n_ref_tokens": 16
}

## Limits

Parameter choice used the smoke split only. This experiment does not study K-block size, BER, or ABFT. Short smoke runs can make the 0.1% top-1 budget overly strict (a single token change exceeds 0.1%). Full phase-1 needs more tokens.
