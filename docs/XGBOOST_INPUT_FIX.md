# XGBoost input validation fix

The modeling boundary now converts only XGBoost-invalid feature values (`+/-inf`
and finite values outside the float32 range) to `NaN`, which XGBoost treats as
missing. It emits a RuntimeWarning with counts so these cases remain visible.
The same conversion is applied to validation and inference inputs.

The feature builder also explicitly treats a zero previous uncertainty value as
missing before computing step ratios.

Notebook 04 now excludes infinite metric values from descriptive means/std.
An infinite competition score can legitimately occur when F2 is zero; this is
separate from the XGBoost feature-input error.

**Important:** this ZIP contains no source dataset (the `data/` directory only
contains `.gitkeep`), so a full end-to-end retraining run could not be performed
here. After extracting, run the notebook from the beginning in the same
environment where your dataset is available. Watch for the
`XGBoost input sanitization` warning and investigate the named feature pipeline
if the counts are nonzero.
