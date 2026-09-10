# Results

This folder contains **small, de-identified summary files** for the main experiments.

Current public summaries:

```text
demo_summary.csv
linear_probe60_summary.csv
mask_ratio_robustness_summary.csv
pretrain20_summary.csv
pretrain60_summary.csv
realtime_simulation_summary.csv
scaling_summary.csv
```

## File descriptions

- `pretrain20_summary.csv` — 20-subject pretraining evaluation
- `pretrain60_summary.csv` — 60-subject pretraining evaluation
- `scaling_summary.csv` — 20 vs. 60 subject scaling comparison
- `linear_probe60_summary.csv` — frozen Linear Probe using the 60-subject encoder
- `mask_ratio_robustness_summary.csv` — 25%, 50%, and 75% mask-ratio results
- `realtime_simulation_summary.csv` — offline real-time pipeline simulation
- `demo_summary.csv` — final same-session calibrated LoongBrain demo summary

## Privacy note

Do **not** publish raw EEG, subject-specific calibration traces, personal prediction logs, identifiable participant information, or local absolute paths.

The final calibrated demo achieved **80.56% Accuracy**, **80.96% Balanced Accuracy**, and **80.54% Macro F1** with **5.21 ms mean pipeline latency**. This is a **same-session subject-specific calibrated demo**, not an independent cross-session generalization score.
