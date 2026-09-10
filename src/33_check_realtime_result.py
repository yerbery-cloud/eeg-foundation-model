from pathlib import Path
from collections import Counter
import csv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "results" / "loongbrain_realtime"

pred_files = sorted(RESULT_DIR.glob("predictions_*.csv"))
raw_files = sorted(RESULT_DIR.glob("raw_eeg_*.csv"))

if not pred_files:
    raise FileNotFoundError(f"没有找到 predictions_*.csv: {RESULT_DIR}")
if not raw_files:
    raise FileNotFoundError(f"没有找到 raw_eeg_*.csv: {RESULT_DIR}")

pred_path = pred_files[-1]
raw_path = raw_files[-1]

print("=" * 72)
print("Realtime Result Diagnostic")
print("=" * 72)
print("Predictions:", pred_path)
print("Raw EEG:", raw_path)

with open(pred_path, "r", encoding="utf-8-sig", newline="") as f:
    pred = list(csv.DictReader(f))

print("\nPrediction rows:", len(pred))

if pred:
    qualities = Counter(row.get("quality", "") for row in pred)
    print("Quality counts:", dict(qualities))

    valid = [r for r in pred if r.get("predicted_label", "") != ""]
    print("Valid predictions:", len(valid))

    max_uv = []
    for r in pred:
        try:
            v = float(r.get("max_abs_uv", ""))
            if np.isfinite(v):
                max_uv.append(v)
        except Exception:
            pass

    if max_uv:
        arr = np.asarray(max_uv, dtype=float)
        print(
            "Window max_abs_uv:",
            f"median={np.median(arr):.2f}, "
            f"p95={np.percentile(arr,95):.2f}, "
            f"max={np.max(arr):.2f}"
        )

with open(raw_path, "r", encoding="utf-8-sig", newline="") as f:
    raw = list(csv.DictReader(f))

print("\nRaw rows:", len(raw))

for col in ["fp1_raw", "fp2_raw", "fp1_volts", "fp2_volts"]:
    vals = []
    for r in raw:
        try:
            v = float(r[col])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass

    if vals:
        arr = np.asarray(vals, dtype=float)
        if "volts" in col:
            arr = arr * 1e6
            unit = "μV"
        else:
            unit = "raw"

        print(
            f"{col}: "
            f"median={np.median(arr):.3f} {unit}, "
            f"p95(abs)={np.percentile(np.abs(arr),95):.3f} {unit}, "
            f"max(abs)={np.max(np.abs(arr)):.3f} {unit}"
        )

print("\nInterpretation:")
if pred:
    qualities = Counter(row.get("quality", "") for row in pred)
    if qualities.get("Artifact", 0) == len(pred):
        print("- 所有窗口都被 Artifact QC 拒绝。")
        print("- 常见原因：真机原始 EEG 带较大 DC offset，而旧版程序在滤波前按绝对幅值判断。")
        print("- 建议换用 30_loongbrain_realtime_v2.py 后重新采集。")
    elif qualities.get("Flat", 0) == len(pred):
        print("- 所有窗口都被判为 Flat，优先检查电极接触、通道映射和 LSL 数据是否变化。")
    elif qualities.get("NaN/Inf", 0) == len(pred):
        print("- 所有窗口包含 NaN/Inf，优先检查 LSL 数据源。")
    else:
        print("- 请根据上面的 Quality counts 判断主要拒绝原因。")
else:
    print("- predictions CSV 没有记录，需要检查 REST/TASK 阶段是否收到足够连续样本。")
