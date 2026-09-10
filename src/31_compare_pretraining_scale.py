from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEW_PRETRAIN = PROJECT_ROOT / "results" / "pretrain60" / "test" / "test_summary.csv"
NEW_LINEAR = PROJECT_ROOT / "results" / "linear_probe60" / "linear_probe60_test_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "scaling_20_vs_60"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_OUT = OUTPUT_DIR / "scaling_summary.csv"
FIG_OUT = OUTPUT_DIR / "scaling_20_vs_60.png"

# Confirmed results from the completed 20-subject experiment/report.
OLD_PRETRAIN_MSE = 0.4336
OLD_LINEAR_F1 = 0.4826
OLD_LINEAR_ACC = 0.4828


def read_metric_csv(path):
    out = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:
        if len(row) >= 2:
            out[row[0].strip()] = row[1].strip()
    return out


def f(d, key):
    return float(d[key])


def main():
    if not NEW_PRETRAIN.exists() or not NEW_LINEAR.exists():
        raise FileNotFoundError(
            "Please finish 15_evaluate_pretrain60.py and 19_linear_probe_pretrain60.py first."
        )

    pre60 = read_metric_csv(NEW_PRETRAIN)
    lin60 = read_metric_csv(NEW_LINEAR)

    mse60 = f(pre60, "pretrained_mse")
    f160 = f(lin60, "test_macro_f1")
    acc60 = f(lin60, "test_accuracy")

    with open(SUMMARY_OUT, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["pretrain_subjects", "test_reconstruction_mse", "linear_probe_accuracy", "linear_probe_macro_f1"])
        writer.writerow([20, OLD_PRETRAIN_MSE, OLD_LINEAR_ACC, OLD_LINEAR_F1])
        writer.writerow([60, mse60, acc60, f160])

    x = np.arange(2)
    labels = ["20 subjects", "60 subjects"]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.3))
    ax1.plot(x, [OLD_PRETRAIN_MSE, mse60], marker="o", linewidth=2, label="Reconstruction MSE")
    ax1.set_ylabel("Test Reconstruction MSE")
    ax1.set_xticks(x, labels)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, [OLD_LINEAR_F1, f160], marker="s", linewidth=2, label="Linear Probe Macro F1")
    ax2.set_ylabel("EEGMAT Linear Probe Macro F1")
    ax2.set_ylim(0, 1)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    plt.title("Pretraining Scale: 20 vs 60 Subjects")
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=180)
    plt.close()

    print("=" * 72)
    print("Pretraining Scaling Experiment")
    print(f"20 subjects: Reconstruction MSE={OLD_PRETRAIN_MSE:.4f}, Linear F1={OLD_LINEAR_F1:.4f}")
    print(f"60 subjects: Reconstruction MSE={mse60:.4f}, Linear F1={f160:.4f}")
    print("MSE change:", f"{mse60 - OLD_PRETRAIN_MSE:+.4f}")
    print("Linear F1 change:", f"{f160 - OLD_LINEAR_F1:+.4f}")
    print("Saved:", SUMMARY_OUT)
    print("Figure:", FIG_OUT)


if __name__ == "__main__":
    main()
