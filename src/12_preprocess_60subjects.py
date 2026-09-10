from pathlib import Path
import json
import time
import numpy as np
import mne
from mne.datasets import eegbci

# ============================================================
# FAST 60-subject preprocessing
# Reuse the already completed S001-S020 processed dataset.
# This script ONLY downloads/processes S021-S060.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "eegmmidb_fp1_fp2_60"
SUBJECT_DIR = OUTPUT_ROOT / "by_subject"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SUBJECT_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT: S001-S020 are reused later by script 13 from the old 20-subject merged dataset.
SUBJECTS = list(range(21, 61))
RUNS = list(range(1, 15))
CHANNELS = ["Fp1", "Fp2"]
LOW_FREQ = 1.0
HIGH_FREQ = 45.0
WINDOW_SECONDS = 4.0
STRIDE_SECONDS = 2.0
EXPECTED_SFREQ = 160.0
MAX_AMPLITUDE_UV = 500.0
FLAT_STD_V = 1e-8


def preprocess_subject(subject: int):
    subject_windows = []
    subject_metadata = []

    qc = {
        "subject": subject,
        "candidate": 0,
        "nan_inf_rejected": 0,
        "flat_rejected": 0,
        "amplitude_rejected": 0,
        "kept": 0,
        "failed_runs": [],
    }

    for run in RUNS:
        try:
            # MNE will reuse any existing EDF cache automatically.
            paths = eegbci.load_data(subject, [run], update_path=True, verbose=False)
            if len(paths) != 1:
                raise RuntimeError(
                    f"Subject {subject} Run {run}: expected 1 EDF, got {len(paths)}"
                )

            raw = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
            eegbci.standardize(raw)
            raw.pick(CHANNELS)

            sfreq = float(raw.info["sfreq"])
            if abs(sfreq - EXPECTED_SFREQ) > 1e-6:
                raw.resample(EXPECTED_SFREQ, npad="auto", verbose=False)
                sfreq = float(raw.info["sfreq"])

            raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
            data = raw.get_data().astype(np.float64)

            window_samples = int(round(WINDOW_SECONDS * sfreq))
            stride_samples = int(round(STRIDE_SECONDS * sfreq))

            local_index = 0
            for start in range(0, data.shape[1] - window_samples + 1, stride_samples):
                stop = start + window_samples
                window = data[:, start:stop]
                qc["candidate"] += 1

                if not np.isfinite(window).all():
                    qc["nan_inf_rejected"] += 1
                    continue

                channel_std = window.std(axis=1)
                if np.any(channel_std < FLAT_STD_V):
                    qc["flat_rejected"] += 1
                    continue

                max_uv = float(np.max(np.abs(window)) * 1e6)
                if max_uv > MAX_AMPLITUDE_UV:
                    qc["amplitude_rejected"] += 1
                    continue

                mean = window.mean(axis=1, keepdims=True)
                std = window.std(axis=1, keepdims=True)
                window = (window - mean) / (std + 1e-8)
                window = window.astype(np.float32)

                subject_windows.append(window)
                subject_metadata.append([subject, run, local_index])
                qc["kept"] += 1
                local_index += 1

            print(
                f"S{subject:03d} R{run:02d} done | current kept={qc['kept']}"
            )

        except Exception as exc:
            qc["failed_runs"].append({"run": run, "error": str(exc)})
            print(f"[WARN] S{subject:03d} R{run:02d} failed: {exc}")

    if len(subject_windows) == 0:
        raise RuntimeError(f"Subject {subject} produced zero valid windows.")

    windows = np.stack(subject_windows).astype(np.float32)
    metadata = np.asarray(subject_metadata, dtype=np.int32)

    rejection = qc["candidate"] - qc["kept"]
    qc["rejection_rate"] = rejection / max(qc["candidate"], 1)

    # Do not mark a subject complete if any run failed.
    if qc["failed_runs"]:
        failed_qc = SUBJECT_DIR / f"S{subject:03d}_qc_failed.json"
        with open(failed_qc, "w", encoding="utf-8") as f:
            json.dump(qc, f, ensure_ascii=False, indent=2)
        print(
            f"[INCOMPLETE] S{subject:03d}: {len(qc['failed_runs'])} run(s) failed. "
            "No final .npy files were saved; rerun script 12 after network recovers."
        )
        return qc

    np.save(SUBJECT_DIR / f"S{subject:03d}_windows.npy", windows)
    np.save(SUBJECT_DIR / f"S{subject:03d}_metadata.npy", metadata)
    with open(SUBJECT_DIR / f"S{subject:03d}_qc.json", "w", encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)

    failed_qc = SUBJECT_DIR / f"S{subject:03d}_qc_failed.json"
    if failed_qc.exists():
        failed_qc.unlink()

    print(
        f"✓ S{subject:03d}: {windows.shape} | "
        f"reject={100 * qc['rejection_rate']:.2f}% | all 14 runs complete"
    )
    return qc


def main():
    print("=" * 76)
    print("EEGMMIDB 60-subject preprocessing — FAST REUSE MODE")
    print("S001-S020: reuse old processed 20-subject dataset")
    print("S021-S060: download/process in this script")
    print("Output:", OUTPUT_ROOT)
    print("=" * 76)

    all_qc = []
    start_time = time.time()

    for subject in SUBJECTS:
        w_path = SUBJECT_DIR / f"S{subject:03d}_windows.npy"
        m_path = SUBJECT_DIR / f"S{subject:03d}_metadata.npy"
        q_path = SUBJECT_DIR / f"S{subject:03d}_qc.json"

        if w_path.exists() and m_path.exists() and q_path.exists():
            print(f"[SKIP] S{subject:03d} already exists")
            with open(q_path, "r", encoding="utf-8") as f:
                all_qc.append(json.load(f))
            continue

        qc = preprocess_subject(subject)
        all_qc.append(qc)

    with open(
        OUTPUT_ROOT / "preprocess_qc_subjects21_60.json", "w", encoding="utf-8"
    ) as f:
        json.dump(all_qc, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 76)
    print("S021-S060 preprocessing finished")
    print(f"Elapsed: {elapsed / 60:.1f} min")
    print("Next: python src/13_merge_and_split_60subjects.py")
    print("=" * 76)


if __name__ == "__main__":
    main()
