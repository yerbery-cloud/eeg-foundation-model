from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
DATA_ROOT = PROCESSED_ROOT / "eegmmidb_fp1_fp2_60"
SUBJECT_DIR = DATA_ROOT / "by_subject"

NEW_SUBJECTS = list(range(21, 61))
TRAIN_SUBJECTS = np.arange(1, 49, dtype=np.int32)
VAL_SUBJECTS = np.arange(49, 55, dtype=np.int32)
TEST_SUBJECTS = np.arange(55, 61, dtype=np.int32)

WINDOWS_PATH = DATA_ROOT / "pretrain_60subjects_windows.npy"
METADATA_PATH = DATA_ROOT / "pretrain_60subjects_metadata.npy"
SPLIT_PATH = DATA_ROOT / "pretrain_60subjects_split.npz"
QC_PATH = DATA_ROOT / "pretrain_60subjects_qc_summary_new40.npz"


def locate_old_20_dataset():
    """Locate the already-generated merged 20-subject processed arrays."""
    exact_window_names = [
        "pretrain_20subjects_windows.npy",
        "pretrain20_windows.npy",
    ]

    window_candidates = []
    for name in exact_window_names:
        window_candidates.extend(PROCESSED_ROOT.rglob(name))

    # Fallback: any processed file clearly named as 20-subject windows.
    if not window_candidates:
        window_candidates = [
            p for p in PROCESSED_ROOT.rglob("*20*subject*window*.npy")
            if "60" not in p.name.lower()
        ]

    if not window_candidates:
        raise FileNotFoundError(
            "Cannot find the original 20-subject merged windows file under "
            f"{PROCESSED_ROOT}. Expected something like pretrain_20subjects_windows.npy."
        )

    # Prefer files outside the new _60 directory.
    window_candidates = sorted(
        window_candidates,
        key=lambda p: ("eegmmidb_fp1_fp2_60" in str(p), len(str(p)))
    )

    for w_path in window_candidates:
        parent = w_path.parent
        metadata_candidates = [
            parent / "pretrain_20subjects_metadata.npy",
            parent / "pretrain20_metadata.npy",
            parent / "metadata.npy",
            Path(str(w_path).replace("windows.npy", "metadata.npy")),
        ]
        for m_path in metadata_candidates:
            if m_path.exists():
                return w_path, m_path

    raise FileNotFoundError(
        "Found a 20-subject windows file, but could not locate its matching metadata.npy.\n"
        + "Candidates:\n"
        + "\n".join(str(p) for p in window_candidates[:10])
    )


def main():
    print("=" * 76)
    print("Merge EEGMMIDB 20 old subjects + 40 new subjects = 60 subjects")
    print("=" * 76)

    # --------------------------------------------------------
    # 1) Reuse the old S001-S020 merged processed dataset.
    # --------------------------------------------------------
    old_w_path, old_m_path = locate_old_20_dataset()
    print("\nReusing old 20-subject processed dataset:")
    print("Windows :", old_w_path)
    print("Metadata:", old_m_path)

    X20 = np.load(old_w_path).astype(np.float32, copy=False)
    M20 = np.load(old_m_path).astype(np.int32, copy=False)

    if X20.ndim != 3 or X20.shape[1:] != (2, 640):
        raise RuntimeError(f"Old 20-subject windows shape invalid: {X20.shape}")
    if len(X20) != len(M20):
        raise RuntimeError("Old 20-subject windows/metadata count mismatch")

    old_subjects = sorted(np.unique(M20[:, 0]).astype(int).tolist())
    expected_old = list(range(1, 21))
    if old_subjects != expected_old:
        raise RuntimeError(
            f"Old dataset subjects are {old_subjects}, expected {expected_old}."
        )

    print(f"Old S001-S020: {len(X20)} windows")

    # --------------------------------------------------------
    # 2) Load newly processed S021-S060.
    # --------------------------------------------------------
    windows_list = [X20]
    metadata_list = [M20]
    qc_rows = []

    for subject in NEW_SUBJECTS:
        w_path = SUBJECT_DIR / f"S{subject:03d}_windows.npy"
        m_path = SUBJECT_DIR / f"S{subject:03d}_metadata.npy"
        q_path = SUBJECT_DIR / f"S{subject:03d}_qc.json"

        if not w_path.exists() or not m_path.exists():
            raise FileNotFoundError(
                f"Missing S{subject:03d}. Run 12_preprocess_60subjects.py first."
            )

        w = np.load(w_path).astype(np.float32, copy=False)
        m = np.load(m_path).astype(np.int32, copy=False)

        if w.ndim != 3 or w.shape[1:] != (2, 640):
            raise RuntimeError(f"S{subject:03d} windows shape invalid: {w.shape}")
        if len(w) != len(m):
            raise RuntimeError(f"S{subject:03d}: windows/metadata count mismatch")
        if not np.all(m[:, 0] == subject):
            raise RuntimeError(f"S{subject:03d}: metadata subject id mismatch")

        windows_list.append(w)
        metadata_list.append(m)

        if q_path.exists():
            with open(q_path, "r", encoding="utf-8") as f:
                qc_rows.append(json.load(f))

        print(f"S{subject:03d}: {len(w):5d} windows")

    # --------------------------------------------------------
    # 3) Merge 20 + 40.
    # --------------------------------------------------------
    X = np.concatenate(windows_list, axis=0)
    metadata = np.concatenate(metadata_list, axis=0)

    final_subjects = sorted(np.unique(metadata[:, 0]).astype(int).tolist())
    if final_subjects != list(range(1, 61)):
        raise RuntimeError(
            f"Final subject list invalid: {final_subjects}. Expected S001-S060."
        )

    train_mask = np.isin(metadata[:, 0], TRAIN_SUBJECTS)
    val_mask = np.isin(metadata[:, 0], VAL_SUBJECTS)
    test_mask = np.isin(metadata[:, 0], TEST_SUBJECTS)

    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)
    test_indices = np.flatnonzero(test_mask)

    if len(train_indices) + len(val_indices) + len(test_indices) != len(X):
        raise RuntimeError("Split does not cover all windows")

    np.save(WINDOWS_PATH, X)
    np.save(METADATA_PATH, metadata)
    np.savez(
        SPLIT_PATH,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_subjects=TRAIN_SUBJECTS,
        val_subjects=VAL_SUBJECTS,
        test_subjects=TEST_SUBJECTS,
    )

    # QC here only summarizes newly processed S021-S060 because the old 20 subjects
    # already have their own historical QC record in the original experiment.
    if qc_rows:
        candidate = sum(int(q.get("candidate", 0)) for q in qc_rows)
        nan_inf = sum(int(q.get("nan_inf_rejected", 0)) for q in qc_rows)
        flat = sum(int(q.get("flat_rejected", 0)) for q in qc_rows)
        amplitude = sum(int(q.get("amplitude_rejected", 0)) for q in qc_rows)
        kept = sum(int(q.get("kept", 0)) for q in qc_rows)
        np.savez(
            QC_PATH,
            subjects=np.asarray(NEW_SUBJECTS, dtype=np.int32),
            candidate=candidate,
            nan_inf_rejected=nan_inf,
            flat_rejected=flat,
            amplitude_rejected=amplitude,
            kept=kept,
            rejection_rate=(candidate - kept) / max(candidate, 1),
        )

    print("\n===== Final Dataset =====")
    print("X shape:", X.shape)
    print("metadata shape:", metadata.shape)
    print("Subjects:", final_subjects[0], "...", final_subjects[-1])
    print("Train subjects:", TRAIN_SUBJECTS.tolist())
    print("Validation subjects:", VAL_SUBJECTS.tolist())
    print("Test subjects:", TEST_SUBJECTS.tolist())
    print("Train windows:", len(train_indices))
    print("Validation windows:", len(val_indices))
    print("Test windows:", len(test_indices))
    print("Total windows:", len(X))
    print("\nSaved:")
    print(WINDOWS_PATH)
    print(METADATA_PATH)
    print(SPLIT_PATH)
    print("\nNext: python src/14_pretrain_60subjects.py")


if __name__ == "__main__":
    main()
