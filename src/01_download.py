from pathlib import Path

from mne.datasets import eegbci


# =========================
# 1. 设置项目路径
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "eegmmidb"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 设置下载对象
# =========================

subjects = [1]

runs = [1, 2]


# =========================
# 3. 下载数据
# =========================

files = eegbci.load_data(
    subjects=subjects,
    runs=runs,
    path=DATA_DIR,
    update_path=False
)


# =========================
# 4. 输出下载结果
# =========================

print("\n下载完成！")

print(f"一共得到 {len(files)} 个 EDF 文件：")

for file in files:
    print(file)