import time
import numpy as np

from pylsl import (
    resolve_streams,
    StreamInlet
)


# ============================================================
# 1. 获取 Channel Metadata
# ============================================================

def get_channel_info(info):

    labels = []
    units = []

    try:

        channel = (
            info
            .desc()
            .child("channels")
            .child("channel")
        )

        for _ in range(
            info.channel_count()
        ):

            label = channel.child_value(
                "label"
            )

            unit = channel.child_value(
                "unit"
            )

            labels.append(
                label if label else ""
            )

            units.append(
                unit if unit else ""
            )

            channel = (
                channel.next_sibling()
            )

    except Exception:

        labels = [
            ""
            for _ in range(
                info.channel_count()
            )
        ]

        units = [
            ""
            for _ in range(
                info.channel_count()
            )
        ]

    return labels, units


# ============================================================
# 2. 搜索 LSL
# ============================================================

print("=" * 75)
print("LoongBrain LSL Stream Check")
print("=" * 75)

print(
    "\n正在搜索 LSL Stream..."
)

streams = resolve_streams(
    wait_time=5.0
)


if len(streams) == 0:

    raise RuntimeError(
        "\n没有发现任何 LSL Stream。\n"
        "请确认：\n"
        "1. LoongBrain 软件已经打开\n"
        "2. 设备已经连接\n"
        "3. LSL 广播模块已经启动"
    )


print(
    f"\n发现 {len(streams)} 个 Stream"
)


# ============================================================
# 3. 打印所有 Stream
# ============================================================

eeg_candidates = []


for index, info in enumerate(
    streams
):

    labels, units = (
        get_channel_info(
            info
        )
    )

    print("\n")
    print("-" * 75)

    print(
        f"Stream #{index}"
    )

    print(
        "Name:",
        info.name()
    )

    print(
        "Type:",
        info.type()
    )

    print(
        "Channel Count:",
        info.channel_count()
    )

    print(
        "Nominal Sampling Rate:",
        info.nominal_srate()
    )

    print(
        "Source ID:",
        info.source_id()
    )

    print(
        "Channel Labels:",
        labels
    )

    print(
        "Channel Units:",
        units
    )


    text = (

        info.name()
        + " "
        + info.type()

    ).lower()


    if (
        info.type().lower() == "eeg"
        or
        "eeg" in text
        or
        "loong" in text
    ):

        eeg_candidates.append(
            info
        )


# ============================================================
# 4. EEG Candidate
# ============================================================

print("\n")
print("=" * 75)
print("EEG Candidates")
print("=" * 75)


if len(eeg_candidates) == 0:

    raise RuntimeError(
        "发现了 LSL Stream，"
        "但没有找到明显的 EEG Stream。"
    )


for index, info in enumerate(
    eeg_candidates
):

    print(
        f"[{index}] "
        f"{info.name()} | "
        f"type={info.type()} | "
        f"channels={info.channel_count()} | "
        f"sfreq={info.nominal_srate()}"
    )


# ============================================================
# 5. 如果只有一个，测试收数据
# ============================================================

if len(eeg_candidates) != 1:

    print(
        "\n⚠️ 当前有多个 EEG Candidate。"
    )

    print(
        "请先记录上面的 Name，"
        "在 30 号程序中手动设置 TARGET_STREAM_NAME。"
    )

    raise SystemExit


info = eeg_candidates[0]


print(
    "\n自动选择：",
    info.name()
)


inlet = StreamInlet(
    info,
    max_buflen=5
)


print(
    "\n正在读取约 2 秒数据..."
)


samples, timestamps = inlet.pull_chunk(

    timeout=2.5,

    max_samples=max(
        int(
            info.nominal_srate()
            * 2
        ),
        100
    )
)


if len(samples) == 0:

    raise RuntimeError(
        "已经找到 EEG Stream，"
        "但没有收到样本。"
    )


data = np.asarray(
    samples,
    dtype=np.float64
)


timestamps = np.asarray(
    timestamps
)


print("\n")
print("=" * 75)
print("Received Data")
print("=" * 75)


print(
    "Data shape:",
    data.shape
)


print(
    "First sample:",
    data[0]
)


print(
    "Min:",
    np.min(
        data,
        axis=0
    )
)


print(
    "Max:",
    np.max(
        data,
        axis=0
    )
)


print(
    "Std:",
    np.std(
        data,
        axis=0
    )
)


# ============================================================
# 6. 用 timestamp 粗略检查真实采样率
# ============================================================

if len(timestamps) > 2:

    duration = (

        timestamps[-1]
        - timestamps[0]

    )

    if duration > 0:

        measured_sfreq = (

            (len(timestamps) - 1)
            / duration

        )

        print(
            "\nMeasured Sampling Rate:",
            f"{measured_sfreq:.2f} Hz"
        )


print("\n")
print("=" * 75)
print("LSL Stream Check 完成")
print("=" * 75)