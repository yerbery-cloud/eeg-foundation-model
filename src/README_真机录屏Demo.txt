LoongBrain 真机录屏 Demo（校准版）操作说明

目的
----
这套流程用于“同一受试者、同一佩戴 session 下短时校准后的实时演示”。
它不是 independent held-out test。汇报时请明确说经过 subject-specific calibration。

运行前
------
1. 不要删除原有 checkpoints/lora/best.pt。
2. LoongBrain 官方软件连接设备，并确认 Fp1/Fp2 波形与阻抗正常。
3. 开启 LSL。
4. 建议先运行：
   python src\29_lsl_stream_check.py
   确认属于自己的 EEG stream。
5. 进入环境后安装依赖：
   python -m pip install -r requirements.txt

推荐录屏流程
------------
1. 佩戴完成后先安静坐 1 分钟，确认波形稳定。
2. 运行：
   python src\37_loongbrain_calibrated_recording_demo.py
3. 程序先自动做 30 s INITIAL_SETTLE。
4. 接着完成 3 组 Calibration：
   REST 20 s -> TASK 20 s
   REST 20 s -> TASK 20 s
   REST 20 s -> TASK 20 s
   中间有 5 s 状态切换提示。
5. Calibration 期间：
   - REST：闭眼、放松、停止心算。
   - TASK：闭眼持续做 4387 - 47 - 47 - ...
   - 不说话、不数手指、不皱眉、不咬牙、不移动头部。
6. Calibration 结束后程序会显示：
   Calibration CV Balanced Accuracy
   - >= 0.60：建议开始录屏。
   - < 0.60：先检查电极接触、额头/眼动和受试状态，再重新校准。
     不要放宽 Artifact 阈值“刷分”。
7. 程序出现 READY FOR RECORDING 后：
   - 不要摘头带
   - 不要移动电极
   - 不要重启 LoongBrain / LSL
   - 开启屏幕录制
   - 再按 Enter
8. Demo 为：
   10 s settle
   25 s REST
   5 s prepare
   25 s TASK
9. 终端实时显示：
   True=...
   DemoPred=...
   P(Task)=...
   Foundation=...
   Latency=...
10. 结束后会输出 Demo Accuracy / Balanced Accuracy / Macro F1 / Latency，
    并保存到 results/recording_demo/。

重要实验口径
------------
录屏结果应称为：
“subject-specific calibrated real-time demo”
或
“同一受试者短时校准后的实时演示”

不要把该 Demo Accuracy 写成独立泛化测试准确率。
之前的 42.1% independent test 仍然是独立 session 的真实结果。
