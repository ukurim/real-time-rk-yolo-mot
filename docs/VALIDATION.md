# RK3588 第一版验证记录

记录日期：2026-09-24。代码、模型和测试均在当前开发板实际运行；不是由 README 的 FPS 换算出的延迟。

本文保留首轮结果和当时的 AUTO/CV1 配置选择。后续已完成 FP16 输入优化、INT8 对照及 CV4/mask7 复测，当前配置和结果见 [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md)。[RKNN_PROFILING.md](RKNN_PROFILING.md)记录 API 计时和三核分工；不能把 `inference` 整体解释为纯 NPU 计算，也不能把 mask=7 解释成所有卷积都分到三核。

## 环境与边界

- RK3588 / aarch64，Linux 5.10.198，Ubuntu 20.04.6，OpenCV 4.2。
- RKNN-Toolkit2 与仓库 runtime 均为 2.3.2，运行时报告 NPU driver 0.9.8。
- 读取到 CPU governor 为 `performance`，NPU 为 `rknpu_ondemand`，当时频率 1 GHz。本轮没有执行旧 `performance.sh` 或修改 governor/频率。
- 官方 Rockchip YOLOv8n COCO80，640×640，非量化 FP16，split9；RKNN SHA256 为 `0d17bff40beb4ef0705d91dc11fceae70bb1c23d7e20af742a4e117479bf190c`。来源、转换及元数据见 [模型说明](../model/YOLOV8N.md)和 [manifest](../model/RK3588/yolov8n.manifest.json)。
- 输入为仓库 `1080p60hz.mov`，H.264、1920×1080、60 fps，80,294,885 字节；SHA256 为 `9e60c1ee13e3804cf16ca8ab88eecc8f89b52367889b8b9556f09ea2e18113d3`。
- BoT-SORT 默认 sparseOptFlow GMC、关闭 ReID；阈值 0.5/0.1/0.6，nominal_fps=60，按实际 PTS 推进跟踪。显示/编码/推流关闭，JSON 写本地文件；OpenCV 内部线程为 1。
- **没有相机和真实标定**。实际视频均显式使用 `--no-los`，LOS 为 null；以下延迟到检测/跟踪结果就绪，不冒充真实相机曝光到有效 LOS 的验证。LOS 数学与合成标定集成由独立测试覆盖。

## 功能验证

`./build-linux_RK3588.sh` 完成构建、测试和安装。六项 CTest 全部通过（2.59 秒）：检测解码/量化边界/letterbox、BoT-SORT 数值与关联、LOS 畸变和符号、队列顺序/取消、GStreamer PTS/所有权及合成跟踪→LOS、JSON 慢消费者/FIFO 无读端。

安装产物使用真实 RKNN 和真实视频，三 detector 离线保留全部帧：

```bash
./install/Aerial_detection_demo_Linux/Aerial_detection_demo \
  --config install/Aerial_detection_demo_Linux/config/video.yaml \
  --input 1080p60hz.mov --no-los --offline --workers 3 --core-mask 0 --input-mode uint8 --cv-threads 1 \
  --max-frames 120 --output build/installed_smoke.jsonl
python3 tests/validate_jsonl.py build/installed_smoke.jsonl --expected-frames 120
```

得到 **120/120 帧、955 条目标记录，其中 69 条明确标记为纯预测**；输入/完成/过期/JSON 丢弃均为 0。ID、时间、bbox、预测标志、null LOS 和数值范围校验通过。运行约 5.23 秒；这种保全帧的离线背压模式不是低延迟基准。独立 `--output -` 三帧测试也通过，RKNN 的 stdout 诊断没有污染 JSONL。

所有下表对应 JSONL 均经 `tests/validate_jsonl.py` 验证，结果 frame_id/图像时间严格递增。原始运行日志和 JSONL 留在本地 `build/benchmarks_*`；可纳入版本控制的完整汇总及丢弃统计见 [latency_results.json](validation/latency_results.json)。

## 输入瓶颈与修正

初始实时试跑不足以比较并发：单线程 `videoconvert` 使输入落后于媒体时钟，300 帧仅产生 7 个结果、286 帧过期；该结果没有作为有效并发基准。

单独处理相同视频前 300 帧、非实时 fakesink 的一次测量：软件解码不转 BGR 为 0.825 秒；增加普通 `videoconvert` 为 9.570 秒；`videoconvert n-threads=4` 为 3.278 秒。因此修复了实际的颜色转换瓶颈，不能把全部耗时归因于视频解码。

MPP 的 `format=BGR` 路径实际可用，但普通用户访问 DMA heap 被拒绝，供应商插件探测失败后崩溃；本轮通过 sudo 加独立 registry 验证，未修改系统设备权限。MPP 直接映射供 CPU 读取比普通内存慢；增加仅针对已选中帧的 `--copy-input` 后，完整链路延迟下降。[MPP 取证记录](MPP_NOTES.md)保留权限和访问成本证据。

## 实时回放并发比较

每项从同一文件起点按 60 fps 播放，接收 300 帧；只保留最新待处理帧。统计剔除前 10 条输出记录，P50/P95/P99 直接对每帧时间差排序计算。不同配置有意丢弃的帧不同，输出数量必须与延迟一起看；P99 小样本只是观测值，不是最坏时间保证。

下表单位 ms；“PTS P95”是 GStreamer 回放时钟映射到单调时钟后的源时间估计，不是传感器曝光时间。

| 输入路径 | detector / NPU 配置 | 输出帧/300 | 接收→结果 P50 | P95 | P99 | PTS P95 |
|---|---|---:|---:|---:|---:|---:|
| auto + 四线程颜色转换 | 1 / AUTO | 91 | 111.72 | 122.14 | 125.55 | 122.90 |
| auto + 四线程颜色转换 | 1 / 三核 mask=7 | 93 | 110.19 | 121.21 | 125.15 | 121.60 |
| auto + 四线程颜色转换 | 2 / 各一核 | 104 | 130.23 | 151.97 | 162.97 | 152.11 |
| auto + 四线程颜色转换 | 3 / 各一核 | 99 | 137.47 | 155.09 | 161.92 | 215.27 |
| MPP BGR 直接映射 | 1 / AUTO | 65 | 149.58 | 160.92 | 165.63 | 161.06 |
| MPP BGR 直接映射 | 1 / 三核 mask=7 | 66 | 147.56 | 152.82 | 155.81 | 152.91 |
| MPP BGR 直接映射 | 2 / 各一核 | 75 | 179.21 | 204.65 | 211.35 | 204.75 |
| MPP BGR 直接映射 | 3 / 各一核 | 71 | 193.30 | 213.68 | 219.20 | 213.81 |
| MPP BGR + 待处理帧复制 | 1 / AUTO | 88 | 109.56 | 117.67 | 119.06 | 117.84 |
| MPP BGR + 待处理帧复制 | 1 / 三核 mask=7 | 90 | 109.54 | 117.42 | 120.44 | 117.56 |
| MPP BGR + 待处理帧复制 | 2 / 各一核 | 109 | 128.86 | 153.37 | 160.12 | 153.57 |
| MPP BGR + 待处理帧复制 | 3 / 各一核 | 101 | 133.07 | 154.73 | 160.99 | 154.86 |

随后各接收 900 帧（约 15 秒）复核单实例候选，仍剔除前 10 条输出：

| 输入路径 | NPU | 输出帧/900 | 接收→结果 P50/P95/P99 | PTS P95 |
|---|---|---:|---|---:|
| auto + 四线程颜色转换 | AUTO | 271 | 110.45 / 122.94 / 126.59 | 123.18 |
| auto + 四线程颜色转换 | mask=7 | 278 | 110.15 / 121.32 / 130.74 | 121.45 |
| MPP BGR + 待处理帧复制 | AUTO | 267 | 108.77 / 116.64 / 117.88 | 116.78 |
| MPP BGR + 待处理帧复制 | mask=7 | 270 | 109.18 / 116.65 / 117.92 | 116.84 |

选择 **1 detector / AUTO**：多实例略增输出量却明显增加时延；mask=7 没有稳定的延迟收益。文件配置保留可直接运行的 auto 输入；具备 MPP 权限时可显式选择 `--decoder mpp --copy-input`。相机配置预置 MPP + copy，但实际相机仍需重测。此结论只针对本机、此模型和输入，不推广为“单线程永远更好”。

900 帧 MPP-copy/AUTO 的分阶段 P50/P95：输入排队 9.01/16.31、预处理（含复制）8.95/10.22、RKNN 调用 42.94/43.83、后处理 3.96/4.32、跟踪 43.89/46.45；结果就绪到发布开始等待 0.138/0.157。不同分位数不能直接相加。RKNN 阶段含 inputs_set、run、outputs_get，并非纯 NPU kernel 时间；“跟踪”含 GMC，也不是只计算关联的耗时。

复现上述比较：

```bash
python3 scripts/benchmark_latency.py --frames 300 --input-mode uint8 --cv-threads 1 --output-dir build/recheck_auto
sudo env GST_REGISTRY=/tmp/aerial-mpp-root-full.bin python3 scripts/benchmark_latency.py \
  --frames 300 --decoder mpp --input-mode uint8 --cv-threads 1 --output-dir build/recheck_mpp_raw
sudo env GST_REGISTRY=/tmp/aerial-mpp-root-full.bin python3 scripts/benchmark_latency.py \
  --frames 300 --decoder mpp --copy-input --input-mode uint8 --cv-threads 1 --output-dir build/recheck_mpp_copy
# 长复测添加 --frames 900 --cases one_auto one_all_cores，并使用新的 output-dir。
```

## 尚需真实设备完成的验证

相机曝光/驱动队列时间、实际镜头标定与 LOS 误差、场景检测/跟踪精度、长时间温控以及实际编码/网络消费者均未被这次文件实验覆盖。不要用 `receiver_to_result` 隐去采集前缓存；不要将 `source_to_result_estimate` 改名为实测曝光延迟。

下一步应先接入相机并校验时间戳和标定；继续降低耗时时，重点测 GMC、FP16 推理及帧复制。RGA 生成网络输入/GMC 缩略图、真实场景 INT8 量化都可比较，但须保留坐标正确性、跟踪质量和精度验证；不为零拷贝标签增加未经验证的内存共享链路。
