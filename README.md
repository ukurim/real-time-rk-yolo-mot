# RK3588 YOLOv8n → BoT-SORT → LOS

第一版 C++ 感知链路：GStreamer 相机/视频 → YOLOv8 RKNN → BoT-SORT → 相机坐标系 LOS。
优化目标是图像到对应结果的延迟。显示、编码和推流是独立的可选消费者，默认关闭。

当前实现使用普通 RKNN 输入 API、CPU RGB letterbox，并保留解码后的 GstSample 映射作为只读图像。MPP 可直接输出 BGR；CPU 读取该映射较慢时，可在 worker 选中待处理帧后复制一次至普通内存。**没有宣称端到端零拷贝**。旧 `Yolo11.cc`、`preprocess.cc`、`rknnPool.hpp` 等保留作历史参考，不参与新程序构建。旧/新调用链、目录及线程结构见 [代码架构](docs/ARCHITECTURE.md)。

## 构建与运行

依赖：C++17、CMake、OpenCV 4（core/imgproc/calib3d/video/videoio/highgui）、GStreamer app/video 开发库、与模型匹配的 RKNN runtime/驱动。

```bash
sudo apt-get install cmake libopencv-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-libav gstreamer1.0-plugins-bad
./build-linux_RK3588.sh
```

构建脚本只编译、测试、安装，不自动运行视频或修改硬件频率。可用 `-DBUILD_RKNN_APP=OFF` 仅构建不依赖 NPU 的测试。

模型使用 Rockchip 官方兼容 YOLOv8n 导出，输入 **640×640**，输出三尺度 DFL/类别分支（6 或 9 个张量）。默认 FP16 配置显式把 RGB/255 转成半精度再提交，已验证输出一致性；INT8 候选使用普通 UINT8 输入和模型内归一化。模型来源、转换步骤与输出约束见 [YOLOV8_MODEL.md](docs/YOLOV8_MODEL.md)。不可直接用旧 YOLO11 文件冒充 YOLOv8n。

没有标定时，显式运行检测/跟踪模式：

```bash
./build/Aerial_detection_demo --config config/video.yaml --no-los --max-frames 120
# 或 ./demo.sh
```

读取仓库 `1080p60hz.mov`，输出 `results.jsonl`。此模式 `calibrated=false`、`los=null`，不会生成假角度。

实时回放使用最新帧模式：

```bash
./build/Aerial_detection_demo --config config/video.yaml --no-los --realtime --max-frames 900
# 本机 MPP 路径需要 DMA heap 访问权限；本轮用独立 registry 和 sudo 验证：
sudo env GST_REGISTRY=/tmp/aerial-mpp-root-full.bin ./build/Aerial_detection_demo \
 --config config/video.yaml --no-los --realtime --decoder mpp --copy-input --max-frames 900
```

INT8 模型、量化数据来源与独立配置已提供：

```bash
./build/Aerial_detection_demo --config config/video_int8.yaml --no-los --max-frames 900
```

900 帧 MPP 实测中，原 FP16 的源 PTS→结果 P95 为 116.95 ms，优化 FP16 为 **96.83 ms**，INT8 为 **71.79 ms**；INT8 的 `rknn_run` P50 为 **14.96 ms**。这些是无真实标定的视频回放结果，不是相机曝光→LOS 测量。INT8 与 FP16 检测输出有差异，因此仍保留 FP16 默认。参数、阶段耗时和检测对照见 [本轮优化与验证](docs/OPTIMIZATION_RESULTS.md)。

本机普通用户因 DMA heap 权限不足，扫描 MPP 插件会触发供应商库崩溃；本轮未修改系统设备权限。部署权限和实际硬件路径见 [MPP 排查记录](docs/MPP_NOTES.md)。`auto` 路径使用四线程 CPU 颜色转换；`mpp` 路径使用插件的 `format=BGR`，省去 CPU `videoconvert`。`source.copy_to_cpu` / `--copy-input` 只复制实际选中的帧，其耗时计入 `preprocess`。

提供真实标定后：

```bash
./build/Aerial_detection_demo --config config/video.yaml --calibration /path/to/camera.yaml
./build/Aerial_detection_demo --config config/camera.yaml --calibration /path/to/camera.yaml
```

配置中的路径相对配置文件目录；命令行路径相对当前目录。安装目录没有附带演示视频，安装后请用 `--input /path/to/video.mov`。`--help` 列出覆盖参数。`config/camera.example.yaml` 是被程序拒绝的占位模板，必须填写实际标定值；不能仅改 `calibrated` 开关。

## 检测、跟踪和 LOS

- **检测**：保比例 letterbox，记录实际整数缩放尺寸与 padding，框以浮点精度还原到原始畸变图像。动态读取类别数，验证张量数量/格式/量化类型；不兼容模型显式报错。默认保留分数 ≥0.1 的候选，供跟踪第二阶段使用。
- **跟踪**：固定上游提交的 BoT-SORT C++ 移植，XYWH 八维 Kalman、两阶段关联、score fusion、全局分配、未确认/丢失/重激活/重复清理；默认 sparseOptFlow GMC，ReID 关闭。配置及适配、上游许可证/源码快照见 [BOTSORT_PROVENANCE.md](docs/BOTSORT_PROVENANCE.md)。每路视频只有一个按图像时间顺序更新的跟踪器。
- **LOS**：跟踪框中心 → 去畸变 → 内参反投影 → 单位射线。相机轴 **+X 向右、+Y 向下、+Z 向前**；方位右正、俯仰上正，单位弧度。`azimuth=atan2(x,z)`，`elevation=atan2(-y,hypot(x,z))`。支持 pinhole/fisheye；标定分辨率默认必须匹配。详见 [LOS.md](docs/LOS.md)。不估距离，不融合 IMU，不实现控制闭环。

每行 JSON 对应一帧，包括：

```json
{
  "frame_id": 42,
  "image_timestamp_ns": 700000000,
  "image_timestamp_domain": "stream_pts",
  "received_monotonic_ns": 1234000000000,
  "result_ready_monotonic_ns": 1234015000000,
  "image_size": [1920, 1080],
  "calibrated": true,
  "capture_latency_measured": false,
  "latency_ms": {"receiver_to_result": 15, "source_to_result_estimate": null, "queue": 1, "preprocess": 1, "inference": 10, "postprocess": 1, "tracking": 2, "los": 0},
  "targets": [{"track_id": 7, "class_id": 0, "score": 0.9, "bbox_xyxy": [800, 400, 1000, 700], "detection_updated": true, "observation": "detection_update", "seconds_since_update": 0, "los": {"unit_vector_camera": [0, 0, 1], "azimuth_rad": 0, "elevation_rad": 0}}]
}
```

以上仅为字段示意，不是该框/相机的真实测量。`bbox_xyxy` 为原图浮点坐标。纯预测目标标记 `detection_updated=false`、`observation=prediction`，默认最多输出丢失后 0.2 秒；其角度来自预测框，不能当作当前帧新观测。框中心出画面或畸变逆解失败时 `los=null`。每帧无目标时仍输出空列表。

## 时间、缓存与并发

- 输入 appsink 仅保留 1 帧；实时模式 `drop=true`，采集到推理之间也只有 1 个最新帧槽。每个 detector 最多 1 帧在途，默认 1 实例，可配置 1–3。
- 实时结果按递增 `frame_id` 接收，较旧的迟到结果丢弃，取消原来“积累三帧后输出”和等待最旧 future 的策略。跟踪使用真实 PTS 时间间隔，跳帧不伪装为固定帧率。
- 文件 `realtime: 0` 保留全部帧并有界重排序；这是功能验证模式，背压可能增加延迟。`--realtime` 按视频时间播放并丢旧帧，适合并发/新鲜度比较。
- 实时模式在推理前和跟踪前检查 `max_age_ms`，超过预算的结果丢弃；这不能保证最坏执行时间。每帧序号标识应用接收帧；appsink 丢帧可由 PTS 间隔观察，序号不是传感器原生帧号。
- 实时输入无 PTS 时从首帧起明确使用接收单调时间，不能声称测到了采集延迟；离线输入无 PTS 则拒绝运行，避免把处理时间当作媒体时间。有 PTS 后拒绝缺失/回退时间，不混合时钟域。
- `receiver_to_result` 从 appsink 取到帧后计时，**不包含此前采集/解码缓存**。实时 `source_to_result_estimate` 使用 PTS、segment 和 GStreamer 时钟映射；它也不是经过验证的传感器曝光时间。离线文件该值为 null，不能由文件回放得出相机真实端到端延迟。
- JSON 发布线程独立运行；实时模式慢消费者会跳过待发布旧记录。可选预览/视频输出同样只有 1 个待处理结果。统计会报告这些丢弃。结果就绪时间在序列化和 I/O 前记录；`publish_started_monotonic_ns` 另记录开始写出时间，两者均不等于外部消费者完整接收时间。
- `--preview` 开启窗口；`output.video_pipeline` 可填 OpenCV VideoWriter GStreamer 管线，例如 `appsrc ! videoconvert ! mpph264enc ! h264parse ! rtspclientsink location=...`。编码阻塞局限于消费者线程；不可用时告警。第三方编码器卡死仍可能影响退出时的线程回收。

并发选择需要测量，不以 FPS 推导：

```bash
python3 scripts/benchmark_latency.py --frames 300
# 测本机已验证的 MPP + 普通内存路径：
sudo env GST_REGISTRY=/tmp/aerial-mpp-root-full.bin python3 scripts/benchmark_latency.py \
  --frames 300 --decoder mpp --copy-input --output-dir build/benchmarks_mpp_copy
# 有标定时添加 --calibration /path/to/camera.yaml
```

分别测试 1/2/3 个实例，另比较单实例三核，保留每帧 JSON 和日志，并汇总 P50/P95/P99、分阶段耗时、输出帧数。`core_mask=0` 时，多实例分别使用 NPU core 0/1/2，单实例使用 AUTO。当前随仓库配置选择 **1 实例 / mask7 / OpenCV 四线程**，可用 `--workers`、`--core-mask` 和 `--cv-threads` 覆盖。测过 OpenCV 1/2/4/8 线程及多实例后选择此组合，不能推广为“单线程通用最优”。完整比较见 [本轮优化](docs/OPTIMIZATION_RESULTS.md)，[首轮记录](docs/VALIDATION.md)保留历史数据。

AUTO 是自动选核，不等于三核协同；`--workers 3 --core-mask 0` 是不同帧各一核，`--workers 1 --core-mask 7` 才请求单帧三核。算子检查发现，两份编译图在 mask=7 时主要 Swish 融合卷积仍由 Core0 执行。本轮消除了大部分 FP16 输入转换开销，并提供了 INT8 模型；没有假设三核能把延迟除以三。JSON 分项 `input_submit`、`rknn_run`、`output_get` 属于 `inference`；`tracking_gmc`、`tracking_association` 属于 `tracking`，不可重复累加。`completion_queue` 记录检测完成后等待跟踪的耗时。官方基准口径及早期取证见 [RKNN 性能定位](docs/RKNN_PROFILING.md)。

在 FP16 配置上通过 `--model` 换成 INT8 时，还须加 `--input-mode uint8`；更简单的方法是直接使用 `config/video_int8.yaml`。快速 FP16 输入只适用于已核对 mean=0/std=255 的模型，详见 [输入契约](docs/INPUT_PATH.md)。

## 测试与边界

```bash
cmake -E chdir build ctest --output-on-failure
```

八项测试涵盖量化/浮点 DFL、letterbox 逆变换、FP16 输入契约、检测对照匹配器、BoT-SORT 关联/时序、LOS 畸变与符号、最新帧/有界重排序、GStreamer PTS/缓冲所有权、慢消费者，以及合成标定下检测结果→跟踪→LOS 集成。它们不能代替真实镜头标定、实际相机时间戳验证和检测/跟踪准确率评估。

默认 CPU 预处理是可验证的起点。MPP 的 BGR 转换已走插件内部硬件通路；应用尚未直接实现 RGA letterbox、RKNN 内存绑定或解码 DMA 直通。旧实现中“DMA 虚拟地址传给 inputs_set”不能证明零拷贝。后续应优先围绕实测的 GMC 和 FP16 推理耗时优化，并同时验证跟踪质量与检测精度。
