# 跨平台实时视觉代码审计（2026-09-24）

范围：只下载、阅读源码和官方文档；未在 Jetson / Raspberry Pi / Luxonis 硬件运行，不把代码机制说成 RK3588 实测收益。主线是等待、丢帧、所有权和时间戳，非平台跑分对比。

## 1. dusty-nv/jetson-utils：借鉴 ReadLatestOnce，不照抄 zeroCopy 标签

- 审计提交 `fae4b4250f985ab92180b6ec955b79995b7a34ff`，MIT。
- [RingBuffer.inl L161-L207](https://github.com/dusty-nv/jetson-utils/blob/fae4b4250f985ab92180b6ec955b79995b7a34ff/cpp/threads/RingBuffer.inl#L161-L207)：`ReadLatest` 把读位置推进到最新写入位置，`ReadOnce` 防止重复消费。`gstBufferManager::Dequeue` 调用 `ReadLatestOnce`。这是真正减少已完成帧排队/年龄的机制，不需要处理所有中间帧。
- [gstBufferManager.cpp L237-L284](https://github.com/dusty-nv/jetson-utils/blob/fae4b4250f985ab92180b6ec955b79995b7a34ff/cpp/codec/gstBufferManager.cpp#L237-L284)：非 NVMM 路径仍 `memcpy(nextBuffer, gstData, gstSize)`；时间戳用 `GST_BUFFER_DTS_OR_PTS`，无有效时间戳时取接收侧 `apptime_nano`。不能把这个时间戳无条件当曝光时刻，DTS 也不能随意代替观测 PTS。
- [gstBufferManager.cpp L304-L452](https://github.com/dusty-nv/jetson-utils/blob/fae4b4250f985ab92180b6ec955b79995b7a34ff/cpp/codec/gstBufferManager.cpp#L304-L452)：NVMM 路径导出 fd→EGL→CUDA 后，仍将 CUDA array 用 `cudaMemcpy2DFromArrayAsync` 复制到线性 CUDA 内存，再 `cudaConvertColor`。减少 CPU 往返不等于没有数据搬运。`zeroCopy` 实际还控制 mapped CPU/GPU 分配；[RingBuffer.inl L78-L94](https://github.com/dusty-nv/jetson-utils/blob/fae4b4250f985ab92180b6ec955b79995b7a34ff/cpp/threads/RingBuffer.inl#L78-L94)。
- RK3588 迁移：用一个包含 frame_id、时间戳、图像句柄及生命周期的完整 FrameHandle 替换 latest slot，不把图像与时间戳放在两个独立无版本槽里。优先在做昂贵预处理前选择最新帧；单在推理后丢弃结果并不能退回已经消耗的 NPU 时间。共享映射只解决部分 CPU 复制；还应单独计 RGA 搬运与 fence 等待。
- 预期改善：过载下的输入等待和 frame age；不直接缩短 NPU 单帧服务时间。读源码技能：环形缓存、条件变量、RAII、GStreamer GstBuffer 生命周期、显式计量复制。

## 2. raspberrypi/picamera2：队列语义、请求生命周期、缓存一致性

- 审计提交 `73ce3b7a27e600abec704a843975f4925f029d0a`，BSD-2-Clause。
- [picamera2.py L1266-L1273](https://github.com/raspberrypi/picamera2/blob/73ce3b7a27e600abec704a843975f4925f029d0a/picamera2/picamera2.py#L1266-L1273) 决定保留最后一个 completed request，或完全不保留；[L1447-L1500](https://github.com/raspberrypi/picamera2/blob/73ce3b7a27e600abec704a843975f4925f029d0a/picamera2/picamera2.py#L1447-L1500) 回收其余请求。`queue=False` 是应用已完成帧缓存策略，不等于摄像头没有在途曝光/ISP 缓存，也不等于只有一个 DMA buffer。buffer_count=1 若长期保留请求会立即卡住采集，源码对此明确规避。
- [request.py L106-L152](https://github.com/raspberrypi/picamera2/blob/73ce3b7a27e600abec704a843975f4925f029d0a/picamera2/request.py#L106-L152)：`acquire/release` 引用计数，最后引用释放才 reuse/queue_request。零复制视图必须受请求的生命周期保护。`MappedArray` 是共享视图 [L53-L80](https://github.com/raspberrypi/picamera2/blob/73ce3b7a27e600abec704a843975f4925f029d0a/picamera2/request.py#L53-L80)；普通 `make_array()` 为将独立图像交给调用者仍会复制 [L168-L183](https://github.com/raspberrypi/picamera2/blob/73ce3b7a27e600abec704a843975f4925f029d0a/picamera2/request.py#L168-L183)。
- [dmaallocator.py L110-L142](https://github.com/raspberrypi/picamera2/blob/73ce3b7a27e600abec704a843975f4925f029d0a/picamera2/allocators/dmaallocator.py#L110-L142)：CPU 访问映射内存有 DMA_BUF_IOCTL_SYNC START/END。它处理 CPU 缓存一致性；**不是**互斥锁、所有权转移或硬件任务完成信号。[Linux DMA-BUF 官方语义](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html#dma-buffer-ioctls) 明确区分 cache coherency 和设备完成同步。
- [picamera2.py L1573-L1585](https://github.com/raspberrypi/picamera2/blob/73ce3b7a27e600abec704a843975f4925f029d0a/picamera2/picamera2.py#L1573-L1585) 和 [L1876-L1885](https://github.com/raspberrypi/picamera2/blob/73ce3b7a27e600abec704a843975f4925f029d0a/picamera2/picamera2.py#L1876-L1885)：`flush=True` 另按时间戳过滤，代码用 SensorTimestamp−ExposureTime 与 monotonic 比较。**不要将这条平台适配公式原样移到 RK**；要依据 RK 驱动 timestamp flags、SOE/EOF 和时钟域核实。libcamera 控件定义与某平台实测也不能混同。
- RK3588 迁移：V4L2 dequeue 后形成 RAII lease；RGA 完成读取原图后尽快 QBUF，NPU只保留小分辨率独立输入 buffer。预先分配的物理 buffer 池数量不等于逻辑待处理队列深度：前者保证 DMA 有可用内存，后者应 latest=1。异步 RGA 时持有 lease 到 fence 真正完成；只 close fd 或设置 mutex 都不能提前归还。
- 预期改善：避免 buffer 被显示/编码/推理长期占用产生相机饥饿、复制抖动和取旧图；不能保证曝光和 ISP 本身变快。技能：V4L2 QBUF/DQBUF、引用计数/移动语义、stride/plane、dma-buf 与 fence、时钟域。

## 3. luxonis/depthai-core：必须管所有层队列，结果绑定真实输入

- 为稳定接口审计 v2.30.0 固定提交 `e0f6b52d048ef7ceac2ecd44deb2e161dbb97699`，MIT；不是声称 v2 参数等同新 v3。完整 clone 在下载中较大，停止 clone 后从相同固定提交按文件下载至 `build/research/depthai-core-audit`，未执行构建或外部脚本。
- [LockingQueue.hpp L105-L130](https://github.com/luxonis/depthai-core/blob/e0f6b52d048ef7ceac2ecd44deb2e161dbb97699/include/depthai/utility/LockingQueue.hpp#L105-L130)：non-blocking 模式满时 pop 最旧消息后 push 新消息；blocking 模式条件变量等待空位。`maxSize=1, blocking=false` 是保新丢旧，而不是所有 non-blocking API 都自动保新。
- [DataQueue.hpp L41](https://github.com/luxonis/depthai-core/blob/e0f6b52d048ef7ceac2ecd44deb2e161dbb97699/include/depthai/device/DataQueue.hpp#L41) 主机输出队列默认 size16 blocking；[NeuralNetwork.hpp L30-L46](https://github.com/luxonis/depthai-core/blob/e0f6b52d048ef7ceac2ecd44deb2e161dbb97699/include/depthai/pipeline/node/NeuralNetwork.hpp#L30-L46) 设备 NN 输入默认 size5 blocking。只把主机输出改 latest，并不会消除设备侧 NN 输入积压。
- [DataQueue.cpp L30-L102](https://github.com/luxonis/depthai-core/blob/e0f6b52d048ef7ceac2ecd44deb2e161dbb97699/src/device/DataQueue.cpp#L30-L102)：独立接收线程先 read/parse→queue.push→执行用户 callbacks；callbacks 仍在同一接收线程且持 callback 锁。慢 callback 或 blocking 的满输出队列都能停止接收，进而造成反压。异步 API 名称不意味着耗时工作已被隔离。
- `NeuralNetwork::passthrough` 传递实际参与推理的输入，专门用于 non-blocking 丢帧场景。RK 的对应机制是让 DetectionResult 携带同一个 FrameHandle 元数据及输入坐标变换；禁止把刚完成的旧检测画在随手取到的最新图像上。
- [ImgFrame.hpp L43-L55](https://github.com/luxonis/depthai-core/blob/e0f6b52d048ef7ceac2ecd44deb2e161dbb97699/include/depthai/pipeline/datatype/ImgFrame.hpp#L43-L55) 区分 host 对齐时间与 device 未同步时间；[ImgFrame.cpp L20-L44](https://github.com/luxonis/depthai-core/blob/e0f6b52d048ef7ceac2ecd44deb2e161dbb97699/src/pipeline/datatype/ImgFrame.cpp#L20-L44) 明确 exposure START/MIDDLE/END 偏移。这是应借鉴的时间戳接口，不是让 RK 使用其具体修正值。
- RK3588 迁移：列出 sensor/ISP、V4L2、GStreamer、应用 latest、RGA、RKNN、结果发布所有缓冲点；在 app 回调只移交句柄和 metadata。检测结果 ready 即写小型 latest-result，GUI/JSON/网络由独立消费者消费。保留 frame_id 和 high-water mark，过期完成结果不能覆盖新结果。效果主要在排队、frame age、输出抖动，非 NPU 内核耗时。

## 4. NVIDIA DeepStream reference apps：批处理可能主动等帧，测量探针也能阻塞

- 示例审计提交 `9946965e8adb1aa93b1b66983ec4196351c9190c`，仓库 Apache-2.0；DeepStream SDK 的闭源插件和分发条款另算。本条证据是**样例 C/C++ 源码 + 官方插件文档**，不是审过 nvstreammux 内部实现。
- [custom_tiler_cfg.c L238-L245](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/blob/9946965e8adb1aa93b1b66983ec4196351c9190c/deepstream-custom-tile-config/custom_tiler_cfg.c#L238-L245)：多源示例配置 batch-size=num_sources、batched-push-timeout=33000 微秒。[官方 nvstreammux 文档](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvstreammux.html) 明确：满 batch 或计时超时才推送；返回下游 buffer 后原输入才能归还。**33 ms 是最大形成批等待配置，不能说每帧固定增加 33 ms**；一个输入且 batch1 时不必等凑批。
- [deepstream_parallel_infer_app.cpp L128-L155](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/blob/9946965e8adb1aa93b1b66983ec4196351c9190c/deepstream_parallel_inference_app/tritonclient/sample/apps/deepstream-parallel-infer/deepstream_parallel_infer_app.cpp#L128-L155)：pad probe 使用 nvds_measure_buffer_latency 并关联 source_id/frame_num，这是正确的逐帧测量思路。但例程在 probe 中加锁、calloc、g_print，不能直接当低干扰生产级 profiler；本 RK 项目应将固定大小 timing record 放有界日志队列，由后台聚合。
- RK3588 迁移：单相机实时路径 batch1；并发只在可证实降低 P95/P99/AoI 时启用，不能为了占满 NPU 增加排队。不要等待按提交顺序排列的 oldest future；保留实际完成时间和源时间。将结果发布挂在后处理完成点，与画框/视频编码消费隔离。
- 预期改善：取消凑 batch、头阻塞、sink同步和重日志造成的尾延迟；批处理本身主要用于吞吐/利用率，不应套到无人机单观测链。

## 可直接归入主方案的机制及验证

| 机制来源 | RK 实现 | 要测的结果 | 技能 |
|---|---|---|---|
| Jetson ReadLatestOnce + DepthAI overwrite-oldest | 采集线程排空已就绪旧 buffer，原子替换最新完整帧描述；最多一个推理在途作为起点 | input age at inference、主动丢帧数、P95/P99/AoI、有效更新间隔 | C++ 并发与 bounded queue |
| Picamera2 Request 引用 + DMA sync | FrameLease 自定义归还；原图在 RGA 完成后归还；小图池与采集池分开 | camera pool starvation、RGA fence wait、hold_time、copy bytes/frame | RAII、V4L2、DMA-BUF、cache/fence |
| DepthAI passthrough/timestamps | FrameMeta 全链路携带；曝光语义和时钟域显式记录；结果拒绝倒序 | frame/result 配对正确性、source-to-ready 与 host-to-ready 分开 | 时间戳、metadata、数据流契约 |
| DeepStream latency probe/batch lesson | batch1；采样轻量事件；慢输出消费者隔离 | t_pre、t_submit、t_run、t_post、ready→publish、队列P99 | perf/ftrace、GStreamer、观测工具 |

核心边界：目前 INT8 inputs_set 约 0.29 ms 已很小，不建议为省这一段立即引入 RKNN 共享 buffer 的所有权复杂度。先让 RGA 输出独立小 RGB buffer，再 ordinary inputs_set；只有 buffer 拷贝/转换在实测中重新成为瓶颈，才验证 native tensor stride/type/quant、import 和 fence 能否共享。以上都是设计推断，没有本轮 RK3588 新实测改善数字。
