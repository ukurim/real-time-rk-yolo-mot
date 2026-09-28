# 两个指定仓库的源码审计（2026-09-24）

只克隆/读代码/重算已有 CSV，没有执行仓库构建或推理脚本。固定提交如下。

## OrangeKat/ZeroCopyInferenceEngine

- 提交：`abaef529e3797da0da61a80a53c553281673512f`（2026-08-22）。
- 实际对象：CUDA/TensorRT 实验，不是 RK3588/Jetson 摄像头部署工程；默认网络输入 224×224，原图 synthetic 512×288 RGB；自定义四个卷积 + AdaptiveAvgPool + 分类/回归两个 Linear，每张图只有一个四维回归向量及 80 分类 logits，不是 YOLO 的密集检测/NMS。
- 固定源码：[模型](https://github.com/OrangeKat/ZeroCopyInferenceEngine/blob/abaef529e3797da0da61a80a53c553281673512f/script/backbone/backbone.py#L22-L62)、[尺寸](https://github.com/OrangeKat/ZeroCopyInferenceEngine/blob/abaef529e3797da0da61a80a53c553281673512f/src/pipeline.h#L11-L24)、[PRNG 输入及测量循环](https://github.com/OrangeKat/ZeroCopyInferenceEngine/blob/abaef529e3797da0da61a80a53c553281673512f/apps/pipeline_main.cpp#L89-L135)。

真实机制：

1. 构造阶段创建 TensorRT context、两个 non-blocking CUDA stream、events、所有大 buffer，一次绑定 tensor 地址；每帧应用代码没有 cudaMalloc。可借鉴固定池、稳定地址及 RAII，但没有工具证据保证 TensorRT 内部绝无分配。[初始化](https://github.com/OrangeKat/ZeroCopyInferenceEngine/blob/abaef529e3797da0da61a80a53c553281673512f/src/pipeline.cpp#L87-L132)
2. `FusedPreprocessKernel` 单 kernel 实现 bilinear resize、/255、mean/std 和 HWC→CHW，减少中间图像及 CPU/GPU阶段切换。RK3588 可借鉴“一次遍历/少中间面”，使用 RGA 合并 resize/CSC，RKNN 按模型合同处理量化；CUDA代码不能直接运行，RGA 也不是通用浮点 normalize/CHW kernel。[kernel](https://github.com/OrangeKat/ZeroCopyInferenceEngine/blob/abaef529e3797da0da61a80a53c553281673512f/src/fused_kernels.cu#L30-L88)
3. stream events 表达 copy→pre/infer/post→copy 依赖，CPU只在取结果时等待完成；避免每段 host 同步。注意测量循环每次 submit 后立即 wait，是单个在途 batch；两个 stream 不等于两帧重叠。[submit/wait](https://github.com/OrangeKat/ZeroCopyInferenceEngine/blob/abaef529e3797da0da61a80a53c553281673512f/src/pipeline.cpp#L135-L198)
4. README“zero staging copies”不能按字面迁移：当前代码 line143 逐帧 H2D，167–172 三次 D2H。`pinned_input_dev_` 虽通过 cudaHostGetDevicePointer 取得，但预处理实际读 `d_raw_`；mapped pointer没有进入执行链。[拷贝证据同上](https://github.com/OrangeKat/ZeroCopyInferenceEngine/blob/abaef529e3797da0da61a80a53c553281673512f/src/pipeline.cpp#L112-L172)。其 benchmarks 文档自己也明确列出 H2D/D2H，故应客观称“预分配、显式异步拷贝和GPU融合”，而非全链路零拷贝。[methodology](https://github.com/OrangeKat/ZeroCopyInferenceEngine/blob/abaef529e3797da0da61a80a53c553281673512f/docs/benchmarks.md#L7-L18)
5. CUDA warp-shuffle Top-K只对每张图80类排序并返回同一个回归框，不是 YOLO 的8400候选解码/NMS，不能直接当成我们后处理方案。[TopK](https://github.com/OrangeKat/ZeroCopyInferenceEngine/blob/abaef529e3797da0da61a80a53c553281673512f/src/fused_kernels.cu#L92-L180)

测量边界：CPU e2e = submit 前到 wait_results 完成，包含这段拷贝，不包含相机曝光、驱动缓存或文件读取。`gpu_latency_ms` 的 start event 排在 H2D 后，反而不含 H2D；profile 的 h2d/pre/trt 则是 CPU submit时间，不是设备执行时长。0.276ms P50/0.441ms P99 属于 RTX 2000 Ada 上的小网络 synthetic input，不能与 YOLO640 RK3588 对标。可借鉴 warmed多样本、P99/P99.9、明确事件边界；batch扩大主要提高吞吐，等满batch会增加实时帧等待。仓库固定tree未见LICENSE/COPYING/NOTICE，适合学习思想，直接移植代码前应明确许可。

## SuyashMullick/real-time-vision-latency-bench

- 提交：`7d5a3a52d2f250b2ff729954a34ddf76530c051f`（2026-03-02），MIT。
- [许可](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/LICENSE)。

真实机制与局限：

1. `std::vector<FrameData>`8槽池，四线程capture/preprocess/inference/tracker，线程间仅传`FrameData*`。SPSC头尾64字节分离，release/acquire可避免部分锁竞争与false sharing。可借鉴所有权和trace，但实际 cv::Mat 只默认构造，不在启动阶段预分配像素；blob/输出/后处理vector仍可能分配。[池/线程](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/src/pipeline/pipeline.cpp#L46-L75)、[SPSC](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/src/pipeline/messaging/spsc_ring.h#L10-L53)
2. 每段FIFO，capture无空闲槽时仅yield、不继续read/清空相机；下游满时循环push/yield。没有“替换旧帧”、deadline或按时间戳抑制过期结果。对视频文件是尽快解码，读取的FPS仅打印、不按PTS节拍播放。--buffers3可以缩短应用排队，但不能消除驱动/后端积压。[pipeline](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/src/pipeline/pipeline.cpp#L104-L181)、[capture](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/src/pipeline/stages/capture.cpp#L12-L43)
3. “zero-copy”只成立于应用线程间传指针，不是摄像头→CUDA。OpenCV read→CPU blobFromImage（RGB/resize/normalize/CHW）→ORT CreateCpu tensor→Session.Run CUDA→CPU遍历输出/NMS，没有DMA-BUF或ORT GPU I/O binding。[preprocess](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/src/pipeline/stages/preprocess.cpp#L6-L21)、[infer](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/src/pipeline/stages/inference_onnx_gpu.cpp#L62-L151)
4. steady_clock逐段trace值得学习；实际`latency_total`起点是`cap.read`完成，排除read本身、驱动缓存、曝光；终点t_command_published又在cout前，不是消费者收取完成。每帧同步cout与stats mutex/动态增长在tracker线程，仍可影响下一帧及队列。[统计边界](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/src/pipeline/timing/stats.cpp#L40-L62)、[输出](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/src/pipeline/stages/command_out.cpp#L46-L60)
5. `--pin`/`--realtime`固定线程core0..3（realtime也会设affinity）、FIFO80优先级及busy-yield。RK3588大/小核拓扑和CPU推理后处理负载不同，不能原样套；先profile绑核/IRQ/频率/thermal，再只提升必要线程。仓库仅说明这种机制，不能据此宣称硬实时保证。[线程参数](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/src/pipeline/pipeline.cpp#L23-L42)

重算仓库自带CSV（未重新跑其GPU程序）：全表146帧、总墙钟3.195664s；按它的`scripts/summarize.py`跳过最初3s，仅剩12帧。其64.04FPS、P50=81.247ms、P99=85.35349ms来自这个很短样本，不能支撑长时稳定性。逐帧直接相减：

| 项目 | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|
| capture_end→publish |81.247|85.15945|85.35349|
| preprocess_end→infer_start等待 |65.743|71.5271|71.57022|
| 三段等待合计 |65.751|71.534|71.578|

这比单看README更能说明“高FPS/lock-free≠低延迟”：约66ms在推理前排队，推理指标P50只有9.398ms且含CPU NMS。各分位不能简单相加/相减；上表合计是先逐帧求和再取分位。

[原始CSV](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/results/raw/uav_gpu_eval.csv)、[汇总脚本](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/scripts/summarize.py#L24-L44)、[发布的summary](https://github.com/SuyashMullick/real-time-vision-latency-bench/blob/7d5a3a52d2f250b2ff729954a34ddf76530c051f/results/summary/uav_gpu_eval_summary.md)。meta的commit写UNKNOWN_COMMIT，输入video，没有CLI参数，复现实验条件不完整。

建议迁移：FrameTrace / 每段队列等待、预分配池、严格生命周期、CPU affinity对照；改成持续采集最新帧mailbox而不是复制其FIFO图。默认单RKNN context、单在途检测，然后以source frame age P95/P99而非吞吐验证是否要加一个并行stage/context。对缓冲有限的设备，帧持有释放和同步完成比“无锁”标签优先；简单短锁mailbox可能足够，应按trace证明再增加无锁复杂度。
