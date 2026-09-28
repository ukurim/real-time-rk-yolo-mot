# RK3588 参考仓库源码审计（2026-09-24）

只 clone、读代码；没有执行参考仓库程序，也没有把 README 的性能当作本机测量。

## wzxzhuxi/rknn-3588-npu-yolo-accelerate-saturated

审计 commit `73c9a1bbc33193a152c749a3a24fa00627ea3be9`。当前版本已重构为 `src/{decoder,preprocess,engine,inference,postprocess}`，不能用旧 leafqycc 的 future 队列描述它。

- [main.cpp:69–89、92–129、144](https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate-saturated/blob/73c9a1bbc33193a152c749a3a24fa00627ea3be9/src/main.cpp#L69)：整段 raw H264/H265 文件读内存，MPP `fps_limit=1000` 基本关闭节流；默认 12 workers；解码回调同步 RGA 到独立 Mat，然后提交。FPS 计数只数按顺序取回的 detection 数量/时间，未测相机采集时间、frame age 或 P95/P99。
- [worker_pool.cpp:18–35、46–113](https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate-saturated/blob/73c9a1bbc33193a152c749a3a24fa00627ea3be9/src/inference/worker_pool.cpp#L18)：每 worker 一 context，`i%3` 分绑单 NPU 核；FIFO task 队列；`tasks.size() <= 10` 后 push，因此单生产者实际可有 11 个等待项，另有 12 个 worker 在途；满队列 sleep 1ms，阻塞解码回调。结果存 map；GetResult 指定 ID 轮询 sleep 1ms，最多约 5s。因此仍有按帧号等待产生的队头阻塞，不是 latest-only。
- [yolov5_detector.cpp:136–155](https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate-saturated/blob/73c9a1bbc33193a152c749a3a24fa00627ea3be9/src/inference/yolov5_detector.cpp#L136)：作者自己的注释把 NPU 观察耗时写成约 67ms = 16.6ms × 每核 4 worker 排队。这个注释不是我们的测量，但清楚说明 175FPS 不等于单帧约 5.7ms。
- [rga_nv12_to_rgb.cpp:15–35](https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate-saturated/blob/73c9a1bbc33193a152c749a3a24fa00627ea3be9/src/preprocess/rga_nv12_to_rgb.cpp#L15)：RGA 一步 NV12→RGB+resize，输入和输出均 `wrapbuffer_virtualaddr`，输出 cv::Mat 640×640。主回调忽略传来的 FD。这里是拉伸到正方形，不是 letterbox；后处理分别用宽高 scale 还原。迁移到我们项目应保留 letterbox/坐标契约，而不能直接搬拉伸。
- [rknn_engine.cpp:87–129](https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate-saturated/blob/73c9a1bbc33193a152c749a3a24fa00627ea3be9/src/engine/rknn_engine.cpp#L87)：一次 `rknn_create_mem`+`rknn_set_io_mem` 绑定输入输出；[detector:15–42、149–151](https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate-saturated/blob/73c9a1bbc33193a152c749a3a24fa00627ea3be9/src/inference/yolov5_detector.cpp#L15) 仍有 RGB Mat→RKNN mem 的 CPU NEON XOR+copy。准确表述是减少 RKNN API 隐含拷贝/转换，不是相机到NPU全链无拷贝。
- XOR 0x80 的成立条件是该 INT8 模型 mean=0/std=255/zp=-128/scale≈1/255，数学上 q=p−128。这里 hardcode INT8/NHWC/pass_through=1；查询的是 logical INPUT_ATTR/OUTPUT_ATTR，不应作为任意模型 native layout 的安全模板。新实现必须核实 native dims/fmt/w_stride/size_with_stride/scale/zp 与 RGB 排列匹配。
- [mpp_decoder.cpp:176–186、221–222、251–256](https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate-saturated/blob/73c9a1bbc33193a152c749a3a24fa00627ea3be9/src/decoder/mpp_decoder.cpp#L176)：PTS/DTS 只打印，不随回调 FrameItem 往下传；MPP buffer group 上限 24 是解码 buffer 资源限制，不能直接等同于 24 帧算法积压。
- [LICENSE.txt](https://github.com/wzxzhuxi/rknn-3588-npu-yolo-accelerate-saturated/blob/73c9a1bbc33193a152c749a3a24fa00627ea3be9/LICENSE.txt)：Apache-2.0（第三方另查）。

迁移价值：MPP NV12+RGA 融合颜色转换/缩放；预分配RKNN I/O；量化输出直接后处理；阶段计时。不要迁移默认12ctx、FIFO深队列、按ID等结果、无时间戳裸码流吞吐benchmark作为无人机低延迟架构。更多context主要是 throughput；减拷贝/转换可能同时减少单帧service time，须做A/B。

## yuunnn-w/rknn-cpp-yolo

审计 commit `470b663456ca437ab6cc256bbc04c3b4e1797c7a`。

- [rknn.cpp:15–23、44–70](https://github.com/yuunnn-w/rknn-cpp-yolo/blob/470b663456ca437ab6cc256bbc04c3b4e1797c7a/rknn.cpp#L15)：读取 bus.jpg 一次，BGR→RGB 在计时外，ctx_index=0；100次串行 run_inference 后算平均FPS。没有相机、视频队列、真实采集时间，也没有P95/P99。因此25FPS不能解释为相机全链延迟40ms。
- [rknn_model.cpp:160–185、270–307](https://github.com/yuunnn-w/rknn-cpp-yolo/blob/470b663456ca437ab6cc256bbc04c3b4e1797c7a/src/rknn_model.cpp#L160)：构造3contexts、分绑3个核、每context预分配输入输出、native tensor查询+set_io_mem。样例却只调用ctx0；没有实际推理线程池。
- [rknn_model.cpp:425–490](https://github.com/yuunnn-w/rknn-cpp-yolo/blob/470b663456ca437ab6cc256bbc04c3b4e1797c7a/src/rknn_model.cpp#L425)：CPU裁剪中心正方形并16对齐，RGA预处理写 input_mem->virt_addr，rknn_run 后每个output每帧 malloc，NC1HWC2→NCHW转排或memcpy，NMS后free。不是所有阶段都零拷贝。
- [rga_utils.cpp:44–70、136–198](https://github.com/yuunnn-w/rknn-cpp-yolo/blob/470b663456ca437ab6cc256bbc04c3b4e1797c7a/src/rga_utils.cpp#L44)：裁剪通过非const Mat引用修改输入，非连续时clone；RGA virtualaddr，尺寸相同用memcpy。非正方形letterbox分支用偏移指针作为dst，却没显式传完整目标行stride；样例先裁正方形规避了这条路径。不能直接迁移至全视场无人机观测/LOS。
- 输入 native attr type 改 UINT8；注释明确 normalize/quantize 可融合进NPU，因此 `set_io_mem` 并不证明运行时转换消失。初始化还开启 `RKNN_FLAG_COLLECT_PERF_MASK`，正式尾延迟测量应关掉性能收集后做分离profiling。
- **许可证矛盾**：[README](https://github.com/yuunnn-w/rknn-cpp-yolo/blob/470b663456ca437ab6cc256bbc04c3b4e1797c7a/README.md) 称 MIT，但 [LICENSE](https://github.com/yuunnn-w/rknn-cpp-yolo/blob/470b663456ca437ab6cc256bbc04c3b4e1797c7a/LICENSE#L1) 实际是 GPLv3 全文。报告写明矛盾；不要默认MIT直接复制实现。

迁移价值：预分配RKNN内存、RGA直接写目标buffer的结构、查询native输出layout。应改进全视场letterbox和坐标还原、buffer复用、直接布局后处理/一次预分配转排，保持每一步正确性对照。其3contexts不能当作已实现3路并行或低延迟证据。

## 自主补充：airockchip/rknn_model_zoo（优先于博客/社区Demo）

审计 commit `bad6c7334531becaf90a561988519b7bec34d0ab`，Apache-2.0 示例文件；模型/第三方license另外核实。

- [yolov8_zero_copy.cc:68–115](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yolov8/cpp/rknpu2/yolov8_zero_copy.cc#L68)：native input/output attr query；按 `size_with_stride` 分配；绑定内存。input改 UINT8 的注释明确会把归一化/量化融合到NPU，默认INT8则外部负责。这两种路径必须分别计时，不能把uint8 RGB直接当signed INT8 passthrough。
- [252–325](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yolov8/cpp/rknpu2/yolov8_zero_copy.cc#L252)：RGA letterbox直接拿RKNN input mem的FD/virt_addr；但输出仍每帧malloc，NC1HWC2→NCHW或memcpy。该示例在此版本写明仅量化路径，FP16分支拒绝；不应泛化为SDK不支持任何FP16绑定方式。
- [utils/image_utils.c:583–627、631–666](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/utils/image_utils.c#L583)：支持源/目标FD导入，颜色填充后RGA improcess，把颜色转换与缩放组合。当前helper每次调用import/release，生产版可在固定buffer池生命周期中缓存RGA handle，节约每帧映射/句柄开销；同时必须显式保存真实图像宽高与分配stride，不把所有buffer假设tight packed。

推荐实现次序：先只替换CPU预处理为RGA并验证逐帧RGB与检测框误差；再A/B uint8绑定native vs正确的signed INT8 passthrough；最后在确有收益时优化native输出读取。观测分项必须包含RGA等待、归一化/量化、run、输出转换，而不是只看inputs_set消失。

## 自主补充：airockchip/librga

审计 commit `2b32edcb97b601b25683e2941d888c8515da6d55`，以下样例Apache-2.0。

- [allocator_dma_cache_demo.cpp:66–105、123–128](https://github.com/airockchip/librga/blob/2b32edcb97b601b25683e2941d888c8515da6d55/samples/allocator_demo/src/rga_allocator_dma_cache_demo.cpp#L66)：DMA heap分配、FD导入、复用RGA handle，CPU写完→device同步；设备完成→CPU读同步。`DMA-BUF fd` 本身不保证CPU cache一致，也不表示无DMA数据移动。
- [dma_alloc.cpp:79–90、93–144](https://github.com/airockchip/librga/blob/2b32edcb97b601b25683e2941d888c8515da6d55/samples/utils/allocator/dma_alloc.cpp#L79)：`DMA_BUF_IOCTL_SYNC` START/END封装、heap ioctl、mmap、close/munmap；CPU访问应以所有权边界配对，不把cache同步当完成fence。
- [rga_async_demo.cpp:119–163](https://github.com/airockchip/librga/blob/2b32edcb97b601b25683e2941d888c8515da6d55/samples/async_demo/src/rga_async_demo.cpp#L119)：release fence→下一步acquire fence→最终imsync。样例教的是依赖关系，不是无等待；同一帧NPU不能读尚未完成的RGA目标buffer。首版同步RGA正确版本往往更可控；异步仅在能与独立工作重叠且in-flight有上限时尝试。

建议C++资源设计：FrameLease持有V4L2/MPP buffer归还操作，DmaBuffer RAII管理fd/mmap/import handle；BufferSlot状态 Free→RGAWriting→NPUReading→Free；preprocess metadata保存源尺寸、stride、色彩矩阵/量程、padding/scale。源帧可在RGA读完后尽早归还，不必等NPU推理，但必须有完成依赖证明。不要为了标称zero-copy让采集池被长时间占满，必要时一份小尺寸copy能降低frame age。

这些阅读能展示的技能：C++ RAII和move-only资源、线程安全有界队列/condition_variable、Linux FD/ioctl/mmap与V4L2 buffer生命周期、DMA缓存一致性与fence、RGA格式/stride约束、RKNN模型输入量化与native布局、分阶段性能分析。收益分别落在排队、预处理、API拷贝/转换、后处理与抖动，必须实测不能相加编造总延迟。
