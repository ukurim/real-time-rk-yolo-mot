# RKNN 输入转换实验

本节对应 `model/RK3588/yolov8n.rknn`，FP16，640×640 RGB，SHA256
`0d17bff40beb4ef0705d91dc11fceae70bb1c23d7e20af742a4e117479bf190c`。
测试运行库 2.3.2，驱动 0.9.8。正式系统性能结果见最新的
[优化实测记录](OPTIMIZATION_RESULTS.md)；这里保留输入路径的正确性依据和早期短探针。

## 采用的路径

`detector.input_mode: fp16_normalized` 显式执行 RGB uint8 → RGB/255 → IEEE binary16，
再调用普通 `rknn_inputs_set(pass_through=1)`。FP16 转换计入 `preprocess_ms`；
`input_submit_ms` 只计随后输入提交。这仍有运行库输入复制，不能称为零拷贝。
C++ `DetectorConfig` 的默认值仍为 `input_mode: uint8`，由运行库完成输入转换/归一化，
作为兼容路径保留。随仓库提供的 `config/video.yaml` 和 `config/camera.yaml` 已显式选择
`fp16_normalized`、一个检测 worker、`core_mask: 7` 和 `pipeline.opencv_threads: 4`。
OpenCV 的四个 CPU 线程与检测 worker 数、NPU 核数是不同参数；该配置请求一帧使用三核，
不代表每个算子都能在三核上执行。独立的 INT8 候选配置 `config/video_int8.yaml` 使用
`input_mode: uint8`，不应用 FP16 归一化路径。

**快速路径仅适用于转换时 mean=[0,0,0]、std=[255,255,255] 的模型。**
这些常量无法从 RKNN 输入属性中恢复，须核对转换 manifest；`fp16_normalized` 是对此契约的显式选择。
`Fp16RgbInput` 在初始化时验证 native 输入必须是 batch=1、NHWC、FP16、RGB 三通道、
无量化或恒等量化、无 stride padding、逻辑/存储尺寸一致；不支持的模型立即报错。
不会自动将该模式应用于 INT8 或其他归一化模型。

代码：`include/aerial/rknn_input.hpp`、`src/yolov8.cpp`。
普通输入和 native 输入属性有区别，必须查询后验证。
`pass_through=1` 绕过数据转换的定义来自
[Rockchip RKNN API](https://github.com/airockchip/rknn-toolkit2/blob/42aa1d426c0a9e0869b6374edba009f7208a1926/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h)。

## 输出一致性与排除的方案

独立工具 `profile_input` 对相同 RGB 字节串分别运行标准路径和候选路径，
全部输出统一请求 FP32 并逐元素比较，任何非有限值/尺寸不一致直接失败。
使用一个覆盖所有 RGB 字节值的合成图案，以及仓库视频 0 秒、8 秒和 45 秒的真实画面：
FP16 `/255` 路径的全部 9 个输出张量、每次 1,218,000 个元素与标准路径**完全一致**。
这证明这些样本上的输入替换一致性，不等于检测数据集精度评估。

在正式 900 帧复测之前，曾执行以下短探针、反证和对照：

| 输入候选 | 正确性 | 探索结果 |
|---|---|---|
| FP16 `/255` + `inputs_set(pass_through=1)` | 合成及三个真实样本逐元素完全一致 | 转换约 1.25–1.28 ms，提交约 0.30–0.35 ms |
| FP16 原始 0–255 + pass-through | 严重不一致 | 拒绝；不能省略归一化 |
| native IO memory UINT8，`pass_through=0` | 合成样本完全一致 | 约 11 ms 转换从 inputs_set 转移到 run，总耗时约 42.7 ms，无实质收益 |
| 原 UINT8 普通 API | 基准 | 输入提交约 11.0–11.1 ms |

上表仅是短探针，包含 CPU 调度波动，不能据此宣称稳定 P95 或完整端到端提升。
后续 900 帧回放中，FP16 快速输入加 OpenCV 四线程、core mask 7 的源 PTS→结果估计延迟
P50/P95 为 89.15/96.83 ms；对应 AUTO 的 P95 为 98.82 ms。
旧 FP16 UINT8 输入加 OpenCV 单线程、AUTO 的 P95 为 116.95 ms。
这些复测同时涉及输入模式、CPU 线程和核配置，不能把全部变化归因于输入转换。
没有真实相机或相机标定，结果使用 `--no-los`；上述时间不是曝光→LOS 的实测延迟。
完整设置、INT8 对照及测量边界见[优化实测记录](OPTIMIZATION_RESULTS.md)。

native UINT8 的普通拷贝与运行库转换仍存在；没有因名字叫 IO memory 就假定加速。
该对照源自
[Rockchip create_mem 示例](https://github.com/rockchip-linux/rknpu2/blob/master/examples/rknn_api_demo/src/rknn_create_mem_demo.cpp)，
但当前板卡/模型实际测试未得到相同的转换融合收益，因此没有作为生产路径。

## 复现

```bash
# 合成图案；退出码 0=逐元素完全一致，2=存在差异，1=操作失败。
./build/profile_input --mode fp16_normalized --iterations 30 --warmup 5

# RGB 文件须恰好是已做 letterbox 的 640*640*3 字节。
./build/profile_input --mode fp16_normalized --rgb build/input_frame_0.rgb

# 保留归一化错误的反证模式；预期返回 2，不作为生产选项。
./build/profile_input --mode fp16_raw --iterations 3 --warmup 1
./build/profile_input --mode io_uint8 --iterations 3 --warmup 1
```

`tests/test_input.cpp` 无须 NPU，验证所有 256 个 uint8 值的半精度误差、通道顺序、
0/1/114÷255 的具体位模式、重复使用输入缓冲区以及不支持属性的拒绝行为。
