# RKNN 延迟、单帧三核与官方基准核对

2026-09-24，在第一版实测之后追加排查。此阶段没有更换模型、修改硬件频率或把三实例吞吐换算成单帧延迟，当时为 640×640、batch=1、FP16 YOLOv8n；原 1920×1080 图像缩放为 640×360，上下各填充 140 像素。

本文保留优化前的定位证据。后续已完成显式 FP16 输入转换、INT8 模型及完整链路复测，最新耗时与质量比较见 [OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md)；下文的约 43 ms 不再是默认快速输入的当前数值。

## 三种配置必须区分

- `workers=1, core_mask=0`：一个上下文，AUTO 自动选择 NPU 核，不能解释为自动使用全部三个核。
- `workers=3, core_mask=0`：本程序分别给三个上下文指定 core0/1/2，处理三张不同帧。
- `workers=1, core_mask=7`：向运行库请求三个核协同处理同一次推理。每个算子实际如何分核，仍取决于模型编译产物和运行库，不能假设耗时除以三。

AUTO 与 combine 的定义来自 [Rockchip RKNN v2.3.2 API](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h)。当前本地 `include/rknn_api.h` 使用相同语义。

## 约 43 ms 的实际组成

对真实视频再次跑 180 个接收帧：MPP BGR、copy-input、单实例 AUTO、关闭显示/编码/LOS，得到 54 个结果；去掉前 10 个结果后统计 44 个样本。新增计时只读取单调时钟，不启用运行库逐层 profiling。

| 调用/阶段 | P50 ms | P95 ms |
|---|---:|---:|
| 输入提交 `rknn_inputs_set` | 11.04 | 11.21 |
| 执行 `rknn_run` | 28.89 | 29.62 |
| 输出获取 `rknn_outputs_get` | 3.00 | 3.08 |
| 原 `inference` 合计 | 42.95 | 43.68 |

输入提交还包含运行库数据处理：主机提供 UINT8 RGB NHWC，模型输入属性为 FP16 NHWC，模型配置包含 `/255`。不能把这一段全称为 memcpy，也尚未逐步确定其中转换/归一化/拷贝各自的比例。输出获取请求 FP32，原始模型输出为 FP16。`rknn_run` 为 API 墙钟时间，仍不等于逐个 NPU kernel 的纯计算时长。

JSON 的 `input_submit`、`rknn_run`、`output_get` 是 `inference` 的子项；`inference` 保留原定义以兼容已有统计，不能把四项重复累加。

## 独立微基准与真实核分工

新增 `tools/profile_rknn.cpp` / `build/profile_rknn`。输入固定为 640×640 RGB=114，排除视频、预处理、跟踪和应用队列，**仅测 API，不测检测准确率**。相同模型、runtime 2.3.2、driver 0.9.8；默认关闭 COLLECT_PERF，同步单上下文。分别测 AUTO、core0、三核；另外仅给测试进程固定 CPU4，减少大小核迁移对输入/输出转换的影响，不修改应用或系统全局亲和性。

CPU4 固定后，每项预热 10 次、统计 60 次的 P50：

| 请求 NPU mask | inputs_set ms | run ms | outputs_get ms | API 合计 ms |
|---|---:|---:|---:|---:|
| AUTO=0 | 10.93 | 28.33 | 2.95 | 42.22 |
| core0=1 | 10.92 | 31.84 | 2.98 | 45.76 |
| 三核=7 | 10.92 | 27.45 | 2.96 | 41.43 |

运行库 `RKNN_QUERY_PERF_RUN` 与同一轮 `rknn_run` 墙钟值接近。API 总计在每轮先相加再取分位，不是表中几个分位简单相加。以上短测试仍有系统波动，不能用 AUTO/core0 的差异推断确定的内核调度原因。

独立未固定 CPU 的 100 次微基准同时采样 `/sys/kernel/debug/rknpu/load`：AUTO 主要为 Core0 约 58–61%、Core1/2 为 0；mask7 主要为 Core0 约 57–60%、Core1/2 各约 6%。采样包含主机 API 间隙，不是单个算子的利用率，且系统计数更新有时间窗。

再单独开启 COLLECT_PERF，仅用于检查算子位置，不把受 profiling 影响的单帧耗时作为无干扰基准。mask7 逐层报告：

| 算子组 | 数量 | 实际分工 | 该次报告耗时 |
|---|---:|---|---:|
| ConvExSwish | 57 | 全部 Core0 | 22.672 ms，占总计 82.64% |
| ConvSigmoid | 3 | 全部 Core0 | 0.497 ms |
| Add、Concat、Conv、ConvClip | 25 | 分到三个核 | 2.512 ms |
| 其余 NPU 算子 | 13 | Core0 | 1.549 ms |
| 输入/输出包装 | 10 | CPU | 0.206 ms |

66 个卷积类算子中 60 个仍为单核，6 个为多核。没有卷积回落 CPU 的证据；表中的 CPU 包装不包含上面 `inputs_set` 的 11 ms。

**当前编译图的主要计算没有有效分摊到三核**，这是 mask7 提升有限的直接证据。该结论限定于这份 FP16 编译产物和当前 runtime/driver，不能泛化成所有 FP16 模型都不能多核。本轮也尚未证明是哪一项编译决策导致这些融合卷积未拆分。

[官方 SDK v2.3.2 用户指南 §5.3](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_EN.pdf)也说明多核模式不保证逐层切分：小网络可能收益有限；某层 100/0/0 可能因为任务量不足切分粒度，或该算子未实现多核切分。这些是官方列出的可能性，不是对本模型具体原因的确诊。

留存证据：[API 汇总与核负载](validation/rknn_api_profile.json)、[mask0 算子报告](validation/rknn_mask0_detail.txt)、[mask7 算子报告](validation/rknn_mask7_detail.txt)。每次微基准 CSV/log 在本地 `build/rknn_profile/`，真实视频分项输出在 `build/rknn_api_split.jsonl`。

## 和官方数值比较

[Rockchip Model Zoo v2.3.2 固定版本性能表](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/README.md#model-performance-benchmarkfps)列出 YOLOv8n `[1,3,640,640]`、**INT8、RK3588 单核 73.5 FPS**，其倒数约 13.61 ms。表注说明最高 NPU 频率、仅模型推理、不含前后处理。该表未给 YOLOv8n FP16 或三核数值，也未明确到每个 C API 的计时边界；不能直接认定它等于 `rknn_run` 或纯 kernel 时间。

本项目当前是 FP16，还把约 11 ms 输入提交和 3 ms 输出获取纳入 `inference`。因此存在**精度格式和计时范围两项差异**。同时，当前 `run` 约 29 ms 和输入提交约 11 ms 确实还有优化空间，不能把“口径不同”当成已经接近硬件最佳性能的证明。

[Ultralytics YOLOv8 性能表](https://docs.ultralytics.com/models/yolov8/#performance-metrics)中 YOLOv8n 640 的 0.99 ms 是 **A100 TensorRT**，不是 RK3588。若比较其他官方表，仍须核对模型版本、设备、量化精度和计时边界。

下一步可用可追溯校准数据构建 INT8 对照，保持 FP16 作为精度参照；随后重新检查分核与每阶段耗时，并验证检测/跟踪质量。输入提交的转换/归一化也应单独定位，不能预先承诺使用共享内存就能消掉全部 11 ms。GMC/跟踪约 44 ms 是独立于上述 NPU 差异的端到端瓶颈。

## 复现

```bash
cmake --build build --target profile_rknn -j4
taskset -c 4 ./build/profile_rknn --mask 0 --iterations 60 --warmup 10 --samples build/profile_auto.csv
taskset -c 4 ./build/profile_rknn --mask 7 --iterations 60 --warmup 10 --samples build/profile_three.csv
./build/profile_rknn --mask 7 --iterations 1 --warmup 1 --collect-perf --perf-detail build/profile_detail.txt
```

普通计时不要打开 COLLECT_PERF。只改变 NPU mask 的比较与多 detector 实例比较是两种不同实验。
