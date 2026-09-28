# 640 输入低延迟优化与复测

2026-09-24，RK3588 实机。已实现 FP16 输入快速转换、OpenCV 并行度配置、INT8 转换/对照工具以及更细的队列和跟踪计时。保留 YOLOv8n 的 **640×640、batch=1**；1920×1080 原图缩放为 640×360，上下各补 140 像素。没有降低输入分辨率或关闭 BoT-SORT GMC。

## 结果与默认配置

同一视频从起点按 60 fps 回放，每项接收 900 帧，约 15 秒；MPP 直接输出 BGR，选中的帧复制至普通 CPU 内存，JSON 写本地文件，显示/编码关闭。每项剔除前 10 个输出后计算分位数。runtime/Toolkit 2.3.2，driver 0.9.8；未修改系统频率或应用 CPU 亲和性。视频及模型 SHA、运行命令、逐阶段 P50/P95/P99、丢弃统计留在 [完整汇总](validation/optimization_results.json)。

**没有实际相机和标定，全部使用 `--no-los`。** 下表是文件回放 PTS→检测/跟踪结果就绪的估计，LOS 为 null，不是传感器曝光到有效 LOS 的验证。软件解码 `auto` 路径可直接运行，但不对应此表的 MPP 性能。

| 配置 | NPU mask | OpenCV 线程 | 输出/900 | 源 PTS→结果 P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|---:|---:|
| 原 FP16 UINT8 输入 | AUTO | 1 | 269 | 108.95 | 116.95 | 118.71 |
| FP16 显式归一化输入 | AUTO | 4 | 306 | 90.58 | 98.82 | 100.11 |
| FP16 显式归一化输入 | 7 | 4 | 313 | 89.15 | 96.83 | 98.10 |
| INT8 UINT8 输入 | AUTO | 4 | 487 | 66.23 | 74.07 | 78.08 |
| INT8 UINT8 输入 | 7 | 4 | 497 | 65.35 | 71.79 | 73.33 |

均为单 detector 实例。最后一行应用接收 900 帧、最新帧槽丢弃 403 帧，完成队列/过期/JSON 丢弃均为 0。丢帧是实时新鲜度策略，不能把 497/900 描述成处理了全部 60 fps 图像。分位数反映此次样本，不是最坏时间保证。

`config/video.yaml`、`config/camera.yaml` 保留 FP16 模型，显式采用 `input_mode: fp16_normalized`、`workers: 1`、`core_mask: 7`、`opencv_threads: 4`。FP16 P95 比本轮基准下降约 **17.2%**。INT8 作为独立候选放在 `config/video_int8.yaml`，P95 下降约 **38.6%**，但检测输出有变化，见下文。AUTO 仍可通过 CLI 比较；mask7 的小幅收益不代表所有算子在三核均分。

## 耗时究竟在哪里

下表为上述基准与两种 mask7 配置的每阶段中位数，单位 ms。各阶段分位数不能相加推导总体分位数。

| 阶段 | 原 FP16 | 优化 FP16 | INT8 |
|---|---:|---:|---:|
| 输入槽等待 | 8.83 | 8.86 | 8.86 |
| 预处理，含选中帧复制 | 8.93 | 11.83 | 10.73 |
| RKNN API 合计 `inference` | 43.08 | 32.07 | 15.87 |
| └ `input_submit` | 11.03 | 0.37 | 0.29 |
| └ `rknn_run` | 29.06 | 28.62 | 14.96 |
| └ `output_get` | 2.98 | 2.99 | 0.59 |
| 后处理 | 3.76 | 3.77 | 3.35 |
| 完成后等待跟踪 | 0.012 | 0.013 | 0.010 |
| 跟踪总计 | 43.85 | 32.89 | 25.98 |
| └ GMC | 43.77 | 32.83 | 25.93 |
| └ 关联/Kalman/生命周期 | 0.067 | 0.065 | 0.056 |
| 就绪到开始发布 | 0.106 | 0.123 | 0.111 |

`rknn_run` 是同步 API 墙钟时间，含运行库调度；不是裸 NPU kernel 时间。新 JSON 分项 `tracking_gmc`、`tracking_association` 属于 `tracking`；RKNN 三项属于 `inference`，不可重复累计。`completion_queue` 与采集后的 `queue` 分开计时，便于发现多实例完成结果等待串行跟踪的问题。

FP16 路径将 RGB/255 和半精度转换显式放到 CPU 预处理，短探针转换约 1.25–1.28 ms；它已计入上表预处理，没有隐藏掉。随后使用普通 `inputs_set(pass_through=1)`，提交从约 11 ms 降至约 0.37 ms。代码验证 native FP16/NHWC/尺寸/stride 契约；归一化常量须对应模型 manifest。C++ 的通用默认仍为 UINT8。native IO memory 的 UINT8 实验只是把转换从 inputs_set 移到 run，没有总延迟收益，因此未引入这条复杂路径。[输入实验](INPUT_PATH.md)记录了正确性与反证。

INT8 独立同步微基准，固定 CPU4、常量图、预热 10 次再测 100 次：AUTO 的 `run` P50=15.37 ms、API 总计=16.25 ms；mask7 分别为 **13.88 ms / 14.55 ms**。这是隔离视频/GMC 的 API 测量，不能代替上表真实链路。具体对照口径见 [官方基准核对](RKNN_PROFILING.md)；[INT8 API/核负载](validation/int8_api_profile.json)保留原始汇总。

INT8 mask7 的逐层报告中，60 个 ConvExSwish/ConvSigmoid 仍全部在 Core0。不能以“请求三个核”推出推理耗时除以三。[算子报告](validation/int8_mask7_detail.txt)开启了 COLLECT_PERF，只用于查看核分工，不能把其中耗时作为无干扰基准。

## 并发由实测选择

每个候选先用 300 个接收帧比较，表中为源 PTS→结果 P95 ms；其他参数一致，GMC 参数保持不变。

| 模型/OpenCV 线程 | 1 实例 AUTO | 1 实例 mask7 | 2 实例各一核 | 3 实例各一核 |
|---|---:|---:|---:|---:|
| INT8 / 1 | 111.89 | 112.85 | 93.51 | 96.64 |
| INT8 / 2 | 90.46 | 91.85 | 76.16 | 83.95 |
| INT8 / 4 | 74.31 | 71.95 | 78.40 | 79.25 |
| INT8 / 8 | 77.87 | 80.32 | 未测 | 未测 |
| FP16 快速输入 / 4 | 98.19 | 96.99 | 121.03 | 133.37 |

单线程 OpenCV 时，多 detector 有局部收益；优化 GMC 的 CPU 并行度后，单 detector 的延迟更低，因此采用 1 实例 + OpenCV 4 线程，再进行上面的 900 帧复测。这里的 4 是 CPU 图像算法线程数，与 NPU 核数无关；整个程序仍有采集、检测、跟踪和发布等线程。

GMC 使用同一 sparseOptFlow、相同半尺寸图像和最多 1000 个角点，未减少角点、关闭补偿或改变关联阈值来换性能。FP16 与 INT8 的跟踪阶段耗时不能完全归因于量化：实时选中的帧和帧间隔也会变化。下一处主要 CPU 瓶颈仍是 GMC，其次为选中帧复制和预处理；若继续修改须一起验证跟踪稳定性。

## INT8 检测差异与功能回归

INT8 由同一官方 ONNX 构建，使用固定版本官方 COCO 子集 20 张图做 W8A8、normal/channel 量化。每张图的来源、SHA、RGB letterbox 和转换 manifest 均可追溯，20 张预处理图已与 C++ 路径逐像素一致。转换中的权重 outlier 告警保留，未当作精度验证通过。模型和复现步骤见 [YOLOV8N_INT8.md](../model/YOLOV8N_INT8.md)。这 20 张图只够工程起步，并不代表实际航拍场景的校准覆盖。

评估使用未参与量化的仓库视频：从第 0 帧起每隔 45 帧抽取，共 120 帧，覆盖约 89.25 秒。两个模型依次处理完全相同的选中像素；按同类、IoU≥0.5 做一对一最大匹配，比较 NMS 后、跟踪前的原图框。该实验使用 AUTO 核请求；不是 mask7 精度认证。工具记录了模型、视频及每帧像素 SHA。

| 分数阈值 | FP16 框 | INT8 框 | 匹配数 | 匹配数/FP16 框 | 未匹配 FP16/INT8 | 匹配框中心偏差 P95 px |
|---|---:|---:|---:|---:|---:|---:|
| 0.10 | 2286 | 2208 | 2133 | 93.31% | 153 / 75 | 2.50 |
| 0.25 | 1548 | 1516 | 1485 | 95.93% | 63 / 31 | 1.87 |
| 0.50 | 970 | 920 | 910 | 93.81% | 60 / 10 | 1.62 |

0.25 下匹配 IoU 中位数为 0.972，分数绝对差 P95=0.079。**这是与 FP16 的一致性，不是 mAP，也不能视为漏检率或真实精度。** 尤其 BoT-SORT 依赖 0.1/0.5/0.6 分数阈值，分数漂移可能改变 ID 和轨迹。没有真实标定，不能把像素偏差转换成已验证的角误差。完整 [对照汇总](validation/int8_agreement.json)及[工具说明](DETECTOR_COMPARISON.md)可复核。

功能验证另使用软件解码、离线保留 120 帧：原 FP16 UINT8/CV1 与默认 FP16 快速输入/CV4 均为 mask7，得到的全部目标字段完全相同：955 条目标记录，其中 69 条预测，ID/框/分数/更新标志均无变化。INT8 为 949 条记录、74 条预测。三组 JSON 均通过顺序、数值、预测状态和 null LOS 检查。FP16 独立探针的合成图及视频 0/8/45 秒画面，全部 9 个输出张量、每次 1,218,000 个元素逐元素一致。八项 CTest 已通过，覆盖 FP16 输入属性、检测对照匹配器及原有检测/跟踪/LOS/缓冲/慢消费者测试。

所有 23 组短/长延迟实验 JSON 均通过验证。原始文件留在 `build/{fp16_baseline_final,fp16_fast_final,int8_final,int8_cv*,fp16_fast_cv4,optimization_smoke,quality}/`。尚未覆盖真实相机时间戳、镜头标定、带标注的检测/跟踪评估和持续温控运行。

## 运行与复现

```bash
./build-linux_RK3588.sh
# 保持 FP16；默认 offline 用于检查完整轨迹，--realtime 开启最新帧回放。
./build/Aerial_detection_demo --config config/video.yaml --no-los --realtime --max-frames 900
# 显式选择较快的 INT8 候选；该配置默认 realtime。
./build/Aerial_detection_demo --config config/video_int8.yaml --no-los --max-frames 900

# 本机测表中 MPP 路径，使用独立 registry；未修改系统 DMA heap 权限。
sudo env GST_REGISTRY=/tmp/aerial-mpp-root-full.bin ./build/Aerial_detection_demo \
  --config config/video_int8.yaml --no-los --decoder mpp --copy-input --max-frames 900

# 基准必须显式指定旧输入模式/CV线程，避免默认配置变化影响复现。
sudo env GST_REGISTRY=/tmp/aerial-mpp-root-full.bin python3 scripts/benchmark_latency.py \
  --frames 900 --decoder mpp --copy-input --input-mode uint8 --cv-threads 1 \
  --cases one_auto --output-dir build/recheck_baseline
sudo env GST_REGISTRY=/tmp/aerial-mpp-root-full.bin python3 scripts/benchmark_latency.py \
  --frames 900 --decoder mpp --copy-input --input-mode fp16_normalized --cv-threads 4 \
  --cases one_auto one_all_cores --output-dir build/recheck_fp16_fast
sudo env GST_REGISTRY=/tmp/aerial-mpp-root-full.bin python3 scripts/benchmark_latency.py \
  --frames 900 --decoder mpp --copy-input --model model/RK3588/yolov8n_int8.rknn \
  --input-mode uint8 --cv-threads 4 --cases one_auto one_all_cores --output-dir build/recheck_int8
```

在 FP16 配置上只用 `--model` 替换成 INT8 会被 native FP16 契约校验拒绝；须同时 `--input-mode uint8`，或直接使用 INT8 配置。真实 LOS 仍须提供真实标定并去掉 `--no-los`。
