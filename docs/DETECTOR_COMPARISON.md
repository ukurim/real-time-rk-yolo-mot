# FP16 / INT8 检测结果对照协议

该工具对照的是**跟踪前、经过相同 NMS 的检测结果**。FP16 仅作为参考模型，不是标注真值；两模型一致也可能同时漏检或错检。输出不称作 mAP、实际 precision/recall 或“INT8 精度无损”。速度须使用独立 profiling / 端到端实验评估，不能直接比较本工具先后执行两个模型的计时。

## 输入与可复核性

`tools/compare_detectors.cpp` 复用应用的 `GstCapture`、`YoloV8Detector` 与原图坐标还原，离线解码视频且不主动丢帧。帧从零编号，选取 `start_frame + k * stride`，不按处理速度抽样。参考模型和候选模型顺序使用同一个 BGR 图像。`--copy-input 1` 默认复制选中帧到 CPU 内存并释放解码器持有对象；复制不计入两模型的阶段时间。

生成 JSONL 包含：

- `run`：视频和两个模型的绝对路径、字节数、SHA256；采样起点、间隔与目标数量；解码器、输入尺寸、置信度阈值、NMS 阈值、类别数量与请求 NPU 核掩码。
- `frame`：样本序号、源帧编号、源 PTS、原始图像尺寸；按行、不含 stride padding 的 BGR 像素 SHA256；两侧的类别、分数、原图 `bbox_xyxy` 与阶段时间。
- `summary`：解码数、选中数、EOF 与中断状态。分析脚本拒绝没有结束记录或被中断的输入；若视频提前结束，报告实际样本数，不能将请求数当作实测数。

为避免 RKNN 运行库的标准输出污染，原始结果只能写到文件，不支持 `--output -`。比较工具会读取完整视频以计算 SHA256，这发生在模型加载之前，且不属于推理计时。

## 运行

```bash
mkdir -p build/quality
./build/compare_detectors \
  --baseline model/RK3588/yolov8n.rknn \
  --candidate model/RK3588/yolov8n-int8.rknn \
  --input 1080p60hz.mov \
  --output build/quality/video_comparison.jsonl \
  --start-frame 0 --stride 45 --samples 120 \
  --decoder auto --copy-input 1 --core-mask 0 \
  --confidence 0.1 --nms 0.7 --max-detections 300

python3 scripts/analyze_detector_comparison.py \
  build/quality/video_comparison.jsonl \
  --output build/quality/video_summary.json \
  --details build/quality/video_matches.jsonl

python3 scripts/test_detector_comparison.py
```

模型路径为示例，按实际转换产物调整。`--decoder mpp` 可使用已有硬件解码路径，但要求主应用相同的设备权限与 GStreamer 插件；一次对照只用一种解码结果喂给两模型。不同解码器的颜色转换误差可能影响检测，不能把不同解码运行之间的差异都归因于量化。

## 匹配约定

默认分别在 **0.1、0.25、0.5** 置信度阈值上报告，过滤条件均为 `score >= threshold`。生成时阈值须不高于分析阈值，默认 0.1。各阈值对同一组已 NMS 输出作筛选，没有重新执行 NMS。

每帧、每个类别单独执行一对一全局关联：IoU 小于 0.5 的边禁止，先最大化可匹配对数，再最大化匹配对的总 IoU。使用带虚拟未匹配列的 Hungarian 算法，权重中的匹配奖励高于其余所有 IoU 的可能增益；因此不会出现贪心最高 IoU 消耗唯一候选而少配一对的问题。类别不同时，即使框完全重合也不配对。

汇总包括两侧框数、匹配数、各自未匹配数、相对于各侧的匹配比例，以及已匹配框 IoU、候选减参考的分数差、绝对分数差、框中心像素位移的分布，并按类别给出计数。详细文件保留每个匹配对及未匹配项在原始帧记录中的索引。空集合对应比例/分布输出 `null`，不伪造 100% 一致。

阈值附近的分数变化可能导致框从一侧消失；未匹配项应结合原始记录核查，不直接称作误检或漏检。

## 校准集与评估集隔离

本次首先使用仓库 `1080p60hz.mov` 作未标注评估视频。**不可使用此视频或其抽帧作为 INT8 量化校准数据，再以此视频的一致度声称泛化效果。** 每次实验须保存量化校准源清单、SHA256 / 来源与转换清单，并将视频及抽帧来源与该清单核对。

工具记录足够的源和像素校验信息，但不会自动断言校准隔离，报告中明确标记 `calibration_overlap: not_checked_by_tool`。隔离结论须来自本次转换的数据来源核查。若后续做 COCO 标签评估，应预先固定与校准集互斥的图片 ID，使用官方标注与 COCO 评估流程单独报告 mAP；本工具的检测一致度不能替代这一评估。
