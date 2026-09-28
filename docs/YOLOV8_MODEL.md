# YOLOv8n RKNN 模型契约与准备

新入口使用 `aerial::YoloV8Detector`，旧 `Yolo11` 不再参与新感知链路。仓库原来的 `yolo11n.rknn`、`test.rknn` **不是**已经验证的 YOLOv8n 模型；文件名和输出 shape 也不足以验证模型身份。实际模型来源、SHA256 和转换配置需要随产物保留。

当前已经生成 `model/RK3588/yolov8n.rknn`：官方 YOLOv8n COCO80、640×640、split9、非量化 FP16，转换工具为 RKNN-Toolkit2 2.3.2。源 ONNX、生成 RKNN 的 SHA256 与张量元信息见 [模型清单](../model/RK3588/yolov8n.manifest.json)，固定上游下载清单见 [来源记录](../model/yolov8n.sources.json)。完整的独立虚拟环境安装和可复现转换入口见 [YOLOV8N.md](../model/YOLOV8N.md)；执行脚本为 [prepare_yolov8n.py](../model/prepare_yolov8n.py)。下面说明模型契约和更换模型时的约束。

后续优化已生成独立的 `yolov8n_int8.rknn` 候选，通过 `config/video_int8.yaml` 使用；
FP16 模型继续保留。当前配置、量化比较与性能结果见[优化实测记录](OPTIMIZATION_RESULTS.md)。

## 已核对的上游

- [Rockchip RKNN Model Zoo v2.3.2](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yolov8)，提交 `bad6c7334531becaf90a561988519b7bec34d0ab`。
- [该版本 C++ 后处理](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yolov8/cpp/postprocess.cc) 和 [RKNN 转换配置](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yolov8/python/convert.py)。本项目独立实现同一 split 输出契约；加入 shape/type 校验、稳定 softmax、浮点框和真实 letterbox 反变换。
- 官方使用的导出器为 [airockchip/ultralytics_yolov8](https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.md)。它将 DFL/框解码移到 CPU，并增加 score-sum 分支；普通 Ultralytics 单一 combined 输出不可直接替换。
- [RKNN Toolkit2 v2.3.2](https://github.com/airockchip/rknn-toolkit2/tree/42aa1d426c0a9e0869b6374edba009f7208a1926)，提交 `42aa1d426c0a9e0869b6374edba009f7208a1926`。模型编译器、板端 runtime 与 NPU driver 的兼容性仍需实际加载验证。

## 输入与输出

输入是一个静态 batch=1 的三通道模型，宽高均为 32 的倍数。应用接收原始 BGR `CV_8UC3` 图像，保持长宽比 resize（`INTER_LINEAR`），居中 letterbox，RGB padding=114；缩放后的整数宽高由 `round` 得到，剩余奇数 padding 加到右/下边。每帧记录实际 `resized_width / original_width`、`resized_height / original_height` 和整数 padding，逆变换后裁剪到原图，保持浮点 xywh。

转换模型时使用 `mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]`，输入提交支持两种显式模式：

- `input_mode: uint8`：RGB UINT8 NHWC，`pass_through=0`，由运行库归一化，应用不再次除以 255。这是第一版的输入路径，也是 C++ `DetectorConfig` 保留的默认值及当前 INT8 配置所用模式。
- `input_mode: fp16_normalized`：应用先执行 RGB/255→FP16，再以 `pass_through=1` 提交；运行库跳过归一化。仅接受已核对上述转换常量、native NHWC FP16 恒等量化且无 stride padding 的模型。FP16 转换耗时计入预处理，普通 API 输入复制仍存在。正确性对照与约束见[输入路径实验](INPUT_PATH.md)。

当前 `config/video.yaml`、`config/camera.yaml` 对本仓库 FP16 模型显式选择 `fp16_normalized`、一个检测 worker、`core_mask: 7` 和 OpenCV 四线程；`config/video_int8.yaml` 使用 UINT8 输入。请求三核不保证每个算子都能拆分到三核。`preprocess=cpu` 仍是已验证的 letterbox 路径；RGA 参数会明确报错，不以未经验证的旧拉伸代码替代 letterbox，也不声称零拷贝。

只支持 Rockchip 优化导出的 6/9 个普通 NCHW 输出，batch=1，按 stride **8、16、32** 三组排列：

| 每组位置 | shape | 含义 |
|---|---|---|
| 0 | `[1,4*reg_max,H/stride,W/stride]` | 四边的 DFL logits |
| 1 | `[1,num_classes,H/stride,W/stride]` | 已经过 sigmoid 的类别概率 |
| 2，可选 | `[1,1,H/stride,W/stride]` | 上游提供的 score sum/clip，用于快速筛选 |

类别数量从三个 score 张量校验推断，不写死 5 或 80。支持 INT8/UINT8 affine 每 tensor 量化、FP16/FP32；浮点输出通过 RKNN `want_float=1` 获取 FP32，整数输出按各自 `zp/scale` 解码。每个 output 都显式设置 index。未知输出布局、类别数不一致、截断 buffer、错误 quantization 会报错。单输出 `[1,84,8400]`、端到端 NMS 模型、segmentation、pose 暂不支持。

本机 runtime 2.3.2 / driver 0.9.8 实测：未量化官方模型的九个 FP16 输出仍标记 `AFFINE_ASYMMETRIC`，全部 `zp=0, scale=1`。这是恒等 metadata；浮点输出检查因此允许 `NONE` 或该恒等组合，仍由 runtime 转 FP32，不把 FP16 当整数解量化。其他浮点量化组合仍拒绝。

每网格保留最高分的一个类别，执行分类别 NMS。默认检测阈值 0.1，NMS 0.7，最多 300 个框；阈值应不高于跟踪器的低分关联阈值，避免提前删去 BoT-SORT 第二阶段需要的候选。输入仍是有畸变原图，检测阶段不去畸变；框中心之后在 LOS 模块用真实标定去畸变，不能混用 letterbox 坐标。

检测阈值采用 `score >= threshold`。只有 score 与 score-sum 都是浮点时才用 sum 做预筛选；整数或混合输出直接检查反量化的类别概率。不同 tensor 的量化步长可能使 sum 小于最大类别分数，因此不以量化 sum 提前删除靠近阈值的跟踪候选。跟踪器的高/低分关联边界仍保持自身的上游语义。

## 准备模型

先建立独立 Python 环境并安装固定版本 Toolkit2；不要覆盖系统 Python 的依赖。ARM64 Python3.8 wheel 路径为上述固定提交中的：

```text
rknn-toolkit2/packages/arm64/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

Rockchip v2.3.2 下载脚本提供的 YOLOv8n ONNX 地址：

```sh
mkdir -p model/onnx
curl -L --fail -o model/onnx/yolov8n.onnx \
  https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8n.onnx
sha256sum model/onnx/yolov8n.onnx
```

在固定提交的 Model Zoo `examples/yolov8/python/` 中按官方 `convert.py` 转换：

```sh
python convert.py /absolute/path/yolov8n.onnx rk3588 fp /absolute/path/yolov8n.rknn
```

FP 模型适合先验证链路，不需要量化数据集。INT8 转换使用 `i8`，并将 `DATASET_PATH` 指向有代表性的本场景图像清单；官方自带少量 COCO 图片只适合 smoke test，不等于航拍部署精度验证。量化校准数据与相机内参标定是不同内容。

部署为 `model/RK3588/yolov8n.rknn`，并记录 ONNX/RKNN SHA256、上游提交、Toolkit2 版本、RGB/255 配置、量化模式与数据集。若从自定义 `.pt` 导出，需要固定 Rockchip fork 的提交并按其 `RKOPT_README.md` 导出；不能只用 `yolo export` 的默认 combined 输出。

## 验证边界

`test_detector` 无需 NPU，覆盖奇数 padding 与舍入、非连续源 ROI、BGR→RGB、6/9 输出、INT8/UINT8/FP32 等价解码、稳定 DFL、低分候选、分类别 NMS、框裁剪及错误契约拒绝。这些测试不替代真实 RKNN 图像精度验证。

后续已执行的输入一致性、量化对照及分项计时见[优化实测记录](OPTIMIZATION_RESULTS.md)；
这些抽样不替代有标注的部署场景精度评估。当前普通 RKNN API 可能存在运行库内部复制；没有采用 DMA FD I/O 绑定，不宣称已完成硬解→RGA→NPU 零拷贝。当前没有真实相机标定，视频性能测试使用 `--no-los`，不会输出假设内参算出的角度或宣称传感器曝光到 LOS 的实测延迟。
