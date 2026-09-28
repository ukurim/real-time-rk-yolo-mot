# YOLOv8n INT8 转换与校准来源

`RK3588/yolov8n_int8.rknn` 是独立 W8A8 候选模型；`RK3588/yolov8n.rknn` 保留为 FP16 基线。两者使用相同的官方 split9 ONNX、640×640 RGB 输入和 COCO80 类别。转换器为 RKNN-Toolkit2 **2.3.2**，目标 RK3588，`quantized_algorithm=normal`、`quantized_method=channel`、`mean=0`、`std=255`。算子实际量化/融合和 NPU 核分配应以运行时查询为准；“INT8”不代表每个辅助算子均为 INT8。

## 数据来源与隔离

校准使用 Rockchip 官方 [YOLOv8 转换脚本](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yolov8/python/convert.py) 引用的 [COCO subset20](https://github.com/airockchip/rknn_model_zoo/blob/bad6c7334531becaf90a561988519b7bec34d0ab/datasets/COCO/coco_subset_20.txt)，固定提交 `bad6c7334531becaf90a561988519b7bec34d0ab`。`coco20.sources.json` 保存原始清单哈希、20 张 JPEG 的完整 URL、字节数和 SHA256。下载的图片及派生文件保存在忽略的 `build/model_prepare/coco20/` 下。

这 20 张图只用于校准，不能再算作独立精度评估集。仓库的视频没有参与校准。小规模通用 COCO 校准集用于本轮工程和性能对照，**不能证明航拍、小目标或用户未来相机的量化精度合格**。与 FP16 输出的一致性也不等于有人工真值的 mAP。

图片版权遵循各自来源；[COCO Terms of Use](https://cocodataset.org/#termsofuse) 明确图片版权不属于 COCO Consortium，图片使用须遵循其指向的 Flickr 条款。COCO 标注的 CC BY 4.0 不应当被误当作图片的统一许可。本项目记录出处和下载方法，没有把这些图片作为代码许可证的一部分。

## 预处理对应关系

官方示例直接把原图清单传给 Toolkit。这里先生成 **640×640 无损 PNG**，使校准几何与应用一致，避免校准阶段把非正方形原图直接拉伸到模型输入：

1. OpenCV 解码 BGR，按较短缩放比例保持纵横比；正数尺寸按 C++ `std::round` 规则四舍五入。
2. 使用 `INTER_LINEAR`，左右/上下居中，奇数剩余边在右/下，填充每通道 114。
3. `imwrite` 接收 BGR 并保存标准 RGB PNG。Toolkit 的 `quant_img_RGB2BGR=False` 读取 RGB，不再次交换通道。该参数及其语义来自已校验官方 Toolkit2 wheel 内 `rknn/api/rknn.py::config` 的接口说明。
4. PNG 像素保持 uint8 0..255，除以 255 在 RKNN 图内完成；不在图片里预先归一化。

`dataset.manifest.json` 保存每张原图/派生图的 SHA256、尺寸、padding、OpenCV 版本和预处理配置。转换前检查清单/图片哈希、640×640×3 uint8 形状、重复图以及 RGB 设置。生成的模型 manifest 包含完整校准 manifest 与模型 SHA256，足以追溯该二进制使用的数据。

本次校准 Python 使用 OpenCV 5.0.0，应用 C++ 使用 OpenCV 4.2.0。已把全部 20 张原始 JPEG 经应用的实际 `aerial::letterbox_rgb` 处理，并与校准 PNG 解码后的 RGB 做逐通道比较：**20/20 张完全相同，最大/平均绝对差均为 0**，没有因为版本或 BGR/RGB 交换产生偏差。比较记录在 `build/model_prepare/coco20/preprocessing_comparison.txt`。

## 复现

先按 [FP16 模型说明](YOLOV8N.md) 准备虚拟环境和 ONNX，再从仓库根目录执行：

```bash
build/model_prepare/venv/bin/python model/prepare_coco20.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  build/model_prepare/venv/bin/python model/prepare_yolov8n.py --int8 \
  --dataset build/model_prepare/coco20/dataset.txt \
  --dataset-manifest build/model_prepare/coco20/dataset.manifest.json
```

不传 `--int8` 的原有命令继续生成 FP16；INT8 默认使用独立文件名，并禁止覆盖默认 FP16 基线。可以显式提供其他经过相同预处理和哈希清单记录的真实场景校准集。转换日志应与评估结果一同保存；本次为 `build/model_prepare/yolov8n_int8_conversion.log`。

转换器报告首层和三个分类分支卷积权重有 outlier，提示可能影响量化精度；保留这一警告，是否可用由后续独立数据验证决定。

本次生成文件为 **4,327,819 字节**，SHA256 `6d31f73602f2295ef29ea4d0a7984e458dc19a3b94dbb2a886c1a0e583039661`。FP16 基线仍为 `0d17bff40beb4ef0705d91dc11fceae70bb1c23d7e20af742a4e117479bf190c`，未被替换。模型输入以及九个输出被 Toolkit 自动改为 INT8；运行时应查询元数据并正确转换输入，输出后处理须使用各张量实际 zero point 和 scale，不能沿用 FP16 原始数值解释。
