# YOLOv8n 模型来源与转换

`RK3588/yolov8n.rknn` 来自 Rockchip 官方 model zoo 提供的 **YOLOv8n COCO80 优化版 ONNX**，不是仓库旧有 YOLO11 模型，也不是把 `.pt` 改扩展名。来源固定到 model zoo 提交 `bad6c7334531becaf90a561988519b7bec34d0ab`（v2.3.2）；下载清单和 SHA256 见 `yolov8n.sources.json`。官方说明上游为 [airockchip/ultralytics_yolov8](https://github.com/airockchip/ultralytics_yolov8)，模型及工具链分别遵循各自上游许可。

官方图已把原 YOLOv8 输出拆成三尺度的 box/class/score-sum 共 9 个输出，并将 DFL 解码移至 CPU。适配代码使用 `split9`，类别数 80，分布回归 16 bins，输入 640×640 RGB。推理输入为 `uint8` 的 `[0,255]` 像素，`mean=0, std=255` 在 RKNN 转换配置中设置；不要在 C++ 中再次除以 255。预处理使用 RGB letterbox，后处理需逆去 padding/缩放回原图。

首版采用 **非量化 FP16** RKNN，优先完成可验证的真实检测链路。没有拿随机图片或测试视频冒充代表性量化标定集。INT8 可在真实目标场景数据上另做量化、精度和延迟对比，不能由当前 FP16 结果推断 INT8 性能。

后续 INT8 工程对照已提供独立转换入口和官方 COCO subset20 校准数据来源，见 [INT8 转换与限制](YOLOV8N_INT8.md)；该候选模型不覆盖这里的 FP16 基线。

转换宿主为 Ubuntu 20.04 aarch64/Python 3.8，使用官方 RKNN-Toolkit2 2.3.2 arm64 wheel，与仓库运行库 2.3.2 对齐。Python 依赖只安装进 `build/model_prepare/venv`；C++ 程序不需要该虚拟环境。模型结构、输出 shape、生成时间和生成文件 SHA256 由脚本记录到 `RK3588/yolov8n.manifest.json`。

## 复现

先准备 Python 3.8 venv 支持、C++ 编译器和 CMake。Ubuntu 20.04 可使用 `sudo apt-get install python3.8-venv python3.8-dev build-essential cmake`。`onnxoptimizer==0.2.7` 在 arm64/Python3.8 上可能从源码编译，首次安装需数分钟。下面从仓库根目录执行：

```bash
python3.8 -m venv build/model_prepare/venv
build/model_prepare/venv/bin/python -m pip install 'pip<25.1'
curl --fail --location --output build/model_prepare/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
  'https://raw.githubusercontent.com/airockchip/rknn-toolkit2/42aa1d426c0a9e0869b6374edba009f7208a1926/rknn-toolkit2/packages/arm64/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl'
sha256sum build/model_prepare/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 必须等于 d78e2ecd77502988dc2dcd46d665102be8fb15f4d4d541ef272f6abaabca0eda
MAX_JOBS=2 build/model_prepare/venv/bin/python -m pip install \
  --no-binary ruamel.yaml.clib -c model/yolov8n-conversion-requirements.txt \
  build/model_prepare/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
build/model_prepare/venv/bin/python model/prepare_yolov8n.py --download
```

脚本校验源 ONNX 的固定 SHA256、检查 ONNX 图，并拒绝 Toolkit2 版本不匹配。仅生成模型及清单，不配置 NPU/CPU 定频，也不会自动启动摄像头、显示或网络推流。转换器不承诺 RKNN 二进制跨机器逐字节可重现，生成后的实际哈希是审计依据。

`ruamel.yaml.clib==0.2.8` 的此平台预编译 wheel 内部标记为 `manylinux_2014_aarch64`，导致 `pip check` 报不支持的平台；因此复现命令显式从源码构建该小依赖，避免沿用错误的 wheel 元数据。

模型为通用 COCO 类别，不等于航拍场景已达到目标精度；实际镜头、尺度、遮挡条件仍需验证。LOS 校准与检测模型无关，不能通过此模型文件获得相机内参。
