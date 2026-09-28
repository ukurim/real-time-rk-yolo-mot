# 相机坐标系 LOS

`aerial::LosProjector` 将跟踪框中心转换成相机坐标系视线方向。检测框和跟踪框均为**原始、未去畸变输入图像**的浮点 `x,y,width,height`；检测器必须先逆变换模型输入的 resize/letterbox，不能把网络输入坐标直接传给 LOS。

## 坐标与输出

相机光心为原点：`+X` 向图像右侧，`+Y` 向图像下侧，`+Z` 沿光轴向前。观测像素为 `(x+width/2, y+height/2)`。OpenCV 根据配置的内参和畸变模型将其去畸变到归一化坐标 `(xn,yn)`，再计算：

```text
unit_vector = [xn, yn, 1] / sqrt(xn² + yn² + 1)
azimuth_rad = atan2(xn, 1)                 # 右正，左负
elevation_rad = atan2(-yn, hypot(xn, 1))   # 上正，下负
angle_deg = angle_rad * 180 / pi
```

角度原始单位为弧度；LOS 是方向，不包含距离。俯仰角定义相对于相机 `XZ` 平面，不是独立使用 `atan2(-yn,1)`。光轴方向是 `[0,0,1]`，两角均为 0。第一版没有机体/世界坐标系转换、视觉/IMU 融合或控制闭环。

同一帧内每个目标分别计算 LOS。当前帧由检测更新的框与跟踪器纯预测框都可以计算，但必须保留跟踪输出的 `detection_updated` 标记；纯预测框不是当帧视觉测量。无效/非有限框、中心已离开输入图像、去畸变失败都会得到 `valid=false`，调用者不得把其零值字段解释为光轴目标。实现用重投影误差 ≤0.05 像素确认畸变逆解，避免把有限但未收敛的数值当成结果。

发布的 JSON 将内部 `bbox` 的 xywh 转为 `bbox_xyxy: [x1,y1,x2,y2]`，仍为原图浮点坐标。目标项包括 `track_id`、`class_id`、`score`、`detection_updated`、`observation`、`seconds_since_update` 和 `los_valid`；有效 LOS 放在 `los.unit_vector_camera`、`los.azimuth_rad`、`los.elevation_rad`。无效结果直接发布 `los:null`、`los_valid:false`，不会把内部默认零值发布成真实角度。

帧级 `frame_id`、`image_timestamp_ns` 及 `image_timestamp_domain` 适用于该帧全部目标；`frame_id` 是应用收到的帧序号，不是传感器帧号。`result_ready_monotonic_ns` 记录跟踪/LOS 计算完成时间，早于序列化、绘图与视频输出；`publish_started_monotonic_ns` 记录 JSON 开始写出时间，也不等于外部消费者收完数据的时间。`receiver_to_result` 是应用接收后的处理延迟，源 PTS 映射值仅是估计，当前 `capture_latency_measured:false`，不能据此宣称测到了曝光到 LOS 的完整延迟。

## 标定文件

从 `config/camera.example.yaml` 复制并填写**实际相机、镜头、对焦位置和采集模式的标定结果**。模板含 `calibrated: 0` 和无效参数，程序会拒绝；不能只把开关改成 1。测试里的合成内参只验证数学关系，不是设备标定。

应用不会自动生成相机标定，也不能从视频尺寸或模型文件推导实际内参。没有标定时可显式跑检测/跟踪：

```bash
./build/Aerial_detection_demo --config config/video.yaml --no-los
./build/Aerial_detection_demo --config config/camera.yaml --no-los
```

这时帧级 `calibrated:false`，所有目标 `los:null`。有真实标定时将 `--no-los` 替换为 `--calibration /path/to/real-camera.yaml`；两者互斥。相机示例默认请求 MPP MJPEG 解码，设备不支持该插件或格式时还需调整采集配置。`calibrated:true` 表示程序接受了用户确认的标定配置，不代表软件自动验证了文件与当前镜头/采集模式的物理对应关系。

| 字段 | 含义 |
|---|---|
| `schema_version` | 固定为 `1` |
| `calibrated` | 实际标定并核对后设为 `1`；其他值拒绝 |
| `image_width`, `image_height` | 标定图像尺寸，正整数 |
| `fx`, `fy`, `cx`, `cy` | 像素单位；正焦距，所有值必须有限；零 skew |
| `distortion_model` | `pinhole` 或 `fisheye`，不可混用系数 |
| `distortion_coefficients` | 显式给出有限系数数组；已确认无畸变也要明确写零数组 |
| `image_geometry` | 当前仅支持 `full_frame` |
| `resize_policy` | `reject` 或明确授权的 `scale_full_frame` |

Pinhole 支持 OpenCV 的 4、5、8、12、14 参数排列：`k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,tau_x,tau_y]]]]`。Fisheye 固定为 `k1,k2,k3,k4`，采用 `cv::fisheye::undistortPoints`。参考 [OpenCV pinhole 实现](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/undistort.dispatch.cpp) 和 [OpenCV fisheye 文档](https://docs.opencv.org/4.10.0/db/d58/group__calib3d__fisheye.html)。

默认 `resize_policy: reject` 要求输入尺寸与标定一致，不会静默猜测相机参数。`scale_full_frame` 表示使用者确认：输入是同一完整视场，只有横纵坐标缩放 `u'=sx*u, v'=sy*v`，其中 `sx=W/Wcalib, sy=H/Hcalib`。此时按各轴缩放 `fx,cx` 与 `fy,cy`，畸变系数不变；允许非等比例缩放。这一选项不表示未知分辨率相机模式自动具有相同视场，也不能自动补偿改变主点的半像素偏移。

裁剪、传感器 ROI、不同视场模式、旋转、镜像、数字防抖或额外像素偏移必须使用对应输出图像的标定，或先显式恢复到已标定图像坐标。当前没有这些任意变换配置；`image_geometry` 声明其他值会被拒绝。程序无法仅靠相同尺寸检测一幅图是否已被裁剪或旋转，因此配置者需确认几何关系。上游若已经去畸变，应提供该输出图像的新内参与显式零畸变，不能重复使用原镜头畸变参数。

## 接口与测试

```cpp
aerial::LosProjector projector("config/camera.yaml");
// 可在收到第一帧时验证实际采集尺寸；配置/尺寸错误抛 std::invalid_argument。
projector.calibration().intrinsicsFor(frame.size());
auto los = projector.project(track.bbox, frame.size());
// 发布 los.valid、方向/角度，以及 frame_id、图像时间戳、track_id、bbox、更新标记。
```

LOS 只处理每个框中心，没有为整幅图生成去畸变图，避免额外图像拷贝和全帧处理。结果可在跟踪后立即发布，不依赖画框、显示或视频编码。`tests/test_los.cpp` 检查光轴、左右上下符号、全部支持的 pinhole 系数长度及 fisheye 投影往返、完整图像缩放、非法标定与非法观测。
