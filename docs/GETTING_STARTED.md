# 入门指南：先看懂这个 RK3588 实时检测/跟踪工程

这份文档面向第一次接触本仓库的人，也适合刚开始学习 **Linux C++ / GStreamer / RKNN / 多目标跟踪** 时作为代码阅读路线。

目标不是一次性理解所有实现细节，而是先建立一个稳定的心智模型：

> **一帧图像从哪里来，经过哪些模块，最后变成什么结果。**

---

## 1. 先记住整个工程只在做一件事

当前主链路是：

```text
相机 / 视频文件
    ↓
GStreamer 解码与取帧
    ↓
YOLOv8n RKNN 检测
    ↓
BoT-SORT 多目标跟踪
    ↓
可选 LOS 角度计算
    ↓
JSONL 结果 / 可选预览或视频输出
```

代码里真正的主入口只有：

```text
src/main.cc
```

第一次阅读时，不要从所有 `.cpp` 文件逐个看。先沿着这条链路走一遍。

---

## 2. 5 分钟理解线程结构

实时视觉系统和普通“读一张图 → 推理 → 输出”程序最大的区别，是不同阶段可能并行执行。

本工程当前结构可以简化成：

```text
                  ┌─────────────────────┐
                  │   Capture Thread    │
                  │ GStreamer 读取图像   │
                  └──────────┬──────────┘
                             │
                             ▼
                    最新帧 FrameSlot
                             │
                             ▼
              ┌──────────────────────────┐
              │ Detection Worker(s)      │
              │ letterbox → RKNN → NMS   │
              └─────────────┬────────────┘
                            │
                            ▼
                    CompletionQueue
                            │
                            ▼
                  ┌──────────────────┐
                  │    Main Thread   │
                  │ BoT-SORT → LOS   │
                  └────────┬─────────┘
                           │
                 ┌─────────┴──────────┐
                 ▼                    ▼
             JSON 输出            Preview / 编码
```

实时模式的核心思想不是“每帧都必须处理”，而是：

> **结果尽量对应最新画面，不让旧帧在队列里越积越多。**

因此仓库中会看到“单槽邮箱”“丢弃过期结果”“最新帧覆盖旧帧”等设计。

---

## 3. 第一次应该按什么顺序看代码

推荐顺序：

### 第 1 步：`src/main.cc`

这里只关注主流程，不要一开始钻进具体算法。

重点找这些对象：

```cpp
AppConfig
LosProjector
JsonPublisher
BotSort
YoloV8Detector
GstCapture
FrameSlot<Frame>
CompletionQueue
```

然后看它们的创建顺序。

你会发现程序大致做了：

```text
1. 读配置
2. 创建 YOLO detector
3. 创建 tracker
4. 创建视频输入
5. 启动采集线程
6. 启动检测 worker
7. 主线程读取检测结果
8. 更新 BoT-SORT
9. 计算 LOS
10. 输出 JSON
```

如果只想快速理解系统，先把这十步读通即可。

---

### 第 2 步：`include/aerial/pipeline.hpp`

这是理解多线程数据流最重要的文件之一。

先看三个结构：

```cpp
Frame
DetectionPacket
CompletionQueue
```

其中 `Frame` 不只是图像，还带：

```text
frame id
图像时间戳
接收时间
GStreamer buffer 生命周期 owner
用于 tracking 的时间
```

这意味着后面的检测、跟踪、延迟统计都可以知道“这到底是哪一帧”。

#### `FrameSlot`

它可以理解成一个容量为 1 的邮箱。

实时模式：

```text
旧帧还没处理
    +
新帧来了
    ↓
保留新帧，丢掉旧帧
```

离线模式则会等待消费者处理，不主动丢帧。

#### `CompletionQueue`

多个 detector worker 完成顺序可能和输入顺序不同。

它负责避免：

```text
frame 12 比 frame 11 先完成
↓
跟踪器先看到 12 又看到 11
```

因为跟踪器要求时间向前走。

---

### 第 3 步：`src/capture.cpp`

这个文件负责“图像是怎么来的”。

当前使用的是：

```text
GStreamer pipeline
        ↓
appsink
        ↓
GstSample
        ↓
cv::Mat
```

不要把这里理解成普通 OpenCV `VideoCapture`。

重点理解三个概念。

#### ① appsink

GStreamer 前面的解码器负责产生图像，`appsink` 是应用程序把图像取出来的出口。

当前实时配置会限制 appsink 缓冲：

```text
max-buffers = 1
```

因此不会无限积累老图像。

#### ② `cv::Mat` 不一定拥有图像内存

代码中的 `cv::Mat` 直接指向 GStreamer 映射出来的 buffer。

所以 `Frame` 里面还有：

```cpp
std::shared_ptr<void> owner;
```

它负责保证 GStreamer 的内存在 `cv::Mat` 使用期间不会被提前释放。

这类“谁拥有内存”的问题，是 C++ 视频系统里非常重要的概念。

#### ③ PTS 和系统时间不是一回事

视频帧有自己的媒体时间戳 PTS。

程序同时还记录：

```text
received_ns
source_monotonic_ns
tracking_time_seconds
```

它们分别服务于：

```text
延迟测量
实时过期判断
跟踪器时间推进
```

---

## 4. YOLO 检测代码怎么读

入口：

```text
src/yolov8.cpp
```

配合：

```text
src/yolo_decode.cpp
include/aerial/yolov8.hpp
include/aerial/yolo_decode.hpp
```

一帧图像进入 detector 后，主要经历：

```text
BGR 原图
  ↓
letterbox resize
  ↓
BGR → RGB
  ↓
RKNN 输入
  ↓
rknn_run
  ↓
获取输出 tensor
  ↓
DFL 解码
  ↓
NMS
  ↓
Detection 列表
```

### 什么是 letterbox

YOLO 模型输入固定为例如：

```text
640 × 640
```

而原图可能是：

```text
1920 × 1080
```

直接拉伸会改变目标形状，所以通常使用：

```text
等比例缩放 + 补边
```

检测结束后还要把 box 坐标重新映射回原图。

### 什么是 RKNN

RKNN 是 Rockchip NPU 的模型格式和运行时接口。

最基础流程就是：

```cpp
rknn_init
rknn_inputs_set
rknn_run
rknn_outputs_get
```

当前实现会分别统计：

```text
preprocess
input_submit
rknn_run
output_get
postprocess
```

所以以后做性能优化时，不要只看总 FPS，要先看时间花在哪一段。

### 当前并不是“端到端零拷贝”

这是阅读代码时很容易误判的地方。

当前检测路径仍然使用普通 RKNN input API，并且 CPU 上做 letterbox / RGB 预处理。

因此不要因为代码里出现：

```text
DMA
MPP
RGA
```

就自动认为整条链路已经零拷贝。

---

## 5. BoT-SORT 怎么理解

入口：

```text
src/bot_sort.cpp
```

第一次不要试图理解所有数学细节。

先只记住一个 tracker 的目标：

```text
第 t 帧检测框
      ↓
判断哪个框属于哪个已有目标
      ↓
为目标维持稳定的 track_id
```

比如：

```text
frame 100: drone → ID 7
frame 101: drone → ID 7
frame 102: detector 漏检
frame 103: drone → 仍然恢复为 ID 7
```

代码主要包含四部分。

### ① Kalman Filter

作用：

```text
根据过去的位置和速度预测下一帧目标在哪
```

状态里包含：

```text
中心 x / y
宽 / 高
对应速度
```

### ② 数据关联

检测框来了以后，需要决定：

```text
Detection A 属于 Track 3 还是 Track 8？
```

当前主要根据 IoU / score 形成代价，然后做全局匹配。

### ③ 两阶段关联

BoT-SORT / ByteTrack 一类方法的重要思想之一是：

```text
先用高置信度检测匹配
再用低置信度检测尝试找回剩余轨迹
```

所以 `config/video.yaml` 中会看到：

```yaml
high_thresh: 0.5
low_thresh: 0.1
```

### ④ GMC

无人机相机自己可能在运动。

于是背景整体也会移动。

GMC（Global Motion Compensation）尝试估计：

```text
前一帧 → 当前帧
相机运动造成的全局图像变换
```

然后先修正轨迹预测，再进行匹配。

当前实现使用：

```text
sparseOptFlow
```

也就是角点 + LK 光流 + 仿射变换估计。

---

## 6. LOS 是什么

入口：

```text
src/los.cpp
```

LOS = Line Of Sight，视线方向。

检测/跟踪告诉你：

```text
目标在图像哪个位置
```

LOS 则进一步告诉你：

```text
目标相对相机朝哪个方向
```

基本流程：

```text
目标 bbox
   ↓
取 bbox 中心像素
   ↓
相机去畸变
   ↓
相机内参反投影
   ↓
三维单位方向向量
   ↓
azimuth / elevation
```

注意：

> **LOS 只有方向，没有距离。**

当前代码不负责单目测距、IMU 融合，也不负责制导控制。

如果没有真实相机标定，应使用：

```bash
--no-los
```

不要伪造焦距来生成“看起来合理”的角度。

---

## 7. 配置文件先只看 `config/video.yaml`

当前视频配置大致分五块：

```yaml
source:
detector:
tracker:
pipeline:
output:
```

### source

控制输入：

```text
文件 / 相机
解码方式
是否实时
FPS
```

### detector

控制 YOLO：

```text
模型路径
置信度阈值
NMS 阈值
RKNN core mask
输入模式
```

### tracker

控制 BoT-SORT：

```text
高低置信度阈值
匹配阈值
轨迹保留时间
GMC
```

### pipeline

控制系统调度：

```text
worker 数
OpenCV CPU 线程数
最大帧数
过期时间 max_age_ms
```

### output

控制：

```text
JSONL
preview
视频输出 pipeline
```

第一次运行时，不要一次改十个参数。

推荐始终：

```text
改一个参数 → 跑一次 → 看结果
```

---

## 8. 最小运行方法

### 编译

在 RK3588 Linux 环境中：

```bash
./build-linux_RK3588.sh
```

### 先跑检测 + 跟踪，不跑 LOS

```bash
./build/Aerial_detection_demo \
  --config config/video.yaml \
  --no-los \
  --max-frames 120
```

输出默认写入：

```text
results.jsonl
```

### 实时模式

```bash
./build/Aerial_detection_demo \
  --config config/video.yaml \
  --no-los \
  --realtime \
  --max-frames 900
```

运行结束后，终端会输出：

```text
received
results
dropped
expired
p50 / p95 / p99 latency
```

对于这个项目，优先关注：

```text
延迟
过期帧数量
丢帧数量
结果是否新鲜
```

不要只盯着 FPS。

---

## 9. CMakeLists.txt 怎么看

现在不需要系统学习整套 CMake。

先看懂这三个 target：

```text
aerial_core
    ├── bot_sort.cpp
    ├── los.cpp
    └── yolo_decode.cpp

aerial_io
    ├── capture.cpp
    └── output.cpp

Aerial_detection_demo
    ├── main.cc
    ├── app_config.cpp
    └── yolov8.cpp
```

关系大致是：

```text
Aerial_detection_demo
        ↓
     aerial_io
        ↓
     aerial_core
```

这样以后碰到链接错误时，你至少知道：

```text
这个 .cpp 属于哪个 target？
这个 target 有没有链接需要的库？
```

---

## 10. 想修改某个功能，应该去哪里

| 需求 | 首先看 |
|---|---|
| 改相机 / 视频输入 | `src/capture.cpp` |
| 改实时丢帧策略 | `include/aerial/pipeline.hpp` |
| 改 YOLO RKNN 推理 | `src/yolov8.cpp` |
| 改 letterbox / YOLO 解码 | `src/yolo_decode.cpp` |
| 改跟踪算法 | `src/bot_sort.cpp` |
| 改 GMC | `src/bot_sort.cpp` 中 `SparseGmc` |
| 改 LOS | `src/los.cpp` |
| 改 JSON 输出 | `src/output.cpp` |
| 改配置参数 | `src/app_config.cpp` + `config/*.yaml` |
| 改线程调度 | `src/main.cc` + `pipeline.hpp` |
| 做性能测试 | `scripts/benchmark_latency.py` |
| 查模型输入输出 | `docs/YOLOV8_MODEL.md` |

---

## 11. 第一阶段不需要学什么

刚开始不要同时学：

```text
GStreamer 全部插件体系
RKNN 所有 API
Kalman Filter 完整数学推导
Hungarian 算法证明
OpenCV 全部模块
CMake 全语法
Linux 驱动开发
```

否则很容易陷入“每个东西都没学完，所以项目也看不懂”。

第一阶段只需要做到：

```text
看到一段代码
↓
知道它在整条数据流的哪个位置
↓
知道输入是什么
↓
知道输出是什么
```

细节可以在真正需要修改这个模块时再补。

---

## 12. 推荐的实际学习顺序

### 阶段 A：只理解数据流

阅读：

```text
main.cc
pipeline.hpp
capture.cpp
```

目标：

> 能自己画出“图像从输入到输出”的流程图。

### 阶段 B：理解检测

阅读：

```text
yolov8.cpp
yolo_decode.cpp
```

目标：

> 能解释 BGR 图像怎样变成一组 bbox。

### 阶段 C：理解跟踪

阅读：

```text
bot_sort.cpp
```

先理解：

```text
prediction
association
update
lost / tracked / removed
```

不用立刻推 Kalman 公式。

### 阶段 D：理解系统优化

再研究：

```text
最新帧策略
worker 数
core mask
MPP
RGA
内存复制
P50/P95/P99
```

这时再做性能优化才有意义。

---

## 13. 建议第一次自己完成的 5 个小实验

不要直接开始“大改架构”。先做几个极小实验建立直觉。

### 实验 1：改变最大处理帧数

```bash
--max-frames 30
--max-frames 300
```

确认程序启动、停止、结果数变化。

### 实验 2：实时 / 离线切换

比较：

```bash
--offline
--realtime
```

观察：

```text
received
results
dropped
latency
```

### 实验 3：关闭 GMC

修改：

```yaml
gmc_method: none
```

确认 tracker 仍然正常工作，并比较 tracking 时间。

### 实验 4：改变 detector threshold

例如：

```yaml
confidence_threshold: 0.3
```

观察输出目标数量变化。

### 实验 5：打印一帧完整 JSON

找到：

```text
frame_id
bbox
track_id
score
LOS
latency
```

把代码里的变量和最终输出字段一一对应起来。

完成这五个实验后，再读代码会容易很多。

---

## 14. 阅读这个仓库时最容易产生的几个误区

### 误区 1：线程越多延迟越低

不一定。

多 worker 可以提高吞吐，但也可能增加：

```text
排队
内存带宽竞争
NPU 调度竞争
结果乱序
```

所以 worker 数必须实测。

### 误区 2：FPS 高就是实时性好

也不一定。

假设系统每秒处理 60 帧，但当前输出一直落后画面 500 ms，控制系统仍然很难使用。

这个项目更关注：

```text
图像 → 对应结果
```

之间的延迟。

### 误区 3：出现 DMA / MPP / RGA 就是零拷贝

不是。

是否零拷贝必须沿完整内存路径确认。

### 误区 4：tracker 可以随便跳时间

不行。

Kalman 预测和轨迹超时都依赖时间，因此输入 tracker 的 timestamp 必须严格向前。

### 误区 5：bbox 中心可以直接变成真实角度

只有完成真实相机标定，并正确处理畸变后才有物理意义。

---

## 15. 最终你应该能回答这些问题

如果下面这些问题都能自己解释，说明已经真正入门这个工程：

1. 一帧图像从哪里进入程序？
2. 为什么 `Frame` 里要保存 `owner`？
3. 实时模式为什么允许丢帧？
4. 为什么 detector 可以有多个实例，但 tracker 只有一个？
5. 为什么 tracker 需要严格递增的时间？
6. letterbox 为什么不能简单直接 resize？
7. `rknn_run` 和整个 inference 时间为什么不是一回事？
8. BoT-SORT 的高低分两阶段匹配是干什么的？
9. GMC 在无人机视频中解决什么问题？
10. LOS 为什么需要相机标定？
11. 为什么 FPS 和 latency 必须分别测？
12. 当前链路哪里还有 CPU copy / CPU preprocessing？

---

## 下一步阅读

当这份文档已经比较熟悉后，再按需要阅读：

```text
docs/ARCHITECTURE.md
    当前系统架构和旧版本差异

docs/OPTIMIZATION_RESULTS.md
    RK3588 实际性能优化结果

docs/RKNN_PROFILING.md
    NPU 性能分析

docs/YOLOV8_MODEL.md
    YOLOv8 RKNN 模型约束

docs/LOS.md
    LOS 坐标和标定约定

docs/BOTSORT_PROVENANCE.md
    BoT-SORT 实现来源和适配
```

建议顺序始终是：

> **先知道系统在做什么，再理解代码怎么做，最后才研究为什么这样优化。**
