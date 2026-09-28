# RK3588 本机 MPP/RGA 输入排查记录

记录日期：2026-09-24。以下为当前开发板的实测与诊断，不能当成所有 RK3588 镜像的性能或权限保证。没有更改设备权限、全局库或删除 GStreamer registry。

## 插件加载崩溃的根因

本机 `/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstrockchipmpp.so` 存在，`ldd` 无缺失依赖，但普通用户扫描插件会崩溃并可能将插件记入 registry 黑名单。更换 registry 本身不能解决根因。

实际取证：

- 当前用户 `topeet` 已属于 `video` 组，`/dev/mpp_service` 和 `/dev/rga` 为 `root:video 0660`，可以访问。
- `/dev/dma_heap/{system,system-dma32,system-uncached,system-uncached-dma32,cma,cma-uncached}` 全部为 `root:root 0600`。
- `strace -e openat,ioctl` 明确显示 MPP 打开这些 DMA heap 返回 `EACCES`；随后打开 `/dev/mpp_service` 和设备 ioctl 均成功。
- 用 `gst-inspect-1.0 --gst-disable-segtrap --gst-disable-registry-fork` 在 gdb 中定位到插件注册阶段的**编码器能力探测**，不是视频解码时才崩溃：

```text
h264e_vepu_stream_amend_deinit(ctx=0)
hal_h264e_vepu580_deinit
hal_h264e_vepu580_init
mpp_enc_hal_init
mpp_enc_init_v2
Mpp::init(MPP_CTX_ENC, MPP_VIDEO_CodingAVC)
mpp_init
gst_mpp_enc_supported
gst_mpp_h264_enc_register
plugin_init
```

缺少分配权限导致初始化失败，当前 MPP 版本的错误清理路径又解引用空指针。因此即使应用只需要 `mppvideodec`，整个插件扫描也可能因编码器探测而失败。

仅对诊断进程使用 root 权限、独立临时 registry，原插件便成功加载六个元素：`mppvideodec`、`mppjpegdec`、`mpph264enc`、`mpph265enc`、`mppvp8enc`、`mppjpegenc`。没有替换 MPP/RGA 动态库。

```sh
sudo -n env GST_REGISTRY=/tmp/aerial-mpp-root-full.bin \
  gst-inspect-1.0 mppvideodec
```

生产环境应由板卡管理员根据设备策略配置 DMA heap 的 udev 组权限，例如给授权的 `video` 组读写权限，并确认进程实际组身份；本轮**未执行**此权限修复。不要用全设备 `chmod 777` 掩盖问题。修复权限后应使用新的、当前用户可写的 `GST_REGISTRY` 重新扫描，避免继续沿用此前的黑名单缓存。

本机还缺少 `h264parse`；已安装 Ubuntu 的 `gstreamer1.0-plugins-bad` 及其依赖后完成下述验证。

## 实际版本信息

| 项目 | 本机记录 |
|---|---|
| 内核 | aarch64 Linux 5.10.198 |
| GStreamer core/tools | 1.18.5 |
| 部分 GStreamer base/good 库包 | 1.16.3，供应商镜像混合包版本 |
| Rockchip 插件 | `gst-inspect` 显示 1.14.4；包 `gstreamer1.0-rockchip1 1.14-4` |
| MPP | 包 `librockchip-mpp1 1.5.0-1`；二进制内嵌提交 `ed377c99`，日期 2023-12-14 |
| RGA | 包 `librga2 2.2.0-1`；运行日志报告 `rga_api version 1.10.1_[0]` |
| 系统 FFmpeg | 4.2.7-0ubuntu0.1，未编入 `h264_rkmpp` 解码器 |

包版本、文件日期和二进制内嵌版本不完全对应，不应据包名推断实际硬件 API 版本。现有 FFmpeg 没有可直接使用的 RKMPP 解码替代，未自行构建或替换系统 FFmpeg。

## 直接 BGR 的硬解通路已验证

`mppvideodec` 和 `mppjpegdec` 的 `format` 属性均支持 `BGR`（枚举 16），同时支持 RGB、RGBA 等格式。H.264 文件已实际跑通：

```text
filesrc location=1080p60hz.mov ! qtdemux ! h264parse
  ! mppvideodec format=BGR
  ! video/x-raw,format=BGR ! appsink name=frames
```

协商结果为 `1920×1080 BGR 60/1`，运行时出现 RGA API 日志。此路径不需要 CPU `videoconvert`。一段复用本项目 `GstCapture` 的短程序取得连续 10 帧，PTS 从 0 到 150 ms；一次运行首帧约 49 ms，第 10 帧约 101 ms，关闭约 109 ms。该记录只证明功能及交付路径可用，包含启动开销且样本少，**不是最终延迟基准或 FPS 声明**。

MJPEG 相机可使用 `mppjpegdec format=BGR`，但本次只核实了属性，没有把文件硬解结果冒充实际相机验证。

## 映射缓冲不代表 CPU 访问最快

MPP BGR 通过 GStreamer 映射后可直接构造 `cv::Mat`。但在当前板卡上，CPU 逐像素读取该缓冲明显慢于普通 OpenCV 分配的内存。行为与未缓存 DMA 映射相符；没有通过页属性证明所有运行条件下的缓存策略。

对一张真实 1920×1080 帧，`cv::setNumThreads(1)`，一次短测：

| 操作 | 直接读取 MPP 映射 | 读取 `clone()` 后内存 |
|---|---:|---:|
| BGR → 灰度 | 38.12 ms | 3.54 ms |
| resize → 640×360 | 53.56 ms | 3.60 ms |
| 完整 BGR clone | 20.12 ms | — |

因此当前 CPU letterbox 和 sparseOptFlow GMC 共同读取原始图像时，复制一次、共享普通内存可能比保留映射更低延迟。项目已增加可选的 `--copy-input` / `--no-copy-input`（配置 `source.copy_to_cpu`）：只在 worker 已选中要处理的帧之后复制，避免为即将丢弃的旧帧付出成本，普通内存帧继续供检测与跟踪共享；复制耗时计入预处理和总延迟。该判断来自访问成本实测，不追求“零拷贝”名称。

另一条可能的后续路径是保留 DMA-BUF 并用 RGA 生成网络输入和 GMC 缩略灰度图，但需要额外验证 stride、所有权、同步和颜色变换。本记录不声称已经实现这条路径。

正式并发选择与端到端结果应以 `docs/VALIDATION.md` 的完整运行记录为准；硬解速度、CPU 读取时间、接收后结果时间、源 PTS 推算时间和真实曝光到结果时间是不同指标。
