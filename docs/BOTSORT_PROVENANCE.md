# BoT-SORT 实现来源、配置与适配

本项目的 `src/bot_sort.cpp` 是官方 **BoT-SORT 的无 ReID 配置**的 C++/OpenCV 移植，不是把逐框贪心 IoU 匹配命名为 BoT-SORT。上游作者 Nir Aharon，论文为 [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651)。

固定源码提交：[`NirAharon/BoT-SORT@251985436d6712aaf682aaaf5f71edb4987224bd`](https://github.com/NirAharon/BoT-SORT/tree/251985436d6712aaf682aaaf5f71edb4987224bd)。移植时实际读取了该提交的下列文件；原样快照保存在 `third_party/botsort/upstream/`，MIT 许可证保存在 `third_party/botsort/LICENSE`。

| 上游文件 | 本项目对应实现 |
|---|---|
| `tracker/bot_sort.py` | `Track` 生命周期、分阶段关联、重激活、重复轨迹消除及 `BotSort::Impl::update` |
| `tracker/kalman_filter.py` | `initiate`、`predict_step`、`correct`：8 维 XYWH 状态和协方差 |
| `tracker/matching.py` | `iou`、分数融合与 `associate` 全局最小成本分配 |
| `tracker/gmc.py` | `SparseGmc::apply` 稀疏光流相机运动补偿 |
| `tracker/basetrack.py` | 轨迹 ID、状态、确认和生命周期字段 |

上游源码仅供审阅/对照，程序不加载 Python、PyTorch、FastReID、SciPy 或 LAP。编译/运行只需 C++17、OpenCV 的 core/imgproc/video/calib3d 和项目其他检测依赖。

## 第一版配置

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `high_thresh` | 0.5 | 第一阶段检测分数须严格大于此值，保留上游边界语义 |
| `low_thresh` | 0.1 | 第二阶段保留 `(low_thresh, high_thresh)` 范围的检测 |
| `new_track_thresh` | 0.6 | 未分配的高分检测初始化轨迹所需最低分数 |
| `match_thresh` | 0.8 | 第一阶段最大分配成本，成本为 `1 - IoU × detection_score` |
| 第二阶段阈值 | 0.5 | 上游固定值；使用 `1 - IoU`，只匹配仍处于 Tracked 状态的未匹配轨迹 |
| 未确认轨迹阈值 | 0.7 | 上游固定值；匹配剩余高分检测，融合分数 |
| `track_buffer_seconds` | 1.0 s | 自最后一次检测更新起保留轨迹的最长图像时间 |
| `prediction_output_seconds` | 0.2 s | 允许发布纯预测的最长时间；可设 0 仅发布当前检测更新 |
| `nominal_fps` | 30 | 卡尔曼标定时间单位：一个上游离散时间步对应 `1 / nominal_fps` 秒 |
| `gmc_method` | `sparseOptFlow` | 或 `none`；需针对目标视频测量 GMC 的收益和耗时 |
| ReID | 关闭 | 不加载外观模型，不计算外观嵌入；长遮挡/相近外观目标仍可能换 ID |
| score fusion | 开启 | 对应上游 `mot20=false`；第一阶段和未确认阶段融合检测分数 |

检测器必须把分数低至 `low_thresh` 的候选传入跟踪器，否则无法执行第二阶段低分关联。GMC 使用上游的 2 倍降采样、最多 1000 个角点、`qualityLevel=0.01`、`minDistance=1`、`blockSize=3`，PyrLK 跟踪后通过 RANSAC 的 `estimateAffinePartial2D` 估计运动。与上游 sparseOptFlow 一致，不使用检测框掩膜；全图移动物体占比过大时估计会受影响。

## 保留的算法

状态为 `[cx, cy, w, h, vx, vy, vw, vh]`，保留上游独立宽/高噪声权重：位置 `1/20`，速度 `1/160`；初始化标准差分别乘 2 和 10。Lost 状态预测前将宽/高变化速度清零，位置速度保留。GMC 对均值及完整 8×8 协方差应用上游的 `kron(I4, R)` 变换。

每帧顺序为：已确认轨迹和 Lost 轨迹联合预测 → GMC → 高分关联并重激活 → 未匹配 Tracked 轨迹与低分检测关联 → 未确认轨迹处理 → 初始化新轨迹 → Tracked/Lost 之间 IoU > 0.85 的重复清理。重复轨迹保留存在更久者，年龄相同时沿用上游保留 Lost 的选择。

上游的 `lap.lapjv(..., extend_cost=True, cost_limit=threshold)` 用本地实现的全局 Hungarian 最小成本分配替代：扩充为 `(tracks + detections)` 方阵，两侧未分配虚拟节点成本分别为 `threshold/2`、虚拟对虚拟为 0，超过阈值的真实配对禁止。优化目标与上游的有成本上限分配一致，复杂度为立方级；相同成本的最优解可能有不同 tie-break。没有使用贪心排序代替全局分配。

## 为实时 LOS 做的必要适配

1. **真实时间间隔。** `update` 的时间必须是同一路图像的时间，允许第一帧为 0，之后严格递增。`dt_frames = (current - previous) × nominal_fps`；跳帧时执行完整的上游单位预测步，最后不足一步使用实际 `F(dt)` 和 `Q × dt`。常规 `dt=1` 与上游滤波数值一致；分数步噪声缩放是显式时间适配，不能解读为上游支持不规则时间戳。`nominal_fps` 必须根据源的时间基配置。未确认轨迹沿用上游仅做 GMC、尚不预测的行为。
2. **超时按秒且关联前检查。** 包括采集长停顿后的首帧，超过 `track_buffer_seconds` 的轨迹先移除，避免上游按处理帧计数在大量丢帧时长期保留过期 ID。ID 在单个跟踪器实例中单调增加，删除后不复用。`nominal_fps × track_buffer_seconds` 限制在 10000 以下，以约束跳帧预测循环工作量。
3. **时间顺序保护。** 非有限、重复或倒退的图像时间抛出 `invalid_argument`，且在修改状态前拒绝。调用方应在并发检测结果进入跟踪器前丢弃过时结果，每个输入源使用单个、串行调用的 `BotSort` 实例。
4. **类别隔离。** 不同 `class_id` 禁止关联或重复清理。无效 bbox、类别或分数在修改跟踪状态前拒绝。
5. **浮点连续图像坐标。** 全部 bbox 为原始畸变图像的浮点 `xywh`，IoU 使用连续面积，不使用一些 `cython_bbox` 版本的整数像素 `+1` 面积约定。这是与原 Python 运行环境可能产生边界数值差异的位置。
6. **确认与预测输出。** 新轨迹只有源的第一帧立即确认；此后新轨迹须下一次高分命中才输出，使用上游已有的 `is_activated` 语义（固定提交默认返回语句没有筛掉未确认轨迹，源码中保留了筛选版本）。已确认 Lost 轨迹在 `prediction_output_seconds` 内可输出，标记 `detection_updated=false` 和 `seconds_since_update`；检测更新标记为 true。纯预测携带上次检测分数，不能理解为当前帧检测置信度。bbox 为 Kalman 估计框，LOS 使用其中心。
7. **工程健壮性。** GMC 遇到无角点、对应不足、图像尺寸改变或无有效仿射矩阵时回退单位变换；奇数尺寸用实际缩放比例还原平移。宽/高做正数下限保护，协方差更新后对称化；无无限增长的 Removed 历史。GMC 的输入必须是原始畸变帧，不是网络 letterbox 图。

这不是论文基准精度复现，也没有声明 ReID 关闭后的 ID 稳定性等同于 BoT-SORT-ReID。若有更长遮挡或明显相机运动，需要用实际场景验证 IDF1/ID switch、GMC 成功率与处理延迟；这些验证不应通过替换为简化跟踪器完成。

## 验证

`tests/test_tracker.cpp` 覆盖低分第二阶段、未确认轨迹、丢失/重激活、时间过期、预测标记、跨类隔离、非贪心全局最优配对、真实跳帧 dt、重复/倒序拒绝、重复轨迹清理、稀疏光流平移和无纹理/尺寸变化回退。

其中 `upstream_kalman_fixture` 的四组坐标由**未修改的上游 `kalman_filter.py` 实际执行生成**，C++ 输出以 `1e-4` 像素误差核对，涵盖三次更新、宽高变化及一次纯预测。可选复现命令（只需开发机 NumPy/SciPy）：

```sh
/usr/bin/python3 third_party/botsort/generate_kalman_fixture.py
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude $(pkg-config --cflags opencv4) src/bot_sort.cpp tests/test_tracker.cpp -o /tmp/aerial-test-tracker $(pkg-config --libs opencv4)
/tmp/aerial-test-tracker
```

这些测试证明实现路径及基础数值行为，不替代真实航拍场景 MOT 评价。
