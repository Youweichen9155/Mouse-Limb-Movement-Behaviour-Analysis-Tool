# MouseLimbLab 1.1.1 — 小鼠后肢运动分析

1.1.1 修复手机视频旋转信息在不同 OpenCV 版本中读取不一致的问题。预览、颜料追踪、标注和叠加导出统一使用旋转后的显示坐标。旧版项目若曾在旋转前画面选点，升级后需重新核对首帧四点和地面；外部坐标 CSV 也须与当前画面方向一致。旧六点训练项目若包含旋转视频但未记录坐标方向，将要求重建并核对标注。

新增 **DeepLabCut 六点标注、训练、评估和预测**，见 [训练使用说明](DEEPLABCUT_ZH.md)。入口位于第 3 个标签页。训练环境独立安装；标注和原有分析功能可离线使用。

在同一个桌面界面中切换两个**独立**分析引擎。每次运行生成新目录，不修改视频或输入 CSV。

| 选项 | 我们的代码 | ALMA |
|---|---|---|
| 核心算法 | 项目 `analyze_video_v2.py` 的逐帧追踪与四点运动学，经审查修正 | 固定版本的官方完整运动学函数 |
| 关键点 | hip、knee、ankle、foot | toe、MTP、ankle、knee、hip、iliac crest |
| 输入 | 彩色标记视频＋首帧四点；或逐帧/DLC CSV | 六点逐帧/DLC CSV |
| 实验 | 跑台、悬空后肢运动 | 跑台、自发行走 |
| 周期检测 | 足端相对髋的前后运动与净空方向 | ALMA 自身周期检测 |
| 角度范围 | 每周期 P95−P5 | ALMA 原生最大值−最小值及其他参数 |
| 输出 | `local_日期_编号` | `alma_日期_编号`，另含官方参数表 |

两种引擎不会在一次运行中混合计算。导入六点数据后，也可分别运行两次，保留两套结果。

## 启动

- **Mac Apple Silicon**：解压 Mac 包，打开 `MouseLimbLab.app`。此构建目标为 macOS 14 或以上的 M 系列芯片。程序尚未经过 Apple 公证；首次被系统拦截时，在“系统设置→隐私与安全”中允许打开。
- **Windows 64 位离线包**：完整解压后双击 `MouseLimbLab.exe`。包内附有独立 Python 与依赖，不需要预装 Python，也不需联网安装。必须保留整个文件夹。Windows 10/11 为目标系统；Windows 构建和启动测试见发布包中的 `SMOKE_TEST.json`。
- **源码包**：安装 Python 3.12，Mac 双击 `Start_Mac.command`，Windows 双击 `Start_Windows.bat`。源码启动器首次联网安装依赖。Intel Mac 使用源码启动，或在 Intel Mac 上运行 `build_app.py` 构建原生应用。

## 操作

1. 在“数据与分析”中选择 **我们的代码** 或 **ALMA**，选择对应实验方式。
2. 选择视频和/或 CSV。仅有 CSV 也能分析，但无法在原视频上逐帧复核。
3. 我们的彩色标记视频流程：在首帧依次点击髋、膝、踝、足端；跑台再点击地面参考线。选择标记时会自动回到首帧。
4. ALMA 流程：导入六点 DLC CSV，或在第 3 页训练模型、预测视频后点击“使用所选预测 → ALMA”。外部 DLC CSV 需选择无侧别后缀 / 左侧 L / 右侧 R。没有 MTP 或髂嵴时会明确拒绝运行，不会从四点伪造六点。
5. 填写时间与标尺。**播放帧率**由视频读取；**实际采集帧率**需根据原始录制信息填写。例如确认无丢帧的 240 fps 慢动作以 30 fps 导出：播放填 30、采集填 240。未知时我们的引擎仍输出角度与比例，真实秒数留空。ALMA 要求实际采集帧率。
6. ALMA 跑台必须填写跑带速度 cm/s，其官方函数用跑带速度与像素运动估算尺度；此模式不使用 px/cm 输入。ALMA 自发行走必须填写实测 px/cm。不要使用未测量的默认肢长作为物理标尺。
7. 填写动物、组别、试次，选择结果目录，点击“开始分析”。过程在后台运行，可取消。ALMA 单段计算期间取消会在该次计算返回后生效。
8. 在“周期与轨迹复核”中查看周期；双击一行跳回视频。`manual_include` 填 `1` 纳入、`0` 排除，空白保留为未复核；保存后才参与动物汇总。
9. 需要修正追踪点时，选定原始帧、关节，在画面点击并保存校正，再运行一次。新结果与原结果分开保存。自动排除周期不能直接强行纳入，应先修正坐标并重分析。
10. “按动物汇总”选择各结果的 `project.json`：周期先在试次内平均，再平均同一动物的试次。输出保留实验类型和引擎分组，不把周期数当作动物 N。

方向：我们的引擎中“正向/负向”指定**摆动相在图像 X 上的方向**；ALMA 中指定**动物行走方向**。默认自动判定，必须结合原视频复核。

## 输入格式

支持 DeepLabCut 的标准三层表头 CSV（单 scorer、单只动物，可选 L/R 侧）。六点别名见 `mouselimblab/io.py`。多动物 DLC 四层表头、HDF5 和 SLEAP 文件需先转换。

逐帧 CSV 必须有原始零起始帧号 `source_frame` 及以下像素列：

```text
source_frame,hip_x_px,hip_y_px,knee_x_px,knee_y_px,ankle_x_px,ankle_y_px,foot_x_px,foot_y_px
```

ALMA 另需 `mtp_x_px,mtp_y_px,crest_x_px,crest_y_px`，其中 `foot` 必须对应真正的 toe。可选 `*_detected` 或 `*_likelihood` 列用于遮挡/置信度过滤。缺失帧保留 NaN，短内部缺口才插值；长缺口分段，不会拼接时间轴。乱序或重复帧号会报错。

旧版 `*_joint_positions.csv` 的 `Frame` 为删帧后编号且使用未确认的 cm 标尺，不能恢复原始时间，不直接导入；请从原视频重追踪。

## 输出

- `raw_tracks.csv`：导入/追踪得到的原始逐帧坐标。
- `frame_tracks.csv`：原始、插值、平滑坐标与逐帧指标；所有图使用同一计算表。
- `cycle_metrics.csv`：周期参数、源帧号、自动排除原因。
- `cycle_review.csv`：人工纳入/排除与备注。
- `normalized_curves.csv`：101 点标准化曲线。
- `clip_summary.csv`：探索性片段汇总，默认尚未人工复核。
- `ALMA_official_parameters.csv`：仅 ALMA；保留官方列名和原始值。帧号相对各连续分段，另附 `segment_source_frame_start`；主周期表则为原始视频帧号。
- `signals/trajectories/cycles.pdf/png`：矢量 PDF、600 dpi PNG 与对应源数据。
- `overlay.mp4`：可选全帧叠加视频，使用播放帧率，不跳过漏检帧。
- `project.json`、`manifest.json`：配置、来源 SHA256、引擎、版本及结果状态；失败则有 `failure.json`。
- `report.html`：可直接打开的本地结果报告。
- 动物汇总：`Prism_animal_level.csv`、`reviewed_trial_summary.csv`、`group_animal_counts.csv`。

## 方法边界

仅针对正立、固定帧率的二维侧视视频。没有多机位三维重建、相机抖动校正或透视校正。彩色标记、遮挡、左右重叠、错误机位均需人工复核。

ALMA 使用 [sollan/alma](https://github.com/sollan/alma) 提交 `47703fc3769064a74c44febe923d8014a0325210` 的 `extract_parameters` / `extract_spontaneous_parameters`。它是当前官方 v2 实现，不能称为对 2022 论文历史版本的逐字复现。本包未移植 ALMA 的疾病分类/PCA、横梯误踏模块。DeepLabCut 训练环境独立安装；没有内置已验证的通用小鼠六点模型。

官方列名 `swing/stance/drag percentage (%)` 的值实际为 0–1；为保持可核对性原表不改名，主表新增 `ALMA_*_fraction` 和乘 100 的 `ALMA_*_pct`。当前上游 `pairwise_dtw` 已改为重采样后的距离，不是真正 DTW。上游连续拖曳过滤的末尾帧行为原样保留，见代码审查说明。

现有无标记试验模型跨录像迁移未通过验证，未作为产品默认追踪器。公开包不含私人研究脚本或实验录像。两引擎的踝角定义、ROM 定义与筛选条件不同，不能直接合并为同一指标。

## 批量与重新构建

```bash
python -m mouselimblab run project.json --output results
python -m mouselimblab batch project1.json project2.json --output results
python -m mouselimblab aggregate result1 result2 --output new_animal_summary
python -m mouselimblab demo --output demo_results
python -m pytest -q
python -m pip install pyinstaller==6.16.0
python build_app.py
```

项目 JSON 内可使用相对于 JSON 的输入路径。Windows 打包脚本为 `Build_Windows.bat`。仓库的 `.github/workflows/desktop.yml` 在 Mac ARM64 和 Windows x64 上运行测试、原生构建及程序启动检查；通过后发布相应压缩包。构建方法见 [BUILDING.md](BUILDING.md)。

## 引用与许可

Aljovic A., Zhao S. et al. *A deep learning-based toolbox for Automated Limb Motion Analysis (ALMA) in murine models of neurological disorders*. Communications Biology 5, 131 (2022). [DOI](https://doi.org/10.1038/s42003-022-03077-6).

由于包含 GPL-3.0 ALMA 代码，本整合源码及派生部分按 GPL-3.0 分发；完整源码与许可证随包提供。原版 `analyze_video.py` 保留 MIT 许可，第三方库保留各自许可，见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
