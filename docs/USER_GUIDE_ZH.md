# MouseLimbLab 1.1.4 — 小鼠后肢运动分析

MouseLimbLab 支持颜料标记追踪、运动学分析、轨迹校正、周期复核和动物级汇总。在桌面界面中选择“我们的代码”或“ALMA”，分别生成结果。

第 3 个标签页提供 **DeepLabCut 六点标注、训练、评估和预测**，见 [训练使用说明](DEEPLABCUT_ZH.md)。训练环境独立安装；标注和原有分析功能可离线使用。

| 选项 | 我们的代码 | ALMA |
|---|---|---|
| 核心算法 | 项目 `analyze_video_v2.py` 的逐帧追踪与四点运动学，经审查修正 | 固定版本的官方完整运动学函数 |
| 关键点 | hip、knee、ankle、foot | toe、MTP、ankle、knee、hip、iliac crest |
| 输入 | 彩色标记视频＋首帧四点；或逐帧/DLC CSV | 六点逐帧/DLC CSV |
| 实验 | 跑台、悬空后肢运动 | 跑台、自发行走 |
| 周期检测 | 足端相对髋的前后运动与净空方向 | ALMA 自身周期检测 |
| 角度范围 | 每周期 P95−P5 | ALMA 原生最大值−最小值及其他参数 |
| 输出 | `local_日期_编号` | `alma_日期_编号`，另含官方参数表 |

每次运行使用所选引擎。导入六点数据后，可分别运行两种方法进行比较。

## 启动

- **Mac Apple Silicon**：解压 Mac 包，打开 `MouseLimbLab.app`。适用于 macOS 14 或以上的 M 系列芯片。首次打开如出现系统提示，在“系统设置→隐私与安全”中允许打开。
- **Windows 64 位离线包**：完整解压后双击 `MouseLimbLab.exe`。包内附有独立 Python 与依赖，可直接离线运行，保留整个解压文件夹即可。Windows 10/11 为目标系统；Windows 构建和启动测试见发布包中的 `SMOKE_TEST.json`。
- **源码包**：安装 Python 3.12，Mac 双击 `Start_Mac.command`，Windows 双击 `Start_Windows.bat`。源码启动器首次联网安装依赖。Intel Mac 使用源码启动，或在 Intel Mac 上运行 `build_app.py` 构建原生应用。

## 操作

1. 在“数据与分析”中选择 **我们的代码** 或 **ALMA**，选择对应实验方式。
2. 选择视频和/或 CSV。CSV 可独立分析；同时选择原视频可逐帧查看轨迹。
3. 我们的彩色标记视频流程：在首帧依次点击髋、膝、踝、足端；跑台再点击地面参考线。选择标记时会自动回到首帧。
4. ALMA 流程：导入六点 DLC CSV，或在第 3 页训练模型、预测视频后点击“使用所选预测 → ALMA”。外部 DLC CSV 需选择无侧别后缀 / 左侧 L / 右侧 R。六点数据需包含 MTP 和髂嵴。
5. 填写时间与标尺。**播放帧率**由视频读取；**实际采集帧率**需根据原始录制信息填写。例如确认无丢帧的 240 fps 慢动作以 30 fps 导出：播放填 30、采集填 240。未知时我们的引擎仍输出角度与比例，真实秒数留空。ALMA 要求实际采集帧率。
6. ALMA 跑台填写跑带速度 cm/s，用于结合像素运动估算尺度；自发行走填写实测 px/cm。空间标尺使用录像中实测的像素与长度对应关系。
7. 填写动物、组别、试次，选择结果目录，点击“开始分析”。过程在后台运行，可取消。ALMA 单段计算期间取消会在该次计算返回后生效。
8. 在“周期与轨迹复核”中查看周期；双击周期编号跳回视频。“人工纳入”填 `1` 纳入、`0` 排除，空白保留为未复核；保存后才参与动物汇总。
9. 需要修正追踪点时，选定原始帧、关节，在画面点击并保存校正，再运行一次。新结果与原结果分开保存。自动排除的周期先修正坐标，再重新分析后选择。
10. “按动物汇总”选择各结果的 `project.json`：周期先在试次内平均，再平均同一动物的试次。输出按实验类型和引擎分组，以动物为统计单位。

方向：我们的引擎中“正向/负向”指定**摆动相在图像 X 上的方向**；ALMA 中指定**动物行走方向**。默认自动判定，可结合原视频确认方向。

## 连续姿态图与速度相位

每次分析自动生成 `limb_sequence.png/pdf` 和 `phase_velocity.png/pdf`。红色为摆动相，黑色为支撑相，灰色为未分类；悬空实验标为前伸和后撤。姿态图将每帧关节依次连线，髋部横向对齐后按时间展开。长视频按行抽样显示，实际绘图帧号保存在 `limb_sequence_points.csv`。

四点方法默认选择“足端速度换向”：先计算足端相对髋部的坐标并平滑，再求逐帧速度。使用 10% 速度死区、至少连续 3 帧的方向确认，以及至少 10% 稳健位移幅度的前伸筛选。正向速度对应摆动，反向对应支撑，边界定位到确认换向前的最后一次过零。自动方向结合足端净空判定；可通过方向选项修正。完整周期按支撑开始→摆动开始→下一次支撑开始计算，片段首尾的可识别相位也显示在图上。

这些是运动学相位估计，配合原视频核对足端接触。前伸运动与相位定义可参见 [Treadmill Locomotion in the Intact and Spinal Mouse](https://pmc.ncbi.nlm.nih.gov/articles/PMC6740531/)。阈值为本软件的检测设置。

“位移峰谷（原方法）”保留旧算法；ALMA 继续使用官方事件。未填写实际采集帧率时显示 px/frame；填写后显示 px/s，同时填写 px/cm 后显示 cm/s。

在第 2 页修改起始、摆动开始和结束帧，点击“应用相位修正并重分析”，即可重新计算相位、时长、图和统计表，并保存到新目录。自动结果保留。新结果需重新确认周期纳入。

完成后窗口保持打开。点击“打开视频…”可分析下一段；新视频会清除上一段的坐标导入、标记点和相位修正。

## 输入格式

支持 DeepLabCut 的标准三层表头 CSV（单 scorer、单只动物，可选 L/R 侧）。六点别名见 `mouselimblab/io.py`。多动物 DLC 四层表头、HDF5 和 SLEAP 文件需先转换。

逐帧 CSV 必须有原始零起始帧号 `source_frame` 及以下像素列：

```text
source_frame,hip_x_px,hip_y_px,knee_x_px,knee_y_px,ankle_x_px,ankle_y_px,foot_x_px,foot_y_px
```

ALMA 另需 `mtp_x_px,mtp_y_px,crest_x_px,crest_y_px`，其中 `foot` 对应解剖学趾尖（toe）。可选 `*_detected` 或 `*_likelihood` 列用于遮挡/置信度过滤。缺失帧保留 NaN，短内部缺口插值，长缺口分段处理。帧号按原视频顺序排列且保持唯一。

旧版 `*_joint_positions.csv` 使用删帧后的编号和旧标尺。导入新版时，从原视频重新追踪以保留原始帧号。

## 输出

- `raw_tracks.csv`：导入/追踪得到的原始逐帧坐标。
- `frame_tracks.csv`：原始、插值、平滑坐标与逐帧指标；所有图使用同一计算表。
- `cycle_metrics.csv`：周期参数、源帧号、自动排除原因。
- `cycle_review.csv`：人工纳入/排除与备注。
- `normalized_curves.csv`：101 点标准化曲线。
- `clip_summary.csv`：自动生成的片段汇总；复核后的动物统计由“按动物汇总”生成。
- `ALMA_official_parameters.csv`：仅 ALMA；保留官方列名和原始值。帧号相对各连续分段，另附 `segment_source_frame_start`；主周期表则为原始视频帧号。
- `signals/trajectories/cycles.pdf/png`：矢量 PDF、600 dpi PNG 与对应源数据。
- `overlay.mp4`：可选全帧叠加视频，使用播放帧率，不跳过漏检帧。
- `phase_frames.csv`：逐帧速度、相位、来源、周期编号与周期质控状态；周期编号空白表示未构成完整周期。
- `phase_events.csv`：周期起始、摆动开始、结束源帧号。
- `limb_sequence_points.csv`：姿态图的帧号、关节与实际绘图坐标。
- `limb_sequence/phase_velocity.pdf/png`：连续姿态图及速度相位曲线。
- `project.json`、`manifest.json`：配置、来源 SHA256、引擎、版本及结果状态；失败则有 `failure.json`。
- `report.html`：可直接打开的本地结果报告。
- 动物汇总：`Prism_animal_level.csv`、`reviewed_trial_summary.csv`、`group_animal_counts.csv`。

## 拍摄与方法参数

适用固定机位、正立且固定帧率的二维侧视视频。拍摄时保持标记清晰、后肢可见；分析后通过叠加轨迹和周期表完成复核。

ALMA 集成的是 [sollan/alma](https://github.com/sollan/alma) 提交 `47703fc3769064a74c44febe923d8014a0325210` 的跑台与自发行走运动学模块，调用 `extract_parameters` / `extract_spontaneous_parameters`。两种分析引擎各自保留角度、ROM 和周期筛选定义，比较时使用对应的指标名称。

官方 `swing/stance/drag percentage (%)` 列保存 0–1 的比例；主表同时提供 `ALMA_*_fraction` 和乘 100 的 `ALMA_*_pct`。该上游版本的 `pairwise_dtw` 计算重采样后的距离；连续拖曳过滤沿用上游实现。算法细节见 [代码审查说明](CODE_AUDIT_ZH.md)，测试结果见 [测试记录](VALIDATION.md)。

1.1.1 统一使用旋转后的显示坐标。复用旧项目时，核对首帧四点、地面和外部 CSV 与当前画面一致；包含旋转视频但缺少坐标方向记录的旧六点项目需重建并核对标注。

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
