MouseLimbLab 将颜料追踪、运动学分析和 DeepLabCut 训练整合到一个桌面程序中，可切换“我们的四点方法”和“ALMA 六点方法”。

## 下载与启动

| 系统 | 下载文件 | 启动方式 |
|---|---|---|
| Mac M 系列，macOS 14+ | `MouseLimbLab_Mac_arm64_1.1.1.zip` | 完整解压，打开 `MouseLimbLab.app` |
| Windows 64 位 | `MouseLimbLab_Windows_x64_1.1.1.zip` | 完整解压，打开 `MouseLimbLab.exe` |
| 源码 / Intel Mac | `MouseLimbLab_Source_1.1.1.zip` | 安装 Python 3.12，运行对应启动脚本 |

桌面包已包含 Python、分析依赖和中文字体。Windows 请保留 EXE 旁的 `_internal` 文件夹。Mac 首次打开如出现系统提示，可在“系统设置 → 隐私与安全”中允许打开。

## 主要功能

- **颜料追踪**：首帧选择髋、膝、踝、足端，自动追踪整段视频。
- **方法切换**：四点运动学与 ALMA 六点运动学分别计算、分别输出。
- **结果复核**：逐帧校正轨迹、选择有效周期，再按动物汇总。
- **DeepLabCut**：六点标注、训练、评估和视频预测；使用自己的标注训练模型，训练环境通过随包脚本安装。
- **结果导出**：CSV、PNG、PDF、叠加视频及 HTML 报告。

本版统一了手机旋转视频在预览、追踪、标注和导出中的坐标方向，并完善了 Windows 中文显示。

## 测试记录与说明

Mac ARM64 和 Windows x64 均通过 **33 项自动测试**，以及打包后的程序启动、合成数据分析导出和中文界面检查。Mac 端另完成了 **8 段颜料视频、7,809 帧**的处理测试。

[中文使用说明](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/blob/main/docs/USER_GUIDE_ZH.md) · [DeepLabCut 训练](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/blob/main/docs/DEEPLABCUT_ZH.md) · [详细测试记录](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/blob/main/docs/VALIDATION.md)

程序包附完整源码、许可证、校验码和构建记录。整合程序采用 GPL-3.0-only，原版脚本保留 MIT 许可。

## English

MouseLimbLab provides paint-marker tracking, selectable four-point and ALMA six-point kinematics, track and cycle review, animal-level summaries, and optional DeepLabCut training. Native Mac ARM64 and Windows x64 packages include Python, dependencies, Chinese fonts and complete source. Both platforms passed 33 automated tests and native analysis, export and GUI checks.
