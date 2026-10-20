MouseLimbLab v1.1.2 同步精简了软件内、安装包和 GitHub 上的使用说明，并重新打包 Mac / Windows 版本。

## 下载与启动

| 系统 | 下载文件 | 启动方式 |
|---|---|---|
| Mac M 系列，macOS 14+ | `MouseLimbLab_Mac_arm64_1.1.2.zip` | 完整解压，打开 `MouseLimbLab.app` |
| Windows 64 位 | `MouseLimbLab_Windows_x64_1.1.2.zip` | 完整解压，打开 `MouseLimbLab.exe` |
| 源码 / Intel Mac | `MouseLimbLab_Source_1.1.2.zip` | 安装 Python 3.12，运行对应启动脚本 |

桌面包已包含 Python、分析依赖和中文字体。解压后可先读“开始使用.txt”。Windows 保留整个解压文件夹；Mac 首次打开如遇系统提示，在“系统设置 → 隐私与安全”中允许打开。

## 本次更新

- **软件内说明**：简化帮助页、DeepLabCut 训练提示和结果报告，直接说明操作步骤。
- **随包文档**：新增“开始使用.txt”，同步中文指南、训练说明及测试记录。
- **分析功能**：继续支持颜料追踪、四点 / ALMA 六点方法切换、轨迹校正、周期复核与动物汇总。

Mac ARM64 和 Windows x64 均通过 33 项自动测试，以及打包程序的分析导出、中文界面、帮助页和训练页检查。具体环境及既有视频测试见[测试记录](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/blob/v1.1.2/docs/VALIDATION.md)。

[中文使用说明](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/blob/v1.1.2/docs/USER_GUIDE_ZH.md) · [DeepLabCut 训练](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/blob/v1.1.2/docs/DEEPLABCUT_ZH.md)

程序包附完整源码、许可证、校验码和构建记录。整合程序采用 GPL-3.0-only，原版脚本保留 MIT 许可。

## English

Version 1.1.2 simplifies in-app help, training guidance and report text, and ships synchronized guides with a short Chinese quick start. Mac ARM64 and Windows x64 packages passed 33 automated tests and native analysis, export and GUI checks, including the help and DeepLabCut pages.
