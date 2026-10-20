MouseLimbLab v1.1.3 修复了 Mac 在分析后生成图片时自动退出的问题。

## 本次修复

- 修复后台绘图时的线程栈不足，恢复 PNG、PDF、报告和叠加视频导出。
- 新增“打开结果文件夹”按钮，分析完成后显示保存位置。
- 补充完整界面测试：从点击“开始分析”，到轨迹处理、图片和视频导出，再到结果加载。

已用用户反馈的 926 帧颜料视频完成全流程，逐帧确认导出视频可读取。Mac 和 Windows 均通过 34 项测试及打包程序的界面分析、图片导出和视频解码检查。

## 下载与使用

| 系统 | 文件 |
|---|---|
| Mac M 系列，macOS 14+ | `MouseLimbLab_Mac_arm64_1.1.3.zip` |
| Windows 64 位 | `MouseLimbLab_Windows_x64_1.1.3.zip` |
| 源码 / Intel Mac | `MouseLimbLab_Source_1.1.3.zip` |

完整解压后打开 `.app` 或 `.exe`。如上次分析中途退出，可在新版中打开该结果目录内的 `project.json`，直接重新分析，沿用已选标记点与参数。导出视频时勾选“导出叠加轨迹视频”。

[中文使用说明](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/blob/v1.1.3/docs/USER_GUIDE_ZH.md) · [测试记录](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/blob/v1.1.3/docs/VALIDATION.md)

## English

Fixes a macOS crash during chart export from the GUI analysis worker. The release adds an Open Results Folder button and tests the complete GUI video-analysis workflow, including PNG/PDF export, MP4 frame decoding and loading completed results. Both platforms passed 34 tests and native package checks.
