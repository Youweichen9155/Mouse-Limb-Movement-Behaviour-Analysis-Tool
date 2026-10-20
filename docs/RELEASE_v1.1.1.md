## 下载与启动

- **Mac M 系列（macOS 14+）**：下载 `MouseLimbLab_Mac_arm64_1.1.1.zip`，完整解压，打开 `MouseLimbLab.app`。
- **Windows 64 位**：下载 `MouseLimbLab_Windows_x64_1.1.1.zip`，完整解压，打开 `MouseLimbLab.exe`，保留旁边的 `_internal` 文件夹。
- 桌面分析程序已包含 Python 和运行依赖，无需另行安装。Intel Mac 请使用源码包和 Python 3.12。
- Mac 程序采用临时签名，尚未经过 Apple 公证；如被拦截，在“系统设置 → 隐私与安全”中允许打开。Windows 程序尚无 Authenticode 签名。

## 本版内容

- 颜料标记视频首帧四点选择、自动追踪、逐帧校正、周期复核和动物级汇总。
- 可切换“我们的四点方法”和“ALMA 六点方法”，分别生成结果。
- DeepLabCut 六点标注、训练、评估和预测入口；训练环境独立安装，不附带已验证的小鼠模型。
- 修正手机视频旋转信息造成的预览、追踪和导出坐标不一致。
- 导出 CSV、PNG、PDF、叠加视频及 HTML 报告；原视频不修改。
- 附带中文说明、中文字体、完整源码、许可证、校验码和构建记录。

## 验证与使用范围

**科研试用版（research preview）。** Mac ARM64 和 Windows x64 均通过 33 项自动测试、原生构建、程序启动、合成数据分析导出及界面截图检查。

Mac 已处理 8 段真实颜料视频、7,809 帧。抽查 96 帧时，6 段未见明显串点，2 段出现漂移并排除；这不是独立人工真值准确率验证。自动轨迹和周期必须复核后再用于统计。Windows 的自动检查不代表已完成真实实验数据的人工验收。DeepLabCut 尚无真实六点数据的准确率验证。

每个程序包内的 `BUILD_INFO.json` / `SMOKE_TEST.json` 记录构建提交与检查结果。私人实验视频不随版本公开。

## English

MouseLimbLab is now a desktop application with independent project-specific four-point and upstream ALMA six-point engines, optional isolated DeepLabCut integration, frame-preserving tracking, explicit cycle review and orientation-aware video handling. Native packages include complete application source and dependency licenses. Both platform builds execute the 33-test suite and synthetic native-analysis/GUI smoke checks. This research preview requires visual review; no biological accuracy certification or trained mouse-specific model is supplied.

The original 2024 script retains its MIT license. The integrated ALMA-based application is GPL-3.0-only. See the included notices and validation documentation.
