MouseLimbLab v1.1.4 新增连续后肢姿态图和速度相位分析。

- 自动导出红黑姿态叠加图：红色为摆动相，黑色为支撑相。
- 同步导出足端速度曲线、逐帧相位和周期边界，支持 PNG、矢量 PDF 和 CSV。
- 四点方法默认通过足端相对髋部的速度换向识别相位，也可选择原位移峰谷方法。
- 在软件内修改相位边界后，可重新生成图和统计结果。ALMA 沿用官方相位。
- 分析完成后窗口保持打开，可以继续下一段视频。切换视频会清除上一段的标记、导入坐标和相位修正。

## 下载

| 系统 | 文件 |
|---|---|
| Mac M 系列，macOS 14+ | `MouseLimbLab_Mac_arm64_1.1.4.zip` |
| Windows 64 位 | `MouseLimbLab_Windows_x64_1.1.4.zip` |
| 源码 / Intel Mac | `MouseLimbLab_Source_1.1.4.zip` |

完整解压后打开 `.app` 或 `.exe`。在“周期与轨迹复核”页查看新图；如需修改相位，编辑起始、摆动开始和结束帧，再点击“应用相位修正并重分析”。

[中文使用说明](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/blob/v1.1.4/docs/USER_GUIDE_ZH.md)

## English

Adds phase-coloured limb stick sequences, foot velocity/speed curves and
framewise phase exports. Four-point analysis supports velocity reversals and
editable event boundaries. ALMA retains its official events. Native tests
exercise two consecutive analyses in the same window.
