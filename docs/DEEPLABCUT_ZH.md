# DeepLabCut 六点训练与 ALMA 分析

在 MouseLimbLab 第 3 页完成六点标注、训练、评估和预测，再将预测坐标送入“我们的代码”或“ALMA”分析。

## 第一次安装

桌面程序包含标注和训练控制界面，DeepLabCut / PyTorch 通过随包脚本安装到独立环境。

1. 若电脑没有 Python，请从 https://www.python.org/downloads/ 安装 Python 3.12。Windows 安装时保留 Python Launcher，并勾选添加到 PATH。
2. 完整解压程序包，双击 `Install_DeepLabCut_Mac.command` 或 `Install_DeepLabCut_Windows.bat`。首次需要联网下载依赖。环境保存在个人用户目录 `MouseLimbLab_DLC`。
3. 打开程序第 3 页“DeepLabCut 训练”，选择环境内的 `bin/python`（Mac）或 `Scripts/python.exe`（Windows），点击“检查环境”。程序会记住手工选择的路径。

固定 DLC 3.0.2、NumPy 1.26.4、OpenCV 4.11.0.86。任务记录实际 Python / DLC / PyTorch 版本及设备，安装脚本另存完整依赖清单。

Mac M 系列可选择 Apple GPU 或 CPU。Windows 可选择 CPU；NVIDIA GPU 需要匹配显卡驱动的 PyTorch。安装脚本支持 `--cuda-index https://download.pytorch.org/whl/cu126` 等官方索引；请按 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 确认，环境检查显示 CUDA 可用后再选 NVIDIA。各平台已完成的检查见 [测试记录](VALIDATION.md)。

## 标注数据

1. 添加相同侧视拍摄方案下多只动物、不同日期和姿势的视频。预先截取连续片段，项目引用原视频路径。不同目录下的同名视频需给副本改为唯一名称。
2. 填写动物编号，并将一部分完整动物留作测试。训练集和测试集按动物分开。
3. 默认每段均匀抽取 20 帧，可按需要追加。创建项目后可选中视频行，追加指定原始帧号，覆盖拖曳、遮挡、极端屈曲、光照变化等困难情形。
4. 依次点击：髂嵴 → 髋 → 膝 → 踝 → 跖趾关节 → 趾尖。看不清的点选“不可见”，保存为缺失值。
5. 点击“确认本帧完成并下一帧”。只有完成帧进入训练。每次点击自动保存，可随时打开项目继续标注。

按统一的解剖点定义标注，分别定位趾尖和 MTP。已创建项目可修改分组、补充同一视频的帧；加入其他视频需创建新项目。

## 训练、评估、预测

1. 选择 ResNet-50 或 ResNet-101、设备、批大小和本次轮数。默认 200 epochs、批大小 4，可根据测试误差和轨迹表现调整。
2. “准备训练数据”生成标准 DLC 配置、HDF5/CSV 标注、按动物划分的显式索引、图像哈希及标注来源。每次生成新目录，保留旧数据集和模型。
3. “开始训练”会在独立进程运行并显示日志。首次运行可下载通用预训练骨干，视频与标注在本地处理。
4. 可停止任务，已保存快照保留。继续时勾选继续训练并填写追加轮数；例如已训练 100 轮，再填写 50 轮会训练到 150 轮。续训从最近保存的模型权重开始；上游默认快照不包含优化器状态。
5. “评估模型”显示测试结果。结合各点测试误差、置信度和独立视频的叠加轨迹判断模型表现。
6. “预测视频”支持多段，输出新目录中的标准六点坐标及 likelihood。输出帧数、零起始帧号和列名会与原视频核对。
7. 选中预测，点击“→ ALMA”或“→ 我们的代码”。预览自动叠加六点骨架，可以逐帧检查。填写此录像的实际采集帧率、跑台速度或标尺后再分析。新视频的这些参数会清空，防止沿用上次设置。

标注或分组更新后，重新准备训练数据。后续分析按所选引擎计算对应的指标和周期。

## 实现与测试

论文使用 DLC 2.3.9、TensorFlow/ResNet-152、Imgaug、700,000 次迭代和 90/10 划分。本程序采用 DLC 3.0.2 的 PyTorch、ResNet-50/101、Albumentations，按动物划分，比例由实际标注帧数决定。论文中的置信度 0.06、拖曳连续 10 帧可在分析页设置，再结合本地轨迹表现调整。

已用合成图像和已知六点坐标完成数据准备、1 轮训练、独立合成样本评估和整段预测。小鼠专用模型使用自己的标注训练，并按上面的评估流程检查。

参考：[DLC 安装](https://deeplabcut.github.io/DeepLabCut/docs/installation.html)、[官方指南](https://deeplabcut.github.io/DeepLabCut/docs/main-workflows/user-guide.html)、[2026 年论文](https://doi.org/10.1038/s42003-026-11110-1)。
