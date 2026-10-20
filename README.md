# Mouse Limb Movement Behaviour Analysis Tool — MouseLimbLab

Desktop software for mouse hindlimb tracking and kinematics, by **Youwei Chen**.
The original colour-marker tool now has a desktop interface, selectable four-point / ALMA six-point analysis, and optional DeepLabCut training.

**[Download v1.1.1](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/releases/tag/v1.1.1)** · [中文使用说明](docs/USER_GUIDE_ZH.md) · [DeepLabCut 训练](docs/DEEPLABCUT_ZH.md) · [Validation](docs/VALIDATION.md)

**Research preview:** inspect overlaid tracks and detected cycles before using measurements. Automatic coordinate coverage is not localization accuracy. Colour tracking can drift onto fur, reflections or background.

## Download and start / 下载与启动

Download the corresponding ZIP from the release page and extract it completely. Desktop packages include Python and analysis dependencies; no separate Python installation is needed for paint tracking or coordinate analysis.

| Computer | Download | Start |
|---|---|---|
| Apple Silicon Mac (M series), macOS 14+ | `MouseLimbLab_Mac_arm64_1.1.1.zip` | Open `MouseLimbLab.app` |
| Windows x64 | `MouseLimbLab_Windows_x64_1.1.1.zip` | Open `MouseLimbLab.exe` |
| Source / Intel Mac | `MouseLimbLab_Source_1.1.1.zip` | Python 3.12; see below |

Keep the entire extracted folder, including `_internal`, together. These packages are not Apple-notarized or Authenticode-signed. macOS may require allowing the application in **System Settings → Privacy & Security**. Verify the published SHA256 checksum.

中文：下载对应压缩包并完整解压，Mac 打开 `.app`，Windows 打开 `.exe`；不要只复制 Windows 的 EXE 文件。Intel Mac 当前提供源码启动方式。

## Two independent analysis methods

| | Our four-point method | ALMA |
|---|---|---|
| Landmarks | Hip, knee, ankle, foot marker | Iliac crest, hip, knee, ankle, MTP, toe |
| Input | Paint-marked video with four initial clicks, or framewise coordinates | Six-point coordinates, including DLC CSV |
| Recording | Treadmill or suspended hindlimb movement | Treadmill or spontaneous walking |
| Cycle detection | Project-specific detector | Pinned upstream ALMA functions |
| Output | Angles, ROM, normalized curves and applicable contact/clearance measures | ALMA parameters and a separate review table |

A four-point foot marker is not automatically equivalent to anatomical MTP/toe points. The ankle-angle definitions differ. The program does not fabricate six-point labels from four-point tracks.

## Paint-marker workflow

1. Open a video, choose **我们的代码 · 四点运动学**, and select treadmill or suspended recording.
2. On the first displayed frame, click the centres of the hip, knee, ankle and foot paint marks. For treadmill videos, select the ground reference line.
3. Enter capture frame rate and spatial calibration only when known. Playback and capture rates are separate; no fixed slow-motion multiplier is assumed.
4. Run analysis and inspect the overlay. Every source frame retains its number; missing values and long gaps remain visible.
5. Correct misplaced points and rerun as needed. Explicitly approve cycles before animal summaries. Cycles are not independent animals.

Outputs include framewise CSV, cycle tables, PNG/PDF figures, an HTML report and optional overlay video. Each run creates a new folder without overwriting the source video.

## Optional DeepLabCut

The third tab supports frame sampling, six-point annotation, animal-separated train/test sets, training, evaluation and prediction. DLC runs in an isolated environment installed with `Install_DeepLabCut_Mac.command` or `Install_DeepLabCut_Windows.bat`.

This optional setup needs Python 3.12 and internet access. DLC 3.0.2 is supported. No trained mouse-specific model is bundled. The connection has been exercised on synthetic labelled data; real six-point biological accuracy has not been validated. See the [training guide](docs/DEEPLABCUT_ZH.md).

## Validation and limitations

- 32 automated tests cover analysis, frame preservation, annotation, GUI handoff and 90°/180° orientation. CI builds and smoke-tests native apps.
- Local Mac testing used 8 real paint-marked videos (7,809 source frames). Across 96 inspected frames, 6 recordings showed no obvious marker switching; 2 showed drift and were excluded. This is sampled visual review, not a measured accuracy.
- After exclusions, 17 automatically eligible cycles still needed event review. No animal-level treatment comparison was claimed.
- Suspended movement does not produce treadmill stance/swing or dragging metrics. ALMA requires true capture rate and the appropriate speed/scale calibration.
- Version 1.1.1 consistently applies video rotation across preview, tracking, annotation and export. Check old seeds and external CSV coordinates against the displayed image before reuse.

## Run from source

Install Python 3.12. Double-click `Start_Mac.command` or `Start_Windows.bat`, or create a virtual environment and run:

```bash
python -m pip install -r requirements.txt
python -m mouselimblab
```

See [BUILDING.md](docs/BUILDING.md) for tests and native builds. The root `analyze_video.py` is the preserved **2024 legacy script**, not the current desktop entry point. Its [original instructions](docs/LEGACY_V1_README.md) remain available.

## License and attribution

The integrated application is distributed under **GPL-3.0-only** because it includes ALMA. The original 2024 script remains under its original [MIT license](LICENSES/LEGACY_MIT.txt), copyright Youwei Chen. Release packages include dependency licenses and complete application source.

ALMA: Aljovic, Zhao and contributors; [repository](https://github.com/sollan/alma), [paper](https://doi.org/10.1038/s42003-022-03077-6). The kinematics file is pinned to commit `47703fc3769064a74c44febe923d8014a0325210`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
