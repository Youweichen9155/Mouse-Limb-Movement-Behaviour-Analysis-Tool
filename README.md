# Mouse Limb Movement Behaviour Analysis Tool — MouseLimbLab

Desktop software for mouse hindlimb tracking and kinematics, by **Youwei Chen**.
Track paint markers, switch between four-point and ALMA six-point analysis, review tracks and cycles, and export animal-level summaries. DeepLabCut annotation and training are available in the same interface.

**[Download v1.1.4](https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool/releases/tag/v1.1.4)** · [中文使用说明](docs/USER_GUIDE_ZH.md) · [DeepLabCut 训练](docs/DEEPLABCUT_ZH.md) · [Test record](docs/VALIDATION.md)

## Download and start / 下载与启动

Download the corresponding ZIP and extract it completely. Desktop packages include Python, analysis dependencies and Chinese fonts.

| Computer | Download | Start |
|---|---|---|
| Apple Silicon Mac (M series), macOS 14+ | `MouseLimbLab_Mac_arm64_1.1.4.zip` | Open `MouseLimbLab.app` |
| Windows x64 | `MouseLimbLab_Windows_x64_1.1.4.zip` | Open `MouseLimbLab.exe` |
| Source / Intel Mac | `MouseLimbLab_Source_1.1.4.zip` | Python 3.12; see below |

On Windows, keep the EXE and `_internal` folder together. If macOS prompts on first launch, allow the application in **System Settings → Privacy & Security**. Checksums are included on the release page.

中文：下载对应压缩包并完整解压，Mac 打开 `.app`，Windows 打开 `.exe`。先读包内“开始使用.txt”即可开始；Intel Mac 使用源码启动方式。

## Two analysis methods

| | Our four-point method | ALMA |
|---|---|---|
| Landmarks | Hip, knee, ankle, foot marker | Iliac crest, hip, knee, ankle, MTP, toe |
| Input | Paint-marked video with four initial clicks, or framewise coordinates | Six-point coordinates, including DLC CSV |
| Recording | Treadmill or suspended hindlimb movement | Treadmill or spontaneous walking |
| Cycle detection | Foot velocity reversals; original position method selectable | Pinned upstream ALMA functions |
| Output | Angles, ROM, normalized curves and applicable contact/clearance measures | ALMA parameters and a separate review table |

Each method produces its own result folder. For ALMA, label the anatomical MTP and toe as separate points. The [user guide](docs/USER_GUIDE_ZH.md) explains each method's angle definitions and settings.

## Paint-marker workflow

1. Open a video, choose **我们的代码 · 四点运动学**, and select treadmill or suspended recording.
2. On the first displayed frame, click the centres of the hip, knee, ankle and foot paint marks. For treadmill videos, select the ground reference line.
3. Enter the capture frame rate and spatial calibration from your recording setup. Playback and capture rates are recorded separately.
4. Run analysis and inspect the overlay. Correct misplaced points and rerun as needed.
5. Review and select cycles, then generate summaries by animal.

Outputs include framewise CSV, cycle tables, PNG/PDF figures, an HTML report and optional overlay video. Source frame numbers and missing-data flags are preserved. Each run saves results in a new folder.

## Limb sequences and gait phases

Analysis exports a red/black stick sequence (swing/stance), signed forward velocity
and speed curves, and framewise phase/event tables. The four-point method detects
sustained foot velocity reversals relative to the hip. Edit event frames in the
review table and apply them to generate a new result folder. ALMA retains its
official events. The window stays open after every analysis.

## DeepLabCut training

The third tab supports frame sampling, six-point annotation, animal-separated train/test sets, training, evaluation and prediction. Train a model from your labelled recordings and send its predicted coordinates to either analysis method.

Install the separate training environment with `Install_DeepLabCut_Mac.command` or `Install_DeepLabCut_Windows.bat`. Setup uses Python 3.12, internet access and DLC 3.0.2. See the [training guide](docs/DEEPLABCUT_ZH.md).

## Testing

The suite contains **40 automated tests**, plus native startup, synthetic analysis/export and Chinese GUI checks on Mac ARM64 and Windows x64. Local Mac testing also processed **8 paint-marked recordings containing 7,809 source frames**.

The [test record](docs/VALIDATION.md) documents the sampled track review, cycle selection, DLC checks and tested environments. Version 1.1.4 adds phase-coloured limb sequences, velocity curves and editable phase boundaries. Native checks run two consecutive analyses in the same window and decode the overlay video.

## Run from source

Install Python 3.12. Double-click `Start_Mac.command` or `Start_Windows.bat`, or create a virtual environment and run:

```bash
python -m pip install -r requirements.txt
python -m mouselimblab
```

See [BUILDING.md](docs/BUILDING.md) for tests and native builds. The original 2024 script remains at `analyze_video.py`, with its [original instructions](docs/LEGACY_V1_README.md).

## License and attribution

The integrated application is distributed under **GPL-3.0-only** because it includes ALMA. The original 2024 script retains its [MIT license](LICENSES/LEGACY_MIT.txt), copyright Youwei Chen. Release packages include dependency licenses and complete application source.

ALMA: Aljovic, Zhao and contributors; [repository](https://github.com/sollan/alma), [paper](https://doi.org/10.1038/s42003-022-03077-6). The kinematics file is pinned to commit `47703fc3769064a74c44febe923d8014a0325210`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
