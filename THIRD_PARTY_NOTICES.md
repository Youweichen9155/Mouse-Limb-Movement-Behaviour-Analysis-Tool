# Sources and third-party notices

## ALMA

- Authors: Almir Aljovic, Shuqing Zhao and contributors.
- Repository: https://github.com/sollan/alma
- Pinned commit: 47703fc3769064a74c44febe923d8014a0325210
- License: GNU General Public License version 3; full text in LICENSE.
- `mouselimblab/vendor/alma_upstream.py`: unmodified `Functions/KinematicsFunctions.py`.
- `audit/originals/ALMA_KinematicsFunctions.py`: identical upstream audit copy.
- `examples/alma_official_demo.csv`: upstream `Resources/Demo_Mouse_Treadmill_30cm_s_650000_filtered.csv`.
- The adapter is `mouselimblab/alma_engine.py`. It validates/imports input, splits unresolved missing blocks, calls the complete official functions, maps source frames, and adds explicit review/percentage conversion fields. It does not substitute the local gait detector.
- Paper: https://doi.org/10.1038/s42003-022-03077-6

## Project code

`tracking.py` reuses the project's frame-preserving color tracker and helpers from `analyze_video_v2.py`. `analysis.py` adapts the local kinematic definitions and fixes missing-gap handling, shared plot calculations and time metadata. The original public script remains at `analyze_video.py` under its MIT license (copyright 2024 Youwei Chen), retained in `LICENSES/LEGACY_MIT.txt`. Historical source hashes are in `audit/source_manifest.json`; private research scripts and machine paths are not distributed. Complete application source is included.

## Binary dependencies

NumPy, SciPy, pandas, matplotlib, OpenCV headless, scikit-learn, PySide6 Essentials / Qt, shiboken6 and their dependencies retain their original licenses. Both native packages include dependency licenses, build metadata and complete source.

Qt/PySide6 is dynamically loaded. LGPL and Qt exception texts are in `LICENSES/Qt`; the GPL text is in the root `LICENSE`. Exact-version upstream source locations are listed in `LICENSES/Qt/SOURCE.txt`. Additional dependency licenses are collected from wheel files. Source and rebuilding information: https://code.qt.io/cgit/pyside/pyside-setup.git/ and https://www.qt.io/licensing/open-source-lgpl-obligations .

The native packages include CPython 3.12; the exact interpreter version is recorded in BUILD_INFO.json. Its license is included in LICENSES/PYTHON.txt. No registry edits or system Python installation are required for the desktop application.

## User-interface font

The unmodified Noto Sans SC variable font is bundled so Chinese labels remain readable on systems without Chinese fonts. Source: https://github.com/google/fonts/tree/main/ofl/notosanssc . Copyright and SIL Open Font License 1.1 are retained in `LICENSES/NotoSansSC-OFL.txt`. This font retains its own license.

## Optional DeepLabCut environment

The native desktop package does not embed DeepLabCut or PyTorch. The optional installer installs DeepLabCut 3.0.2 in an isolated user environment, together with its upstream license files. The application calls its documented Python API in a separate process. No trained mouse-specific model is bundled. See https://github.com/DeepLabCut/DeepLabCut and docs/DEEPLABCUT_ZH.md.
