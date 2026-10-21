# Changelog

## 1.1.5 — 2026-10-21

- Draw the complete clip in a single continuous limb sequence with all valid frames.
- Select swing regions by dragging on the speed curve; adjust, delete, or clear inclusive frame intervals.
- Save manual intervals with the project and export CSV. Manual colours replace all automatic phases, and local cycle metrics are recomputed.
- Native GUI checks exercise mouse selection, saving, reloading and a second analysis without closing the window.

## 1.1.4

- Add velocity-based phase detection, red/black limb sequences and velocity plots.
- Export per-frame phases and editable cycle boundaries; reanalysis saves a new result.
- Verify consecutive GUI analyses and clear prior CSV/phase state on a new video.

## 1.1.3 — 2026-10-20

- Fixes a macOS crash when the GUI worker generates charts by giving the analysis thread an 8 MiB stack.
- Adds a full GUI video-analysis regression test, including PNG/PDF, result loading and MP4 export.
- Checks the same GUI workflow in both native packages and decodes every exported test-video frame.
- Adds an Open Results Folder button and displays the completed output path.

## 1.1.2 — 2026-10-20

- Simplifies in-app help, DeepLabCut guidance and analysis report explanations.
- Synchronizes the bundled guides and adds a short Chinese quick-start file.
- Captures help and DeepLabCut screens during native Mac/Windows checks.
- Verifies bundled instructions against the tested source before publication.

## 1.1.1 — 2026-10-20

First packaged desktop update of the original public colour-marker tool.

- Adds separate four-point and ALMA analysis, frame/cycle review and animal summaries.
- Adds optional DeepLabCut annotation, animal-separated splits, training,
  evaluation and prediction in an isolated environment.
- Preserves source frames and long gaps; separates capture/playback rates and
  prevents overwriting original videos.
- Fixes video rotation across preview, tracking, annotations and export.
- Fixes Chinese progress output with redirected Windows streams using Western system encodings.
- Adds Mac ARM64/Windows x64 packages, source, licenses, checksums and native tests.
- Bundles a licensed Chinese font and checks glyph availability in the native GUI.

Real-recording track review and model-training checks are documented in
[the test record](docs/VALIDATION.md).

## V1.0 — 2024-07-09

Original OpenCV script, preserved as `analyze_video.py` under its MIT license.
