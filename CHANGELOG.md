# Changelog

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

Research preview: drift was found in two of eight real recordings. Review and
correct automated outputs. No validated biological six-point model is bundled.

## V1.0 — 2024-07-09

Original OpenCV script, preserved as `analyze_video.py` under its MIT license.
