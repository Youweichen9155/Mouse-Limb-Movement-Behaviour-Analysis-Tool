# Validation record — v1.1.1

## Real-video testing (2026-10-20)

The Mac native application processed eight paint-marked recordings: five
treadmill and three suspended, totaling 7,809 source frames. Original files were
unchanged and every source frame number was retained.

Four recordings carried rotation metadata. Explicitly enabling orientation fixed
the mismatch between display coordinates and OpenCV's stored-pixel defaults.
One foot seed was also moved from adjacent skin to the paint centre and rerun.

Twelve frames per recording were inspected: uniform samples plus large inter-frame
movements (96 total). Six recordings showed no obvious marker switching in these
samples. Two showed drift onto fur/background and their cycles were excluded.
Review used agent-assisted visual spot checks, without a blinded framewise
reference dataset. The observations above describe the inspected frames.

Thirty cycles were initially automatically eligible. After excluding those two
recordings, 17 remained pending event review and animal-summary inclusion.
Capture rate, scale and animal identity fields remained unset where recording
metadata was unavailable.

## Automated checks

33 tests passed on Mac ARM64 and Windows x64, covering frame preservation, missing
markers, long gaps, capture/playback timing, suspended exclusions, stationary
clips, explicit cycle approval, independent engines, upstream ALMA equivalence,
annotation persistence, animal-separated splits, GUI handoff, video rotation and
Chinese progress output under Western Windows stream encodings.

The 90°/180° fixtures check decoded pixels, dimensions, tracked marker positions,
annotation images, overlay export and the scoped external DLC reader.

`Desktop CI` runs the tests, builds the native app, then launches a synthetic
analysis and GUI screenshot on Windows x64 and Mac ARM64. Packages contain
`BUILD_INFO.json` and `SMOKE_TEST.json` with the exact commit and platform.
These checks cover software execution, export and rendering. The release builds
ran on macOS 14.8.9 and Windows Server 2022.
The screenshot check uses the same application setup and bundled Noto Sans SC
font as ordinary startup, verifies Chinese/Latin/digit glyphs, and uses the
native Windows Qt backend. Mac CI uses offscreen rendering; normal Cocoa
startup was additionally checked locally.

## DeepLabCut checks and distribution

DLC 3.0.2/PyTorch was exercised on synthetic six-point images: dataset preparation,
one epoch, held-out synthetic evaluation, full video prediction and two resumed
epochs. These training checks used Mac CPU execution. Mouse-specific models
are trained and evaluated from the user’s labelled recordings.

Mac ARM64 is the native Mac target (macOS 14+); Intel Mac has source instructions
only. Mac uses ad-hoc signing without notarization. Windows has no Authenticode
certificate. Startup instructions are in the [user guide](USER_GUIDE_ZH.md).
