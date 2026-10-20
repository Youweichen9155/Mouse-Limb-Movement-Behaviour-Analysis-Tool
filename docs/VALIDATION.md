# Validation record

## v1.1.4 velocity phases and continuous use (2026-10-20)

The phase fixtures check known 65% stance timing, mirrored recordings, body
translation, capture-rate scaling, missing runs, stationary signals and manual
event recalculation. Both phase plots and their source tables are checked.
The GUI test runs two analyses in one window, confirms the window remains
visible and the Run button returns, and verifies separate outputs.
ALMA event definitions remain unchanged.

The reported 926-frame recording was also used to inspect the new stick
sequence and velocity plots. This is an execution/rendering check; automated
event timings are validated against the synthetic timing fixture.

## v1.1.3 GUI export fix (2026-10-20)

Two user crash reports on Mac identified an OpenBLAS matrix inversion failure
in the analysis QThread while Matplotlib was drawing event lines. The saved
CSV tables were complete; PNG/PDF and overlay exports had not started. A source
reproduction using the same tables exited with SIGBUS on the default QThread
stack. Giving the worker an 8 MiB stack completed chart export.

The full GUI workflow was rerun on the reported 926-frame paint-marked video,
including tracking, PNG/PDF, HTML report, result-table loading and MP4 export.
The exported video was checked by decoding all frames. Original inputs and
previous run folders were preserved.

The suite now has 34 tests, including a subprocess that clicks the real GUI
analysis button on a synthetic marker video. Native Mac and Windows checks
repeat that workflow and decode all 180 output frames. Earlier CLI analysis
and static GUI checks did not exercise this background-thread export path.

## v1.1.2

v1.1.2 updates interface text and bundled instructions. The native checks also
capture the help and DeepLabCut pages. Real-video and training observations
below were recorded for v1.1.1; analysis algorithms are unchanged.

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

The v1.1.1/v1.1.2 builds passed 33 tests on Mac ARM64 and Windows x64, covering frame preservation, missing
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
