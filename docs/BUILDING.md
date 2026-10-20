# Build and test

Use Python 3.12 on the target platform. Build Windows on Windows and Mac on Mac.
DeepLabCut is optional and is not frozen into the desktop application.

```bash
python -m pip install -r requirements.txt pytest==8.4.2 pyinstaller==6.16.0
python -m pytest -q
python build_app.py
python tools/smoke_native.py --dist dist --output native-validation
python tools/package_release.py --dist dist --output release --smoke native-validation/SMOKE_TEST.json
```

On macOS, build outside synchronized folders if Finder flags interfere with Qt:
`python build_app.py --distpath /private/tmp/mouselimblab-dist`. Pass the same
dist path to the other commands.

`Desktop CI` runs on pull requests and main. It uploads platform ZIPs, checksums
and smoke evidence. Release publication is a separate step after both jobs pass;
PR workflows have read-only repository permissions and no release credentials.

Packages include native executables, complete source, dependency licenses,
instructions, optional DLC installers, source-commit and dependency metadata.
Windows users must retain the EXE and `_internal` folder together.
