#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"
if [ -d "dist/MouseLimbLab.app" ]; then
  /usr/bin/open "dist/MouseLimbLab.app"
  exit 0
fi
if [ ! -x .venv/bin/python ]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    echo 'Please install Python 3.12 from python.org, then run this launcher again.'
    read -r -p 'Press Enter to close.'
    exit 1
  fi
fi
if [ ! -f .venv/mouselimblab_ready ]; then
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/mouselimblab_ready
fi
exec .venv/bin/python launch.py
