#!/bin/bash
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys;sys.exit(not ((3,10)<=sys.version_info[:2]<=(3,12)))'; then
        "$candidate" install_deeplabcut.py
        result=$?
        read -r -p 'Press Return to close. '
        exit "$result"
    fi
done
echo 'Please install Python 3.12 from https://www.python.org/downloads/ then run this installer again.'
read -r -p 'Press Return to close. '
exit 1
