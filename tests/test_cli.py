"""Exercise the CLI under the encoding used by redirected Windows processes."""
import os
import subprocess
import sys


def test_chinese_progress_with_western_redirected_encoding():
    code = """
from mouselimblab.cli import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
print('读取坐标并保留源帧号…')
"""
    result = subprocess.run([sys.executable, '-c', code],
                            env={**os.environ, 'PYTHONIOENCODING': 'cp1252'},
                            capture_output=True, timeout=60, check=True)
    assert '读取坐标并保留源帧号…' in result.stdout.decode('utf-8')
