"""One display-coordinate convention across preview, tracking and DLC.

OpenCV defaults differ by version: rotation metadata must be enabled explicitly.
No source pixels are rewritten. Coordinates refer to the displayed orientation.
"""
from contextlib import contextmanager
import cv2

_capture = cv2.VideoCapture
COORDINATE_SYSTEM = 'display_orientation_v1'

def open_video(source, *args, **kwargs):
    cap = _capture(str(source), *args, **kwargs)
    if cap.isOpened():
        rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        enabled = cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        if rotation % 360 and not enabled:
            cap.release()
            raise ValueError('视频带旋转信息，但当前解码器不能应用旋转。请先将视频导出为正立画面后再分析。')
    return cap

@contextmanager
def oriented_opencv_reads():
    """Scope DLC's internal cv2 readers to the same convention in its worker."""
    previous = cv2.VideoCapture
    cv2.VideoCapture = open_video
    try:
        yield
    finally:
        cv2.VideoCapture = previous
