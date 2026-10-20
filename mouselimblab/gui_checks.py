"""Exercise the same background analysis and completion path as a button click."""
from pathlib import Path
import json
import sys
import traceback
from PySide6.QtCore import QTimer
from .gui import Window, create_application


def run_gui_analysis(spec, output, screenshot):
    app=create_application([])
    window=Window()
    window.apply_spec(spec)
    window.out.setText(str(output))
    failures=[]
    previous_hook=sys.excepthook

    def record_error(message):
        failures.append(str(message))
        print(message,flush=True)

    def exception_hook(kind,value,tb):
        record_error(''.join(traceback.format_exception(kind,value,tb)))

    # Failed automated checks must exit with diagnostics, not wait on a dialog.
    window.error=record_error
    sys.excepthook=exception_hook
    result_code=1

    def check():
        nonlocal result_code
        if window.worker is not None and window.worker.isRunning():
            return
        if not failures and window.result is None:
            return
        timer.stop()
        try:
            if failures:
                return
            folder=window.result
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            assert manifest['state']=='complete'
            for name in ('signals.png','signals.pdf','trajectories.png',
                         'trajectories.pdf','report.html','cycle_metrics.csv'):
                assert (folder/name).stat().st_size>0,name
            if spec.get('overlay') and spec.get('video'):
                assert (folder/'overlay.mp4').stat().st_size>0
            assert window.tracks is not None and window.review is not None
            assert window.run_button.isEnabled()
            window.tabs.setCurrentIndex(1)
            assert window.grab().save(str(screenshot))
            print(f'GUI analysis completed: {folder}',flush=True)
            result_code=0
        except Exception:
            record_error(traceback.format_exc())
        finally:
            window.close()
            app.quit()

    def start():
        window.run_button.click()
        if window.worker is not None:
            window.worker.progress.connect(lambda message:print(message,flush=True))
        elif not failures:
            record_error('GUI did not start the analysis worker.')
        timer.start()

    Path(output).mkdir(parents=True,exist_ok=True)
    Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
    timer=QTimer();timer.setInterval(100);timer.timeout.connect(check)
    window.show()
    QTimer.singleShot(100,start)
    try:
        app.exec()
    finally:
        sys.excepthook=previous_hook
    return result_code
