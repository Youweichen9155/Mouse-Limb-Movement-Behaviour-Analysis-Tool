"""Exercise the same background analysis and completion path as a button click."""
from pathlib import Path
import json
import sys
import traceback
from PySide6.QtCore import QTimer, QPoint, Qt
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
    completed=0

    def check():
        nonlocal result_code, completed
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
                         'trajectories.pdf','report.html','cycle_metrics.csv',
                         'limb_sequence.png','limb_sequence.pdf','phase_velocity.png',
                         'phase_velocity.pdf','phase_frames.csv'):
                assert (folder/name).stat().st_size>0,name
            if spec.get('overlay') and spec.get('video'):
                assert (folder/'overlay.mp4').stat().st_size>0
            assert window.tracks is not None and window.review is not None
            assert window.run_button.isEnabled()
            assert window.isVisible()
            completed+=1
            print(f'GUI analysis {completed} completed: {folder}',flush=True)
            if completed < 2:
                # Exercise the actual modal editor, mouse drag, Save button and
                # the next background analysis in this same application window.
                from PySide6.QtTest import QTest
                from PySide6.QtWidgets import QDialogButtonBox
                def select_regions():
                    try:
                        editor=window.phase_editor
                        assert not editor.regions  # automatic guesses are not preselected
                        canvas=editor.canvas
                        y=round(canvas.area().center().y())
                        offset=int(canvas.frames[0])
                        a=QPoint(round(canvas.pixel(offset+10)),y)
                        b=QPoint(round(canvas.pixel(offset+20)),y)
                        expected=sorted([canvas.frame_at(a.x()),canvas.frame_at(b.x())])
                        QTest.mousePress(canvas,Qt.LeftButton,pos=a)
                        QTest.mouseMove(canvas,b)
                        QTest.mouseRelease(canvas,Qt.LeftButton,pos=b)
                        assert editor.regions==[expected]
                        editor.add_region(offset+40,offset+50)
                        editor.table.item(1,1).setText(str(offset+52))
                        assert editor.regions[-1]==[offset+40,offset+52]
                        window._smoke_regions=[list(pair) for pair in editor.regions]
                        assert editor.grab().save(str(Path(screenshot).with_name('swing_region_editor.png')))
                        editor.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
                    except Exception:
                        record_error(traceback.format_exc())
                        window.phase_editor.reject()
                QTimer.singleShot(150,select_regions)
                window.edit_swing_regions()
                if failures:return
                timer.start()
                return
            import numpy as np
            import pandas as pd
            frame_data=pd.read_csv(folder/'phase_frames.csv')
            expected=np.zeros(len(frame_data),bool)
            for a,b in window._smoke_regions:
                expected|=frame_data.source_frame.between(a,b).to_numpy()
            assert np.array_equal(frame_data.phase.eq('swing'),expected)
            assert frame_data.phase_source.eq('manual_regions').all()
            saved=json.loads((folder/'project.json').read_text(encoding='utf-8'))
            assert saved['swing_regions']==window._smoke_regions
            poses=pd.read_csv(folder/'limb_sequence_points.csv')
            assert poses.row.nunique()==1 and poses.frame_stride.eq(1).all()
            print('Manual swing region editor, save, reload and whole-clip plot: passed',flush=True)
            window.tabs.setCurrentIndex(1)
            assert window.grab().save(str(screenshot))
            print(f'GUI analysis completed: {folder}',flush=True)
            result_code=0
        except Exception:
            record_error(traceback.format_exc())
        finally:
            if completed >= 2 or failures:
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
