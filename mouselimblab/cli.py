import argparse
import json
import sys
from pathlib import Path
from .workflow import run_clip,summarize_reviewed

def load_project(path):
    path=Path(path).resolve();spec=json.loads(path.read_text(encoding='utf-8'))
    for key in ('video','coordinates','corrections'):
        if spec.get(key):
            p=Path(spec[key]).expanduser()
            spec[key]=str((path.parent/p).resolve() if not p.is_absolute() else p)
    return spec

def main(argv=None):
    # Redirected Windows streams can default to a Western encoding. Chinese
    # progress text and file names must not abort an otherwise valid analysis.
    for stream in (sys.stdout,sys.stderr):
        if stream is not None and hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    p=argparse.ArgumentParser(description='MouseLimbLab — independent local / ALMA engines')
    sub=p.add_subparsers(dest='command')
    run=sub.add_parser('run');run.add_argument('project');run.add_argument('--output',required=True)
    batch=sub.add_parser('batch');batch.add_argument('projects',nargs='+');batch.add_argument('--output',required=True)
    summary=sub.add_parser('aggregate');summary.add_argument('results',nargs='+');summary.add_argument('--output',required=True)
    sub.add_parser('gui')
    smoke=sub.add_parser('gui-smoke');smoke.add_argument('--screenshot',required=True);smoke.add_argument('--project');smoke.add_argument('--dlc-project');smoke.add_argument('--tab',type=int,default=0)
    smoke.add_argument('--dlc-python');smoke.add_argument('--dlc-probe-output')
    demo=sub.add_parser('demo');demo.add_argument('--output',required=True)
    a=p.parse_args(argv)
    if a.command=='gui-smoke':
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QRawFont
        from .gui import Window,create_application
        app=create_application([])
        font=QRawFont.fromFont(app.font())
        assert all(font.supportsCharacter(c) for c in '中文Mouse123'), 'Missing UI font glyphs'
        window=Window()
        if a.project:window.apply_spec(load_project(a.project))
        if a.dlc_project:window.dlc_panel.load(a.dlc_project)
        window.tabs.setCurrentIndex(a.tab)
        window.show()
        def finish():
            window.grab().save(a.screenshot);window.close();app.quit()
        if a.dlc_probe_output:
            if not a.dlc_python:raise ValueError('--dlc-python is required for a probe')
            window.dlc_panel.python.setText(a.dlc_python)
            def probe():
                window.dlc_panel.start_action('probe')
                def complete(code,status):
                    import shutil
                    if window.dlc_panel.result_file.exists():shutil.copyfile(window.dlc_panel.result_file,a.dlc_probe_output)
                    finish()
                window.dlc_panel.process.finished.connect(complete)
            QTimer.singleShot(100,probe)
        else:QTimer.singleShot(500,finish)
        return app.exec()
    if a.command in (None,'gui'):
        from .gui import main as gui
        return gui()
    if a.command=='aggregate':print(summarize_reviewed(a.results,a.output));return 0
    if a.command=='demo':
        from .demo import make_demo
        spec=make_demo(Path(a.output)/'demo_input')
        print(run_clip(spec,Path(a.output)/'results',print));return 0
    failures=0
    for project in ([a.project] if a.command=='run' else a.projects):
        try:print(run_clip(load_project(project),a.output,print))
        except Exception as e:print(f'FAILED {project}: {e}');failures+=1
    return 1 if failures else 0
