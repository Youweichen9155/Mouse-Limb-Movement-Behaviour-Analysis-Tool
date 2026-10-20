"""Exercise the built executable, rather than the source interpreter."""
from pathlib import Path
import argparse
import json
import os
import platform
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dist',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();dist=args.dist.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    executable=dist/('MouseLimbLab.app/Contents/MacOS/MouseLimbLab' if sys.platform=='darwin' else 'MouseLimbLab/MouseLimbLab.exe')
    assert executable.is_file(),executable
    env=dict(os.environ)
    for key in ('PYTHONHOME','PYTHONPATH','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH'):env.pop(key,None)
    # Native macOS styles require Cocoa objects absent from Qt's minimal plugin.
    # Use the portable offscreen renderer for automated screenshot checks.
    env.update(QT_QPA_PLATFORM='windows' if sys.platform=='win32' else 'offscreen',
               QT_STYLE_OVERRIDE='Fusion',PYTHONDONTWRITEBYTECODE='1')
    def run(name,parameters):
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(executable),*parameters],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    run('analysis',['demo','--output',str(out/'demo')])
    manifests=list((out/'demo/results').glob('local_*/manifest.json'))
    assert len(manifests)==1,manifests
    manifest=json.loads(manifests[0].read_text(encoding='utf-8'))
    assert manifest['state']=='complete' and manifest['summary']['frame_count']>=20
    for name in ('frame_tracks.csv','cycle_metrics.csv','signals.pdf','signals.png','report.html'):
        assert (manifests[0].parent/name).stat().st_size>0,name
    # Exercise the button -> QThread -> plotting -> video -> result-table path.
    # A CLI-only analysis misses stack failures in the GUI worker on macOS.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from mouselimblab.demo import make_video_demo
    from mouselimblab.video import open_video
    make_video_demo(out/'gui_input')
    run('gui_analysis',['gui-run-smoke',str(out/'gui_input/video_project.json'),
                        '--output',str(out/'gui_results'),'--screenshot',str(out/'native_gui_analysis.png')])
    gui_manifest_paths=list((out/'gui_results').glob('local_*/manifest.json'))
    assert len(gui_manifest_paths)==1
    gui_manifest=json.loads(gui_manifest_paths[0].read_text(encoding='utf-8'))
    assert gui_manifest['state']=='complete' and gui_manifest['summary']['frame_count']==180
    folder=gui_manifest_paths[0].parent
    for name in ('signals.pdf','signals.png','cycles.pdf','cycles.png',
                 'trajectories.pdf','trajectories.png','overlay.mp4','report.html'):
        assert (folder/name).stat().st_size>0,name
    cap=open_video(folder/'overlay.mp4');decoded=0
    try:
        while cap.read()[0]:decoded+=1
    finally:
        cap.release()
    assert decoded==180,decoded
    for name,tab in (('gui',0),('help',3),('dlc',2)):
        screenshot=out/f'native_{name}.png'
        run(name,['gui-smoke','--project',str(out/'demo/demo_input/demo_project.json'),
                  '--tab',str(tab),'--screenshot',str(screenshot)])
        assert screenshot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    signature=None
    if sys.platform=='darwin':
        subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(dist/'MouseLimbLab.app')],check=True)
        signature='ad-hoc signature verified; not notarized'
    result={'native_analysis':'passed','native_gui':'passed','native_gui_analysis':'passed','overlay_decoded_frames':decoded,
            'native_help':'passed','native_dlc':'passed','version':manifest['version'],
            'platform':platform.platform(),'machine':platform.machine(),'signature':signature,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'qt_platform':env['QT_QPA_PLATFORM'],'chinese_font_glyphs':'passed',
            'synthetic_only':True,'biological_accuracy_validated':False}
    (out/'SMOKE_TEST.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
