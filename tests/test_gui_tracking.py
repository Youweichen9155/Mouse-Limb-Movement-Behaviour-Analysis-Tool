import os
os.environ.setdefault('QT_QPA_PLATFORM','minimal')
import cv2
import numpy as np
from types import SimpleNamespace
from mouselimblab.tracking import JOINTS,track_video

def test_video_tracking_keeps_frame_numbers_during_missing_marker(tmp_path):
    p=tmp_path/'markers.avi'
    writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),30,(640,480))
    assert writer.isOpened()
    points=[(100,100),(200,150),(300,200),(400,250)]
    colours=[(200,0,200),(0,220,0),(220,80,0),(0,160,255)]
    for f in range(30):
        frame=np.full((480,640,3),30,np.uint8)
        for j,(point,colour) in enumerate(zip(points,colours)):
            if not (j==3 and 12<=f<16):cv2.circle(frame,point,11,colour,-1)
        writer.write(frame)
    writer.release()
    config=SimpleNamespace(clip_id='test',protocol='treadmill',seed_frame=0,seeds=dict(zip(JOINTS,points)))
    table,meta=track_video(p,config)
    assert len(table)==30
    assert list(table.source_frame)==list(range(30))
    assert table.loc[12:15,'foot_x_px'].isna().all()

def test_gui_engine_switch_and_configuration_roundtrip():
    # Test application startup in a fresh process, as launched by the user.
    import subprocess,sys
    script='''
from PySide6.QtWidgets import QApplication
from mouselimblab.gui import Window
app=QApplication([])
w=Window();w.engine.setCurrentIndex(1)
assert w.protocol.itemData(1)=='spontaneous'
assert not w.seed_buttons[0].isEnabled()
w.capture_fps.setValue(240);w.belt.setValue(30)
spec=w.spec();w.apply_spec(spec)
assert w.spec()==spec
w.engine.setCurrentIndex(0)
assert w.protocol.itemData(1)=='suspended'
import mouselimblab.gui as gui
gui.video_info=lambda p:dict(video_fps=30,frame_count=180)
w.show_frame=lambda n:None
w.coordinates='previous.csv';w.phase_events_path='previous_events.csv';w.swing_regions=[[10,20]]
w.seeds={'hip':[1,2]};w.target='foot'
w.set_video('next.mp4',reset=True)
assert w.coordinates==w.phase_events_path==''
assert w.swing_regions is None
assert w.seeds=={} and w.target is None and w.result is None
class Busy:
    def isRunning(self):return True
w.worker=Busy();errors=[];w.error=errors.append
w.start();w.save_phases()
assert len(errors)==2 and all('尚未完成' in message for message in errors)
w.worker=None
w.close()
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env={**os.environ,'QT_QPA_PLATFORM':'minimal'},timeout=30)
    assert result.returncode==0,result.stderr

def test_gui_video_analysis_exports_images_and_full_overlay(tmp_path):
    import json
    import subprocess
    import sys
    from mouselimblab.demo import make_video_demo
    from mouselimblab.video import open_video
    make_video_demo(tmp_path/'input')
    result=subprocess.run(
        [sys.executable,'-m','mouselimblab','gui-run-smoke',str(tmp_path/'input/video_project.json'),
         '--output',str(tmp_path/'results'),'--screenshot',str(tmp_path/'completed.png')],
        capture_output=True,text=True,timeout=120,
        env={**os.environ,'QT_QPA_PLATFORM':'offscreen','OPENBLAS_NUM_THREADS':'2'})
    assert result.returncode==0,result.stdout+result.stderr
    manifests=list((tmp_path/'results').glob('*/manifest.json'))
    assert len(manifests)==2
    manifest=json.loads(manifests[0].read_text())
    assert manifest['state']=='complete' and manifest['summary']['frame_count']==180
    folder=manifests[0].parent
    for name in ('signals.png','signals.pdf','cycles.png','cycles.pdf',
                 'trajectories.png','trajectories.pdf','report.html',
                 'limb_sequence.png','limb_sequence.pdf','phase_velocity.png',
                 'phase_velocity.pdf','phase_frames.csv'):
        assert (folder/name).stat().st_size>0,name
    cap=open_video(folder/'overlay.mp4');frames=0
    try:
        while cap.read()[0]:frames+=1
    finally:
        cap.release()
    assert frames==180
    assert (tmp_path/'completed.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
