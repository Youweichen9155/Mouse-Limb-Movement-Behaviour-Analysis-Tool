from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
import pytest
from mouselimblab.video import open_video, oriented_opencv_reads, COORDINATE_SYSTEM
from mouselimblab.workflow import video_info, render_overlay
from mouselimblab.tracking import track_video, JOINTS
from mouselimblab.dlc_project import create_project, load_project

@pytest.mark.parametrize('rotation', [90, 180])
def test_rotated_video_preview_tracking_annotation_and_export_share_coordinates(tmp_path, rotation):
    source = Path(__file__).parent/'fixtures'/f'markers_rotate_{rotation}.mp4'
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    ok, encoded = cap.read(); cap.release()
    assert ok
    code = {90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[orientation]
    expected = cv2.rotate(encoded, code)
    cap = open_video(source);ok, upright = cap.read();cap.release()
    assert ok and np.array_equal(upright, expected)
    info = video_info(source)
    assert (info['height'],info['width']) == upright.shape[:2]
    assert info['coordinate_system'] == COORDINATE_SYSTEM
    # Locate the encoded seed positions after rotation independently using an ID mask.
    mask = np.zeros(encoded.shape[:2], np.uint8)
    for i,(x,y) in enumerate([(100,100),(200,150),(300,200),(400,250)],1):mask[y,x]=i
    mask = cv2.rotate(mask, code)
    seeds = {j:np.argwhere(mask==i)[0][::-1].tolist() for i,j in enumerate(JOINTS,1)}
    config = SimpleNamespace(seed_frame=0,clip_id='rotation',protocol='suspended',seeds=seeds)
    table, meta = track_video(source, config)
    assert meta['coordinate_system'] == COORDINATE_SYSTEM
    assert len(table)==30 and np.array_equal(table.source_frame,np.arange(30))
    for j,p in seeds.items():
        assert table[j+'_detected'].all()
        assert np.linalg.norm(table[[j+'_x_px',j+'_y_px']].to_numpy()-p,axis=1).max()<3
    project = create_project(tmp_path,'Rotation','Tester',[{'path':str(source),'animal_id':'a','split':'train'}],2)
    data = load_project(project)
    image = cv2.imread(str(project.parent/data['frames'][0]['image']))
    assert np.array_equal(image,expected)
    export = tmp_path/'overlay.mp4'
    render_overlay(source,table,export,lambda _:None,lambda:False)
    cap = open_video(export);ok, overlay = cap.read()
    assert ok and overlay.shape==upright.shape
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==30
    assert cap.get(cv2.CAP_PROP_ORIENTATION_META)==0
    cap.release()
    # DLC opens cv2 captures internally; its scoped reader must see identical pixels.
    original = cv2.VideoCapture
    with oriented_opencv_reads():
        cap = cv2.VideoCapture(str(source));ok, dlc_frame = cap.read();cap.release()
        assert ok and np.array_equal(dlc_frame,expected)
    assert cv2.VideoCapture is original

def test_dlc_capture_scope_restores_reader_on_error():
    original = cv2.VideoCapture
    with pytest.raises(RuntimeError), oriented_opencv_reads():
        raise RuntimeError('prediction failed')
    assert cv2.VideoCapture is original
