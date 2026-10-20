import copy
import json
from pathlib import Path
import subprocess
import sys
import os
import cv2
import numpy as np
import pytest
from mouselimblab.dlc_project import BODY_PARTS, create_project, load_project, save_json, validate_annotations, fingerprint, child_environment
from mouselimblab.io import load_coordinates

def make_project(tmp_path):
    videos=[]
    for animal,split in [('mouse_a','train'),('mouse_b','test')]:
        video=tmp_path/f'{animal}.avi'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(160,120))
        assert writer.isOpened()
        for n in range(16):
            frame=np.full((120,160,3),40,np.uint8)
            for j in range(6):cv2.circle(frame,(20+j*20,30+j*10+(n%3)),4,(40+j*30,210,140),-1)
            writer.write(frame)
        writer.release();videos.append({'path':str(video),'animal_id':animal,'split':split})
    project=create_project(tmp_path,'SyntheticTest','Tester',videos,4)
    data=load_project(project)
    for f in data['frames']:
        data['labels'][f['image']]={'complete':True,'points':{p:[20+j*20,30+j*10+(f['source_frame']%3)] for j,p in enumerate(BODY_PARTS)}}
    save_json(project,data)
    return project,data

def test_extract_preserves_source_frames_and_images(tmp_path):
    project,data=make_project(tmp_path)
    assert [f['source_frame'] for f in data['frames'][:4]]==[0,5,10,15]
    assert len(validate_annotations(data))==8
    assert all((project.parent/f['image']).is_file() for f in data['frames'])

def test_split_disallows_same_animal_in_test_and_train(tmp_path):
    _,data=make_project(tmp_path)
    data['videos'][1]['animal_id']='mouse_a'
    with pytest.raises(ValueError,match='同时'):validate_annotations(data)

def test_annotation_requires_complete_labels_and_finite_coordinates(tmp_path):
    _,data=make_project(tmp_path)
    key=data['frames'][0]['image']
    data['labels'][key]['points']['toe']=[float('nan'),3]
    with pytest.raises(ValueError,match='坐标'):validate_annotations(data)
    data['labels'][key]['points']['toe']=None
    assert validate_annotations(data)[0]['points']['toe'] is None
    data['labels'][key]['points'].pop('hip')
    with pytest.raises(ValueError,match='逐点'):validate_annotations(data)

def test_fingerprint_ignores_run_but_detects_new_labels(tmp_path):
    _,data=make_project(tmp_path)
    original=fingerprint(data);data['active_run']='runs/a/config.yaml'
    assert fingerprint(data)==original
    data['labels'][data['frames'][0]['image']]['points']['toe'][0]+=1
    assert fingerprint(data)!=original

def test_external_worker_cannot_write_bytecode_inside_signed_bundle(monkeypatch,tmp_path):
    monkeypatch.setenv('PYTHONHOME','/signed/app/runtime')
    monkeypatch.setenv('QT_PLUGIN_PATH','/signed/app/qt')
    (tmp_path/'packaged_helper.py').write_text('VALUE = 42\n')
    command='import sys;sys.path.insert(0,sys.argv[1]);import packaged_helper;assert packaged_helper.VALUE==42'
    result=subprocess.run([sys.executable,'-c',command,str(tmp_path)],env=child_environment(),capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert list(tmp_path.iterdir())==[tmp_path/'packaged_helper.py']

def test_new_dlc_keypoint_alias_and_missing_confidence(tmp_path):
    import pandas as pd
    columns=pd.MultiIndex.from_product([['DLC'],BODY_PARTS,['x','y','likelihood']],names=['scorer','bodyparts','coords'])
    table=pd.DataFrame([[20,30,.99]*6,[21,31,.1]*6],columns=columns)
    path=tmp_path/'predictions.csv';table.to_csv(path)
    parsed,_=load_coordinates(path,likelihood=.9)
    assert parsed.crest_x_px.iloc[0]==20 and np.isnan(parsed.crest_x_px.iloc[1])
    assert list(parsed.source_frame)==[0,1]

def test_training_ui_annotation_roundtrip_and_prediction_handoff(tmp_path):
    project,_=make_project(tmp_path)
    import pandas as pd
    columns=pd.MultiIndex.from_product([['DLC'],BODY_PARTS,['x','y','likelihood']],names=['scorer','bodyparts','coords'])
    pd.DataFrame([[20,30,.99]*6]*16,columns=columns).to_csv(tmp_path/'coordinates.csv')
    code='''
from PySide6.QtWidgets import QApplication
from mouselimblab.gui import Window
from mouselimblab.dlc_project import load_project
import sys
from pathlib import Path
app=QApplication([]);w=Window();panel=w.dlc_panel
panel.load(sys.argv[1]);panel.frame_select.setCurrentIndex(0);panel.part.setCurrentIndex(0)
panel.place_point(30,40)
assert load_project(sys.argv[1])['labels'][panel.data['frames'][0]['image']]['points']['iliac_crest']==[30,40]
panel.complete_frame()
assert load_project(sys.argv[1])['labels'][panel.data['frames'][0]['image']]['complete']
video=panel.data['videos'][0]['path']
w.capture_fps.setValue(240);w.belt.setValue(7)
coordinates=str(Path(video).parent/'coordinates.csv')
w.use_dlc_prediction(video,coordinates,'alma',.8)
assert w.engine.currentData()=='alma' and w.coordinates==coordinates
assert len(w.tracks)==16
assert w.capture_fps.value()==0 and w.belt.value()==0
assert w.conf.value()==.8
w.close()
'''
    p=subprocess.run([sys.executable,'-c',code,str(project)],capture_output=True,text=True,env={**os.environ,'QT_QPA_PLATFORM':'minimal'},timeout=30)
    assert p.returncode==0,p.stderr
