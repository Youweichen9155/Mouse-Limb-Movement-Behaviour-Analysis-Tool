from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
from mouselimblab.alma_engine import analyse_alma,BODY
from mouselimblab.analysis import Settings
from mouselimblab.io import load_coordinates
from mouselimblab.demo import demo_table
from mouselimblab.vendor import alma_upstream as official

def test_requires_six_markers_and_real_capture_fps():
    table=demo_table().drop(columns=['mtp_x_px','mtp_y_px'])
    with pytest.raises(ValueError,match='六点'):analyse_alma(table,Settings(capture_fps=120,belt_cm_s=30))
    with pytest.raises(ValueError,match='采集帧率'):analyse_alma(demo_table(),Settings(belt_cm_s=30))

def test_official_file_is_unmodified():
    source=Path(__file__).parents[1]/'audit/originals/ALMA_KinematicsFunctions.py'
    assert hashlib.sha256(Path(official.__file__).read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()

def test_official_demo_matches_direct_ALMA_call(monkeypatch):
    from mouselimblab import analysis
    monkeypatch.setattr(analysis,'analyse',lambda *a: (_ for _ in ()).throw(AssertionError('local engine called')))
    p=Path(__file__).parents[1]/'examples/alma_official_demo.csv'
    table,_=load_coordinates(p,likelihood=0.)
    assert table[[f'{j}_{a}_px' for j in BODY for a in ('x','y')]].notna().all().all()
    settings=Settings(capture_fps=120,video_fps=120,belt_cm_s=30,direction='negative')
    data=pd.DataFrame()
    for j,bp in BODY.items():
        for axis in ('x','y'):data[bp+' '+axis]=table[j+'_'+axis+'_px'].to_numpy()
        data[bp+' likelihood']=1.
    np.random.seed(23)
    expected=official.extract_parameters(120,data.copy(),6,'toe',cm_speed=30,right_to_left=True,
        step_height_min_cm=0.,step_height_max_cm=1.5,stride_length_min_cm=0.,stride_length_max_cm=8.,drag_clearance_cm=.1,drag_min_consecutive_frames=4)[0]
    np.random.seed(23)
    _,cycles,_,summary,actual=analyse_alma(table,settings)
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False)
    assert len(actual)>0
    rejected=cycles['cycle duration (s)'].isna()
    assert not cycles.loc[rejected,'automatic_include'].any()
    assert summary['engine']=='alma'
    finite=cycles['drag percentage (%)'].notna()
    np.testing.assert_allclose(cycles.loc[finite,'ALMA_drag_pct'],100*cycles.loc[finite,'drag percentage (%)'])

def test_alma_suspended_is_rejected():
    with pytest.raises(ValueError):Settings(protocol='suspended',capture_fps=120).validate('alma')

def test_spontaneous_handles_upstream_string_numeric_fields():
    x=demo_table()
    for joint in BODY:x[joint+'_x_px']+=x.source_frame*(70/39)
    _,cycles,_,summary,raw=analyse_alma(x,Settings(protocol='spontaneous',capture_fps=30,px_per_cm=50))
    assert len(raw)>0 and len(cycles)>0
    assert summary['engine']=='alma'
    assert np.issubdtype(cycles.source_frame_start.dtype,np.integer)
