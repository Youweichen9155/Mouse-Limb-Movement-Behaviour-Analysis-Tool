from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from mouselimblab.analysis import Settings,analyse,smooth_segmented
from mouselimblab.demo import demo_table
from mouselimblab.io import load_coordinates,validate_frames
from mouselimblab.workflow import run_clip,summarize_reviewed,write_json

def test_missing_source_frames_are_not_compressed(tmp_path):
    x=demo_table().drop(index=[15,16,17]);p=tmp_path/'x.csv';x.to_csv(p,index=False)
    result,_=load_coordinates(p)
    assert len(result)==480
    assert result.loc[15:17,'hip_x_px'].isna().all()
    assert result.loc[18,'source_frame']==18

def test_long_gaps_remain_missing_and_no_cycle_crosses_them():
    x=demo_table()
    x.loc[170:205,['foot_x_px','foot_y_px']]=np.nan
    work,cycles,_,_=analyse(x,Settings(ground_y_px=220))
    assert work.loc[170:205,'foot_x_px_smooth'].isna().all()
    assert not ((cycles.source_frame_start<170)&(cycles.source_frame_end>205)).any()

def test_smoothing_does_not_bridge_missing_blocks():
    a=np.r_[np.zeros(50),np.full(10,np.nan),np.ones(50)*100]
    result=smooth_segmented(a,Settings())
    assert np.isnan(result[50:60]).all()
    assert abs(result[49])<1e-8 and abs(result[60]-100)<1e-8

def test_slow_motion_uses_explicit_capture_rate():
    x=demo_table()
    _,a,_,_=analyse(x,Settings(ground_y_px=220,capture_fps=30))
    _,b,_,_=analyse(x,Settings(ground_y_px=220,capture_fps=240))
    assert len(a)>2
    np.testing.assert_allclose(a.cycle_duration_real_s/8,b.cycle_duration_real_s)
    np.testing.assert_allclose(a.duty_factor,b.duty_factor)
    _,unknown,_,_=analyse(x,Settings(ground_y_px=220))
    assert unknown.cycle_duration_real_s.isna().all()

def test_suspended_never_has_ground_metrics():
    _,c,_,_=analyse(demo_table(),Settings(protocol='suspended'))
    assert len(c)>0
    assert c[['duty_factor','stance_swing_ratio','swing_drag_pct']].isna().all().all()

def test_source_offset_preserved():
    x=demo_table();x.source_frame+=100
    _,c,_,_=analyse(x,Settings(ground_y_px=220))
    assert c.source_frame_start.min()>=100

def test_stationary_clip_is_explicit_fail():
    x=demo_table()
    for col in x:
        if col!='source_frame':x[col]=x[col].iloc[0]
    _,c,_,s=analyse(x,Settings(ground_y_px=220))
    assert c.empty and s['status']=='FAIL'

@pytest.mark.parametrize('ids',[[0,1,1],[0,2,1],[0,1.2,2]])
def test_bad_frame_ids_rejected(ids):
    with pytest.raises(ValueError):validate_frames(pd.DataFrame({'source_frame':ids}))

def test_legacy_compressed_csv_rejected(tmp_path):
    p=tmp_path/'legacy.csv';pd.DataFrame({'Frame':[0,1],'Hip_X_cm':[1,2]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='source_frame'):load_coordinates(p)

def test_dlc_sides_are_not_merged(tmp_path):
    cols=pd.MultiIndex.from_product([['DLC'],['hipL','kneeL','ankleL','toeL','hipR','kneeR','ankleR','toeR'],['x','y','likelihood']])
    x=pd.DataFrame(np.ones((30,len(cols))),columns=cols)
    x[('DLC','hipL','likelihood')]=0.1
    p=tmp_path/'DLC.csv';x.to_csv(p)
    l,_=load_coordinates(p,side='l');r,_=load_coordinates(p,side='r')
    assert l.hip_x_px.isna().all() and r.hip_x_px.notna().all()

def test_non_destructive_outputs_and_reviewed_animal_summary(tmp_path):
    p=tmp_path/'tracks.csv';demo_table().to_csv(p,index=False);before=p.read_bytes()
    spec={'engine':'local','coordinates':str(p),'settings':{'ground_y_px':220},'animal_id':'A','group':'G','trial_id':'T'}
    folder=run_clip(spec,tmp_path/'results')
    assert p.read_bytes()==before
    assert (folder/'signals.pdf').exists() and (folder/'manifest.json').exists()
    with pytest.raises(ValueError,match='人工'):summarize_reviewed([folder],tmp_path/'summary0')
    review=pd.read_csv(folder/'cycle_review.csv');review.loc[review.automatic_include.eq(True),'manual_include']=1;review.to_csv(folder/'cycle_review.csv',index=False)
    out=summarize_reviewed([folder],tmp_path/'summary1')
    animals=pd.read_csv(out/'Prism_animal_level.csv');assert len(animals)==1 and animals.engine.iloc[0]=='local'
    with pytest.raises(ValueError,match='重复'):summarize_reviewed([folder,folder],tmp_path/'summary2')

def test_local_does_not_call_alma(monkeypatch):
    from mouselimblab.vendor import alma_upstream
    def forbidden(*a,**kw):raise AssertionError('ALMA must not run in local engine')
    monkeypatch.setattr(alma_upstream,'extract_parameters',forbidden)
    _,cycles,_,_=analyse(demo_table(),Settings(ground_y_px=220))
    assert len(cycles)>0

def test_manual_correction_survives_export_reload(tmp_path):
    from mouselimblab.workflow import apply_corrections
    table=demo_table();table['foot_likelihood']=.2
    edits=tmp_path/'edits.csv'
    pd.DataFrame([{'source_frame':20,'joint':'foot','x_px':5.,'y_px':6.}]).to_csv(edits,index=False)
    fixed=apply_corrections(table,edits);p=tmp_path/'corrected.csv';fixed.to_csv(p,index=False)
    loaded,_=load_coordinates(p,likelihood=.9)
    assert loaded.loc[20,'foot_x_px']==5.
    assert loaded.loc[19,'foot_x_px']!=loaded.loc[19,'foot_x_px']
    assert loaded.loc[20,'foot_likelihood']==.2
