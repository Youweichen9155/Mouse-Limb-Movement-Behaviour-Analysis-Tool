from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from mouselimblab.analysis import Settings, analyse
from mouselimblab.demo import demo_table
from mouselimblab.phases import derivative, manual_events, velocity_cycles, add_phase_columns
from mouselimblab.workflow import run_clip


def test_velocity_events_match_known_65_percent_stance():
    work, cycles, _, _ = analyse(demo_table(), Settings(ground_y_px=220))
    assert len(cycles) >= 6
    for row in cycles.itertuples():
        assert min(row.source_frame_start % 60, 60 - row.source_frame_start % 60) <= 2
        assert abs(row.source_frame_transition % 60 - 39) <= 2
        assert abs(row.duty_factor - .65) < .05
        assert work.loc[row.source_frame_start, 'phase'] == 'stance'
        assert work.loc[row.source_frame_transition, 'phase'] == 'swing'


def test_mirroring_and_body_translation_preserve_phase_and_speed():
    data = demo_table()
    settings = Settings(ground_y_px=220, capture_fps=240, px_per_cm=20)
    work, cycles, _, _ = analyse(data, settings)
    mirrored = data.copy()
    for col in data:
        if col.endswith('_x_px'):
            mirrored[col] = 1000 - data[col] + data.source_frame * 1.3
    other, other_cycles, _, _ = analyse(mirrored, settings)
    np.testing.assert_array_equal(cycles.source_frame_transition, other_cycles.source_frame_transition)
    np.testing.assert_allclose(work.foot_forward_velocity_cm_s.iloc[10:-10],
                               other.foot_forward_velocity_cm_s.iloc[10:-10], atol=1e-8)
    assert (work.phase == other.phase).all()


def test_speed_units_require_recording_calibration():
    settings = Settings(ground_y_px=220, px_per_cm=20)
    unknown, _, _, _ = analyse(demo_table(), settings)
    assert unknown.foot_forward_velocity_cm_s.isna().all()
    slow, _, _, _ = analyse(demo_table(), replace(settings, capture_fps=240))
    normal, _, _, _ = analyse(demo_table(), replace(settings, capture_fps=30))
    np.testing.assert_allclose(slow.foot_forward_velocity_cm_s, normal.foot_forward_velocity_cm_s * 8)
    assert (slow.phase == normal.phase).all()


def test_missing_intervals_and_stationary_clips_do_not_create_cycles():
    values = np.r_[np.arange(20.), np.full(20, np.nan), np.arange(20.) + 1000]
    speed = derivative(values)
    assert np.isnan(speed[20:40]).all()
    assert speed[19] == 1 and speed[40] == 1
    data = demo_table()
    data.loc[160:200, ['foot_x_px', 'foot_y_px']] = np.nan
    work, cycles, _, _ = analyse(data, Settings(ground_y_px=220))
    assert work.loc[160:200, 'phase'].eq('unclassified').all()
    assert not ((cycles.source_frame_start < 160) & (cycles.source_frame_end > 200)).any()
    events, _, info = velocity_cycles(np.ones(100), 30)
    assert events == [] and not info['states'].any()


def test_manual_boundary_recomputes_phase_durations_and_exports(tmp_path):
    data = demo_table()
    settings = Settings(ground_y_px=220, capture_fps=30)
    _, auto, _, _ = analyse(data, settings)
    events = auto[['source_frame_start', 'source_frame_transition', 'source_frame_end']].copy()
    events.loc[0, 'source_frame_transition'] -= 3
    work, reviewed, _, summary = analyse(data, settings, events)
    row = reviewed.iloc[0]
    assert summary['phase_method'] == 'manual'
    assert work.loc[int(row.source_frame_transition), 'phase'] == 'swing'
    assert row.stance_duration_real_s == (row.source_frame_transition - row.source_frame_start) / 30
    with pytest.raises(ValueError, match='重叠'):
        manual_events(pd.concat([events.iloc[[0]], events.iloc[[0]]]), data.source_frame, np.ones(len(data), bool))
    tracks = tmp_path / 'tracks.csv'; data.to_csv(tracks, index=False)
    edit = tmp_path / 'events.csv'; events.to_csv(edit, index=False)
    from mouselimblab.cli import load_project
    config = tmp_path / 'config.json'
    config.write_text('{"phase_events":"events.csv"}')
    assert load_project(config)['phase_events'] == str(edit.resolve())
    folder = run_clip({'coordinates': str(tracks), 'phase_events': str(edit),
                       'settings': {'ground_y_px': 220, 'capture_fps': 30}}, tmp_path / 'out')
    exported = pd.read_csv(folder / 'phase_frames.csv')
    assert exported.phase.equals(work.phase)
    for stem in ('limb_sequence', 'phase_velocity'):
        assert (folder / (stem + '.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert (folder / (stem + '.pdf')).read_bytes().startswith(b'%PDF-')
    poses = pd.read_csv(folder / 'limb_sequence_points.csv')
    assert set(poses.joint) == {'hip', 'knee', 'ankle', 'foot'}
    assert poses.source_frame.min() == 0 and poses.source_frame.max() == len(data) - 1
    invalid = reviewed.iloc[[0]].copy()
    invalid.source_frame_transition = invalid.source_frame_start
    unclassified = add_phase_columns(work, invalid, settings, 'alma')
    assert unclassified.phase.eq('unclassified').all()


def test_suspended_phase_names_and_legacy_option():
    work, _, _, _ = analyse(demo_table(), Settings(protocol='suspended'))
    assert {'retraction', 'protraction'}.issubset(set(work.phase))
    assert not set(work.phase) & {'stance', 'swing'}
    _, cycles, _, summary = analyse(demo_table(), Settings(ground_y_px=220, phase_method='position'))
    assert len(cycles) > 2 and summary['phase_method'] == 'position'


def test_manual_regions_replace_all_automatic_phases_and_allow_empty_selection(tmp_path):
    from mouselimblab.phases import normalize_swing_regions
    data=demo_table(); settings=Settings(ground_y_px=220,capture_fps=30)
    regions=[[39,59],[99,119],[len(data)-5,len(data)-1]]
    work,cycles,_,summary=analyse(data,settings,swing_regions=regions)
    expected=np.zeros(len(data),bool)
    for a,b in regions: expected[a:b+1]=True
    np.testing.assert_array_equal(work.phase.eq('swing'),expected)
    assert work.phase_source.eq('manual_regions').all()
    assert summary['phase_method']=='manual_regions'
    assert cycles.iloc[0].source_frame_start==60 and cycles.iloc[0].source_frame_transition==99
    assert cycles.iloc[0].source_frame_end==120
    assert cycles.iloc[0].swing_duration_real_s==21/30
    empty,empty_cycles,_,_=analyse(data,settings,swing_regions=[])
    assert empty.phase.eq('stance').all() and empty_cycles.empty
    assert normalize_swing_regions([[15,20],[4,8],[8,16]],data.source_frame)==[[4,20]]
    for bad in ([[1.5,2]], [[0,len(data)]], [[8,2]], [[float('nan'),4]]):
        with pytest.raises(ValueError):normalize_swing_regions(bad,data.source_frame)
    tracks=tmp_path/'tracks.csv';data.to_csv(tracks,index=False)
    folder=run_clip({'coordinates':str(tracks),'swing_regions':regions,
                     'settings':{'ground_y_px':220,'capture_fps':30}},tmp_path/'out')
    exported=pd.read_csv(folder/'phase_frames.csv')
    np.testing.assert_array_equal(exported.phase.eq('swing'),expected)
    assert pd.read_csv(folder/'swing_regions.csv').values.tolist()==regions
    poses=pd.read_csv(folder/'limb_sequence_points.csv')
    assert poses.row.nunique()==1 and poses.frame_stride.eq(1).all()
    assert poses.source_frame.nunique()==len(data)
    assert poses.groupby('source_frame').phase.first().equals(exported.set_index('source_frame').phase)


def test_manual_region_crossing_missing_data_does_not_invent_a_complete_cycle():
    data=demo_table();data.loc[70:80,['foot_x_px','foot_y_px']]=np.nan
    work,cycles,_,_=analyse(data,Settings(ground_y_px=220),swing_regions=[[39,59],[75,119]])
    assert work.loc[70:80,'phase'].eq('unclassified').all()
    assert cycles.empty
    assert work.loc[82:119,'phase'].eq('swing').all()
    assert work.loc[120:,'phase'].eq('stance').all()
