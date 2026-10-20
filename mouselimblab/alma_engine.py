"""Adapter around the unmodified, pinned official ALMA kinematics engine.

No local cycle detector or local gait metrics are called here. Missing runs are
kept and analyzed as separate continuous segments; output uses source frames.
"""
import numpy as np
import pandas as pd
from .analysis import finite_runs, normalize, CYCLE_COLUMNS
from .tracking import interpolate_short_gaps
from .vendor import alma_upstream as alma

BODY = {'foot':'toe','mtp':'mtp','ankle':'ankle','knee':'knee','hip':'hip','crest':'iliac crest'}

def analyse_alma(table, settings):
    settings.validate('alma')
    required=[f'{j}_{a}_px' for j in BODY for a in ('x','y')]
    missing=[c for c in required if c not in table]
    if missing:raise ValueError('ALMA 需要六点轨迹：toe/MTP/ankle/knee/hip/iliac crest。缺少：'+', '.join(missing))
    work=table.copy(); filled=pd.DataFrame(index=range(len(work)))
    interpolated=np.zeros(len(work),bool)
    for j,bp in BODY.items():
        for axis in ('x','y'):
            col=f'{j}_{axis}_px'
            values,flags=interpolate_short_gaps(work[col].to_numpy(float),settings.max_gap_frames)
            filled[bp+' '+axis]=values
            interpolated |= flags
            work[col+'_interpolated']=values
            work[col+'_smooth']=np.nan
        filled[bp+' likelihood']=1.0  # input has already been confidence-masked
    coords=[bp+' '+a for bp in BODY.values() for a in ('x','y')]
    good=np.isfinite(filled[coords]).all(axis=1).to_numpy()
    frames=work.source_frame.to_numpy(int)
    for c in ['knee_angle_deg','ankle_angle_deg','hip_angle_deg','mtp_angle_deg',
              'foot_x_relative_to_hip_px','foot_marker_clearance_pct_shank']:
        work[c]=np.nan
    all_cycles=[];curve_rows=[];raw_tables=[];segment_notes=[]
    fps=settings.capture_fps
    for segment,(a,b) in enumerate(finite_runs(good),1):
        if b-a<30:
            segment_notes.append({'segment':segment,'first_frame':int(frames[a]),'last_frame':int(frames[b-1]),'state':'shorter_than_30_frames'})
            continue
        data=filled.iloc[a:b].copy().reset_index(drop=True)
        # Degenerate geometry cannot be evaluated by ALMA's acos angle formula.
        for j1,j2 in zip(list(BODY.values())[:-1],list(BODY.values())[1:]):
            distance=np.hypot(data[j1+' x']-data[j2+' x'],data[j1+' y']-data[j2+' y'])
            if (distance<1e-7).any():raise ValueError(f'ALMA 六点中 {j1}/{j2} 重叠，请修正坐标。')
        common=dict(step_height_min_cm=0.,step_height_max_cm=settings.step_height_max_cm,
                    stride_length_min_cm=0.,stride_length_max_cm=settings.stride_length_max_cm,
                    drag_clearance_cm=settings.alma_drag_clearance_cm,drag_min_consecutive_frames=settings.drag_min_frames)
        if settings.protocol=='treadmill':
            direction={'auto':'auto','positive':False,'negative':True}[settings.direction]
            result=alma.extract_parameters(fps,data.copy(),settings.cutoff_hz,'toe',cm_speed=settings.belt_cm_s,right_to_left=direction,**common)
        else:
            result=alma.extract_spontaneous_parameters(fps,data.copy(),settings.cutoff_hz,pixels_per_cm=settings.px_per_cm,**common)
        if result is None:raise ValueError('ALMA 未返回有效结果；请检查轨迹、方向和录制方式。')
        raw=result[0].copy()
        raw['segment_id']=segment;raw['segment_source_frame_start']=frames[a]
        raw_tables.append(raw)
        # Official joint-angle/filter functions for review plots; no local detector.
        filtered={}
        for j,bp in BODY.items():
            for axis in ('x','y'):
                values=alma.butterworth_filter(data[bp+' '+axis].to_numpy(),fps,settings.cutoff_hz)
                filtered[bp+' '+axis]=values
                work.loc[work.index[a:b],f'{j}_{axis}_px_smooth']=values
        angles=alma.compute_limb_joint_angles(*[filtered[bp+' '+axis] for bp in BODY.values() for axis in ('x','y')])
        for name,vals in zip(['mtp_angle_deg','ankle_angle_deg','knee_angle_deg','hip_angle_deg'],angles[:4]):
            work.loc[work.index[a:b],name]=vals
        work.loc[work.index[a:b],'foot_x_relative_to_hip_px']=filtered['toe x']-filtered['hip x']
        for _,row in raw.iterrows():
            # Spontaneous ALMA tables contain a string limb column, making
            # NumPy's assembled numeric fields strings as well.
            row=row.copy()
            for col in row.index:
                if col!='limb (hind left / right)':row[col]=pd.to_numeric(row[col],errors='coerce')
            s,e=row.get('stride_start (frame)'),row.get('stride_end (frame)')
            if not np.isfinite([s,e]).all():continue
            s,e=int(s),int(e)
            if not 0<=s<e<b-a:continue
            stance=row.get('stance duration (s)',np.nan)
            t=s+int(round(stance*fps)) if np.isfinite(stance) else s
            t=min(e,max(s,t))
            cid=f'A{len(all_cycles)+1:04d}'
            reason=[]
            if not np.isfinite(row.get('cycle duration (s)',np.nan)):
                reason.append('excluded_by_ALMA_stride_filter')
            elif not np.isclose(row['cycle duration (s)']*fps,e-s,atol=.01):
                reason.append('ALMA_time_frame_inconsistency')
            if any(interpolated[max(0,a+k-1):min(len(work),a+k+2)].any() for k in (s,t,e)):
                reason.append('interpolated_event_boundary')
            if np.mean(interpolated[a+s:a+e+1])>1-settings.min_coverage:
                reason.append('insufficient_raw_coverage')
            item=dict.fromkeys(CYCLE_COLUMNS,np.nan)
            item.update(cycle_id=cid,source_frame_start=frames[a+s],source_frame_transition=frames[a+t],source_frame_end=frames[a+e],
                        automatic_include=not reason,automatic_exclusion_reasons=';'.join(reason),manual_include='',review_note='',
                        cycle_frames=e-s,cycle_duration_video_s=(e-s)/settings.video_fps,
                        cycle_duration_real_s=row.get('cycle duration (s)',np.nan),engine='alma')
            # Keep ALMA parameter names unchanged; generic timing fields only aid review.
            item.update({str(k):v for k,v in row.items()})
            item['duty_factor']=stance/row['cycle duration (s)'] if row['cycle duration (s)']>0 else np.nan
            item['stance_swing_ratio']=stance/row['swing duration (s)'] if row['swing duration (s)']>0 else np.nan
            for label in ('swing','stance','drag'):
                item[f'ALMA_{label}_fraction']=row.get(label+' percentage (%)',np.nan)
                item[f'ALMA_{label}_pct']=100*row.get(label+' percentage (%)',np.nan)
            all_cycles.append(item)
            if not reason:
                columns=['knee_angle_deg','ankle_angle_deg','hip_angle_deg','mtp_angle_deg']
                vals={c:normalize(work[c].iloc[a+s:a+e+1].to_numpy()) for c in columns}
                for phase in range(101):curve_rows.append({'cycle_id':cid,'cycle_pct':phase,**{c:v[phase] for c,v in vals.items()}})
        segment_notes.append({'segment':segment,'first_frame':int(frames[a]),'last_frame':int(frames[b-1]),'state':'analysed','raw_rows':len(raw)})
    cycles=pd.DataFrame(all_cycles) if all_cycles else pd.DataFrame(columns=CYCLE_COLUMNS)
    curves=pd.DataFrame(curve_rows) if curve_rows else pd.DataFrame(columns=['cycle_id','cycle_pct','knee_angle_deg','ankle_angle_deg'])
    work['any_marker_interpolated']=interpolated
    work['video_time_s']=frames/settings.video_fps;work['real_time_s']=frames/settings.capture_fps
    n=int(cycles.automatic_include.sum()) if len(cycles) else 0
    summary={'engine':'alma','protocol':settings.protocol,'status':'REVIEW' if n else 'FAIL',
             'frame_count':len(work),'n_candidate_cycles':len(cycles),'n_automatic_cycles':n,
             'qc_reasons':'manual_review_required' if n else 'no_valid_ALMA_cycles',
             'ALMA_commit':'47703fc3769064a74c44febe923d8014a0325210',
             'spatial_calibration':'upstream_belt_speed_and_pixel_motion_estimate' if settings.protocol=='treadmill' else 'user_px_per_cm',
             'segments':segment_notes,'timing':'capture_fps_confirmed'}
    accepted=cycles.loc[cycles.automatic_include.eq(True)]
    upstream_cols=list(raw_tables[0].columns) if raw_tables else []
    for col in upstream_cols:
        if col in accepted and col not in ('segment_id','segment_source_frame_start'):
            summary['ALMA mean: '+col]=pd.to_numeric(accepted[col],errors='coerce').mean()
    raw=pd.concat(raw_tables,ignore_index=True) if raw_tables else pd.DataFrame(columns=['stride_start (frame)','stride_end (frame)'])
    return work,cycles,curves,summary,raw
