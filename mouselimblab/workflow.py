from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import hashlib
import html
import json
import platform
import uuid
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import __version__
from .analysis import Settings, analyse
from .io import load_coordinates
from .tracking import JOINTS, track_video
from .video import open_video, COORDINATE_SYSTEM
from .phases import add_phase_columns
from .phase_plots import plot_phases

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def json_safe(value):
    if isinstance(value, dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, Path): return str(value)
    return value

def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def video_info(path):
    cap = open_video(path)
    try:
        if not cap.isOpened(): raise ValueError('无法打开视频，请检查文件是否完整或转换为 MP4。')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0: raise ValueError('视频帧率无效。')
        return {'video_fps': fps, 'frame_count':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'orientation_degrees':float(cap.get(cv2.CAP_PROP_ORIENTATION_META)), 'coordinate_system':COORDINATE_SYSTEM}
    finally: cap.release()

def plot_results(work, cycles, curves, folder):
    plt.rcParams.update({'font.size':8, 'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(3,1, figsize=(8,6), sharex=True, layout='constrained')
    x = work.source_frame
    axes[0].plot(x,work.foot_x_relative_to_hip_px,lw=.7,color='#2563eb')
    axes[0].set_ylabel('Foot X - hip X (px)')
    for name,color in [('knee','#7c3aed'),('ankle','#059669')]:
        axes[1].plot(x,work[name+'_angle_deg'],label=name,color=color,lw=.7)
    axes[1].legend(frameon=False,ncol=2)
    axes[1].set_ylabel('Angle (deg)')
    if work.foot_marker_clearance_pct_shank.notna().any():
        axes[2].plot(x,work.foot_marker_clearance_pct_shank,lw=.7,color='#2563eb')
        axes[2].set_ylabel('Clearance (% shank)')
    else:
        axes[2].plot(x,work.foot_y_px_smooth,lw=.7,color='#2563eb')
        axes[2].set_ylabel('Foot image Y (px)')
    axes[2].set_xlabel('Original source frame')
    for row in cycles.itertuples():
        for ax in axes:
            ax.axvline(row.source_frame_start,lw=.4,color='#64748b')
            ax.axvline(row.source_frame_transition,lw=.4,color='#ea580c',ls='--')
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'signals.pdf'); fig.savefig(folder/'signals.png',dpi=600); plt.close(fig)
    if not curves.empty:
        fig,axes=plt.subplots(1,2,figsize=(7,3),layout='constrained')
        for ax,col in zip(axes,['knee_angle_deg','ankle_angle_deg']):
            for _, sub in curves.groupby('cycle_id'): ax.plot(sub.cycle_pct,sub[col],color='#94a3b8',alpha=.5,lw=.5)
            mean=curves.groupby('cycle_pct')[col].mean()
            ax.plot(mean.index,mean,color='#2563eb',lw=1.2)
            ax.set(xlabel='Cycle (%)',ylabel=col.replace('_',' '))
            ax.spines[['top','right']].set_visible(False)
        fig.savefig(folder/'cycles.pdf');fig.savefig(folder/'cycles.png',dpi=600);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
    for j in JOINTS: ax.plot(work[j+'_x_px_smooth'],work[j+'_y_px_smooth'],lw=.6,label=j)
    ax.invert_yaxis();ax.set_aspect('equal');ax.set(xlabel='Image X (px)',ylabel='Image Y (px)')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'trajectories.pdf');fig.savefig(folder/'trajectories.png',dpi=600);plt.close(fig)

def overlay_frame(frame, row):
    frame=frame.copy();points=[]
    joints=('crest','hip','knee','ankle','mtp','foot') if 'crest_x_px' in row.index and 'mtp_x_px' in row.index else JOINTS
    colours=[(210,70,170),(60,210,60),(250,110,40),(20,200,255),(80,160,255),(80,80,240)]
    for j,c in zip(joints,colours):
        x,y=row.get(j+'_x_px_smooth',row.get(j+'_x_px',np.nan)),row.get(j+'_y_px_smooth',row.get(j+'_y_px',np.nan))
        p=(round(x),round(y)) if np.isfinite([x,y]).all() else None
        points.append(p)
        if p:
            cv2.circle(frame,p,5,c,-1,cv2.LINE_AA)
            cv2.putText(frame,j,p,cv2.FONT_HERSHEY_SIMPLEX,.45,c,1,cv2.LINE_AA)
    for a,b in zip(points[:-1],points[1:]):
        if a and b:cv2.line(frame,a,b,(230,230,230),2,cv2.LINE_AA)
    return frame

def render_overlay(video,work,out,progress,cancelled):
    info=video_info(video);cap=open_video(video)
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),info['video_fps'],(info['width'],info['height']))
    if not writer.isOpened():
        cap.release();raise ValueError('此系统无法写入 MP4，请关闭叠加视频选项后重试。')
    lookup=work.set_index('source_frame')
    n=0
    try:
        while True:
            if cancelled(): raise InterruptedError('已取消导出')
            ok,frame=cap.read()
            if not ok:break
            if n in lookup.index: frame=overlay_frame(frame,lookup.loc[n])
            writer.write(frame);n+=1
            if n%60==0:progress(f'导出叠加视频 {n}/{info["frame_count"]}')
    finally: cap.release();writer.release()
    if n < info['frame_count']-1:raise ValueError('叠加视频读取提前停止，未完成导出。')

def apply_corrections(table,path):
    edits=pd.read_csv(path)
    required={'source_frame','joint','x_px','y_px'}
    if not required.issubset(edits):raise ValueError('校正表需要 source_frame,joint,x_px,y_px。')
    result=table.copy().set_index('source_frame')
    for row in edits.itertuples():
        if row.joint not in (*JOINTS,'mtp','crest') or row.source_frame not in result.index:
            raise ValueError('校正表中的关节或帧号不在本次轨迹范围内。')
        if not np.isfinite([row.x_px,row.y_px]).all():raise ValueError('校正点坐标必须是有限数值。')
        result.loc[row.source_frame,[row.joint+'_x_px',row.joint+'_y_px']]=[row.x_px,row.y_px]
        result.loc[row.source_frame,row.joint+'_detected']=True
        result.loc[row.source_frame,row.joint+'_manual_corrected']=True
    return result.reset_index()

def run_clip(spec,output_root,progress=lambda s:None,cancelled=lambda:False):
    engine=spec.get('engine','local')
    if engine not in ('local','alma'):raise ValueError('未知分析引擎。')
    settings=Settings(**spec.get('settings',{}));settings.validate(engine)
    if engine=='alma' and not spec.get('coordinates'):
        raise ValueError('ALMA 模式请导入六点 DLC/逐帧 CSV；四色追踪仅提供四点。')
    if engine=='alma' and spec.get('phase_events'):
        raise ValueError('手工相位边界用于四点方法；ALMA 使用官方相位事件。')
    if engine=='alma' and spec.get('swing_regions') is not None:
        raise ValueError('手动摆动区间用于四点方法；ALMA 使用官方相位事件。')
    spec = {**spec, 'engine':engine, 'settings':asdict(settings), 'coordinate_system':COORDINATE_SYSTEM}
    root=Path(output_root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    folder=root/(engine+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir()
    write_json(folder/'project.json',spec)
    try:
        sources={}
        if spec.get('coordinates'):
            progress('读取坐标并保留源帧号…')
            table,meta=load_coordinates(spec['coordinates'],spec.get('side','unsided'),spec.get('likelihood',.9))
            sources['coordinates']={'path':spec['coordinates'],'sha256':sha256(spec['coordinates'])}
        else:
            if not spec.get('video'):raise ValueError('请选择视频或坐标 CSV。')
            if set(spec.get('seeds',{})) != set(JOINTS):raise ValueError('请依次选择髋、膝、踝、足端四个初始点。')
            cfg=SimpleNamespace(clip_id='clip',protocol=settings.protocol,seed_frame=0,seeds=spec['seeds'])
            progress('逐帧追踪彩色标记…')
            table,meta=track_video(Path(spec['video']),cfg,lambda i,n:progress(f'追踪 {i}/{n} 帧'),cancelled)
            if abs(float(meta['fps_metadata'])-settings.video_fps) > .01:
                raise ValueError('项目播放帧率与视频不一致，请重新打开视频获取帧率。')
        if spec.get('video'):
            sources['video']={'path':spec['video'],'sha256':sha256(spec['video'])}
            info=video_info(spec['video'])
            if table.source_frame.max() >= info['frame_count']:
                raise ValueError('坐标帧号超过视频长度，请确认视频与 CSV 匹配且帧号从 0 开始。')
            if abs(info['video_fps']-settings.video_fps)>.01:
                raise ValueError('CSV 分析帧率与选中视频不一致，请重新加载视频或修正设置。')
        table.to_csv(folder/'raw_tracks.csv',index=False)
        if spec.get('corrections'):
            sources['corrections']={'path':spec['corrections'],'sha256':sha256(spec['corrections'])}
            table=apply_corrections(table,spec['corrections'])
        if cancelled():raise InterruptedError('已取消')
        progress('计算关节角、周期和质控…')
        if engine=='alma':
            from .alma_engine import analyse_alma
            work,cycles,curves,summary,raw_alma=analyse_alma(table,settings)
            raw_alma.to_csv(folder/'ALMA_official_parameters.csv',index=False)
        else:
            phase_events = None
            if spec.get('phase_events'):
                sources['phase_events']={'path':spec['phase_events'],'sha256':sha256(spec['phase_events'])}
                phase_events=pd.read_csv(spec['phase_events'])
            work,cycles,curves,summary=analyse(table,settings,phase_events,spec.get('swing_regions'))
        if engine=='alma':
            work=add_phase_columns(work,cycles,settings,'alma')
        if spec.get('swing_regions') is not None:
            from .phases import normalize_swing_regions
            regions=normalize_swing_regions(spec['swing_regions'],work.source_frame)
            pd.DataFrame(regions,columns=['swing_start_frame','swing_end_frame_inclusive']).to_csv(folder/'swing_regions.csv',index=False)
            sources['swing_regions']={'intervals':regions,'end_inclusive':True}
        summary.update(animal_id=spec.get('animal_id',''),group=spec.get('group',''),trial_id=spec.get('trial_id',''))
        work.to_csv(folder/'frame_tracks.csv',index=False)
        cycles.to_csv(folder/'cycle_metrics.csv',index=False)
        curves.to_csv(folder/'normalized_curves.csv',index=False)
        pd.DataFrame([summary]).to_csv(folder/'clip_summary.csv',index=False)
        cycles[['cycle_id','source_frame_start','source_frame_transition','source_frame_end','automatic_include','manual_include','review_note']].to_csv(folder/'cycle_review.csv',index=False)
        progress('导出轨迹图与周期质控图…');plot_results(work,cycles,curves,folder)
        progress('导出速度相位曲线与连续姿态图…');plot_phases(work,cycles,settings,folder,engine)
        if spec.get('overlay') and spec.get('video'):render_overlay(spec['video'],work,folder/'overlay.mp4',progress,cancelled)
        if cancelled():raise InterruptedError('已取消')
        write_json(folder/'manifest.json',{'version':__version__,'python':platform.python_version(),'platform':platform.platform(),
                   'engine':engine,'settings':asdict(settings),'inputs':sources,'import':meta,'summary':summary,
                   'ALMA_commit':'47703fc3769064a74c44febe923d8014a0325210','state':'complete'})
        method_note=('ALMA 踝角：膝–踝–跖趾关节。官方参数表 percentage 列保存 0–1 比例；周期表 ALMA_*_pct 列为百分数。'
                     if engine=='alma' else '四点踝角：膝–踝–足端。ROM 为每周期角度 P95−P5。')
        report=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MouseLimbLab 分析报告</title>
<style>body{{font:16px system-ui;max-width:1040px;margin:40px auto;padding:20px;color:#243045}}img{{max-width:100%}}td,th{{padding:6px;border-bottom:1px solid #ddd}}h1{{color:#2458a6}}</style>
<h1>小鼠后肢运动分析 · {engine.upper()}</h1><p>版本 {__version__} · {html.escape(summary['status'])} · 自动周期 {summary['n_automatic_cycles']}</p>
<p>在“周期与轨迹复核”中选择有效周期，保存后按动物汇总。动物编号与组别来自项目设置。</p>
<p>时序：{'已填写采集帧率，可报告真实秒数。' if settings.capture_fps else '仅播放时间；真实秒数留空。慢动作视频请填写实际采集帧率。'}</p>
<p>{method_note}</p>
<h2>连续后肢姿态</h2><p>{'红色：前伸；黑色：后撤。' if settings.protocol=='suspended' else '红色：摆动相；黑色：支撑相。'}灰色：未分类。
整段视频在同一幅图中连续展开。每帧髋部横向对齐，再按时间等距排布。</p><img src="limb_sequence.png">
<h2>足端速度与相位</h2><p>速度以髋部为参考，前伸为正。虚线标记摆动开始；点线标记周期起始。
{'ALMA 相位沿用官方事件。' if engine=='alma' else '相位方法：'+{'velocity':'速度换向','position':'位移峰谷','manual':'人工修正','manual_regions':'手动框选摆动区域（含起止帧）；其余有效帧为支撑相'}[summary['phase_method']]+'。'}
未填写实际采集帧率时，速度单位为 px/frame。</p><img src="phase_velocity.png">
{pd.DataFrame([summary]).T.to_html(header=False,escape=True)}<h2>轨迹</h2><img src="trajectories.png"><h2>原始帧与事件</h2><img src="signals.png">
<p><a href="limb_sequence.pdf">姿态图 PDF</a> · <a href="phase_velocity.pdf">速度相位图 PDF</a> ·
<a href="phase_frames.csv">逐帧速度与相位</a> · <a href="phase_events.csv">相位边界</a> ·
<a href="cycle_metrics.csv">逐周期数据</a> · <a href="frame_tracks.csv">逐帧数据</a> · <a href="normalized_curves.csv">标准化曲线</a> · <a href="manifest.json">参数与来源</a></p></html>'''
        (folder/'report.html').write_text(report,encoding='utf-8')
        progress(f'完成：{folder}')
        return folder
    except Exception as exc:
        write_json(folder/'failure.json',{'state':'cancelled' if isinstance(exc,InterruptedError) else 'failed','error':str(exc)})
        raise

def summarize_reviewed(folders,output):
    """Only explicitly approved, automatically valid cycles enter animal means."""
    rows=[]
    for folder in map(Path,folders):
        spec=json.loads((folder/'project.json').read_text(encoding='utf-8'))
        if not spec.get('animal_id') or not spec.get('group') or not spec.get('trial_id'):
            raise ValueError('每份结果都需要填写动物编号、组别、试次编号。')
        cycles=pd.read_csv(folder/'cycle_metrics.csv');review=pd.read_csv(folder/'cycle_review.csv')
        if review.cycle_id.duplicated().any():raise ValueError('复核表周期 ID 重复。')
        approved=set(review.loc[pd.to_numeric(review.manual_include,errors='coerce').eq(1),'cycle_id'])
        good=cycles.loc[cycles.cycle_id.isin(approved)&cycles.automatic_include.eq(True)]
        if good.empty:continue
        metrics=[c for c in good if c.endswith(('_deg','_pct','_real_s','(deg)','(cm)','(s)','(%)','(cm/s)')) or c in ('duty_factor','stance_swing_ratio')]
        means=good[metrics].apply(pd.to_numeric,errors='coerce').mean().to_dict()
        rows.append({**means,'animal_id':spec['animal_id'],'group':spec['group'],'trial_id':spec['trial_id'],
                     'engine':spec.get('engine','local'),'protocol':spec['settings']['protocol'],'n_reviewed_cycles':len(good),'run':str(folder)})
    if not rows:raise ValueError('没有人工明确纳入（manual_include=1）的合格周期。')
    clips=pd.DataFrame(rows)
    if clips.duplicated(['group','animal_id','protocol','engine','trial_id']).any():raise ValueError('同一动物试次重复，请仅选择一个分析版本。')
    if clips.groupby('animal_id')['group'].nunique().max()>1:raise ValueError('同一动物被分配到多个组，请核对映射。')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    clips.to_csv(out/'reviewed_trial_summary.csv',index=False)
    numeric=[c for c in clips.select_dtypes(include='number') if c!='n_reviewed_cycles']
    animals=clips.groupby(['group','animal_id','protocol','engine'])[numeric].mean().reset_index()
    animals.to_csv(out/'Prism_animal_level.csv',index=False)
    animals.groupby(['group','protocol','engine']).size().rename('n_animals').to_csv(out/'group_animal_counts.csv')
    return out
