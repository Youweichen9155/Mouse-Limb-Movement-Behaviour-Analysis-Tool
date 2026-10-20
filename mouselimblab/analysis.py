"""Independent project four-point gait analysis; no ALMA calculations."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from .tracking import (JOINTS, interpolate_short_gaps, angle_deg, robust_cv,
                       robust_scale, find_candidate_cycles)

@dataclass
class Settings:
    protocol: str = 'treadmill'
    video_fps: float = 30.0
    capture_fps: float | None = None
    ground_y_px: float | None = None
    px_per_cm: float | None = None
    belt_cm_s: float | None = None
    direction: str = 'auto'
    max_gap_frames: int = 2
    min_coverage: float = 0.95
    cutoff_hz: float = 6.0
    drag_min_frames: int = 4
    alma_drag_clearance_cm: float = 0.1
    step_height_max_cm: float = 1.5
    stride_length_max_cm: float = 8.0

    def validate(self, engine="local"):
        if self.protocol not in (('treadmill', 'suspended') if engine == 'local' else ('treadmill', 'spontaneous')):
            raise ValueError('请选择跑台或悬空实验类型。')
        for name in ('video_fps', 'capture_fps', 'px_per_cm', 'belt_cm_s', 'cutoff_hz'):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} 必须大于 0。')
        if engine == 'local' and self.protocol == 'treadmill' and (self.ground_y_px is None or not np.isfinite(self.ground_y_px)):
            raise ValueError('跑台分析需要选择地面参考线。图像 y 轴向下，视频必须正立。')
        if self.max_gap_frames < 0 or self.drag_min_frames < 1 or not 0 < self.min_coverage <= 1:
            raise ValueError('缺口、连续拖曳帧数或覆盖率参数无效。')
        if engine == 'alma':
            if not self.capture_fps:
                raise ValueError('ALMA 模式必须填写实际采集帧率，以计算秒数和速度。')
            if not 0 < self.cutoff_hz < self.capture_fps / 2:
                raise ValueError('ALMA 截止频率必须小于实际采集帧率的一半。')
            if self.protocol == 'treadmill' and not self.belt_cm_s:
                raise ValueError('ALMA 跑台模式必须填写跑带速度 cm/s。')
            if self.protocol == 'spontaneous' and not self.px_per_cm:
                raise ValueError('ALMA 自发行走模式必须填写空间标尺 px/cm。')
            if self.alma_drag_clearance_cm < 0 or self.step_height_max_cm <= 0 or self.stride_length_max_cm <= 0:
                raise ValueError('ALMA 拖曳阈值和步长/步高筛选上限无效。')
        if self.direction not in ('auto','positive','negative'):
            raise ValueError('行走方向无效。')

def finite_runs(valid):
    edges = np.diff(np.r_[False, np.asarray(valid, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def smooth_segmented(values, settings):
    result = np.asarray(values, float).copy()
    for a, b in finite_runs(np.isfinite(result)):
        v = result[a:b].copy()
        if len(v) >= 5:
            window = min(11, max(5, round(0.1 * settings.video_fps)))
            window += 1 - window % 2
            window = min(window, len(v) if len(v) % 2 else len(v)-1)
            result[a:b] = savgol_filter(median_filter(v, size=3, mode='nearest'), window, 2)
    return result

def sustained_mask(values, minimum):
    """Keep every frame in qualifying runs, including terminal runs."""
    out = np.zeros(len(values), bool)
    for a, b in finite_runs(values):
        if b - a >= minimum:
            out[a:b] = True
    return out

def normalize(values):
    return np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, len(values)), values)

CYCLE_COLUMNS = ['cycle_id', 'source_frame_start', 'source_frame_transition', 'source_frame_end',
    'automatic_include', 'automatic_exclusion_reasons', 'manual_include', 'review_note',
    'knee_rom_p95_p5_deg', 'ankle_rom_p95_p5_deg', 'cycle_frames',
    'cycle_duration_video_s', 'cycle_duration_real_s', 'duty_factor', 'stance_swing_ratio',
    'peak_clearance_pct_shank', 'swing_drag_pct']

def analyse(table, settings):
    settings.validate()
    if len(table) < 20:
        raise ValueError('至少需要 20 帧才能进行周期分析。')
    frames = table['source_frame'].to_numpy(int)
    if not np.all(np.diff(frames) == 1):
        raise ValueError('先补齐原始帧号再分析。')
    work = table.copy()
    points, interpolated = {}, np.zeros(len(work), bool)
    raw_valid = np.ones(len(work), bool)
    for joint in (*JOINTS, 'mtp', 'crest'):
        if not all(f'{joint}_{a}_px' in work for a in ('x', 'y')):
            continue
        coords = []
        for axis in ('x', 'y'):
            col = f'{joint}_{axis}_px'
            raw = work[col].to_numpy(float)
            if joint in JOINTS:
                raw_valid &= np.isfinite(raw)
            filled, flags = interpolate_short_gaps(raw, settings.max_gap_frames)
            if joint in JOINTS:
                interpolated |= flags
            work[col + '_interpolated'] = filled
            work[col + '_smooth'] = smooth_segmented(filled, settings)
            coords.append(work[col + '_smooth'].to_numpy())
        points[joint] = np.column_stack(coords)
    work['any_marker_interpolated'] = interpolated
    valid = np.column_stack([points[j] for j in JOINTS])
    valid = np.isfinite(valid).all(axis=1)
    if valid.sum() < 20:
        raise ValueError('四关节完整坐标少于 20 帧；请检查追踪、侧别和置信阈值。')
    segments = {'thigh': ('hip', 'knee'), 'shank': ('knee', 'ankle'), 'ankle_foot': ('ankle', 'foot')}
    for name, (a, b) in segments.items():
        work[name + '_length_px'] = np.linalg.norm(points[a] - points[b], axis=1)
    shank = float(np.nanmedian(work['shank_length_px']))
    if not np.isfinite(shank) or shank <= 0:
        raise ValueError('胫段长度无效；膝和踝可能选到了同一点。')
    angles = {'knee_angle_deg': ('hip', 'knee', 'ankle'), 'ankle_angle_deg': ('knee', 'ankle', 'foot')}
    for name, (a, b, c) in angles.items():
        work[name] = angle_deg(points[a], points[b], points[c])
    xrel = points['foot'][:, 0] - points['hip'][:, 0]
    work['foot_x_relative_to_hip_px'] = xrel
    raw_clearance = settings.ground_y_px - points['foot'][:, 1] if settings.protocol == 'treadmill' else None
    events = []
    signs = []
    for a, b in finite_runs(valid):
        if b - a < 20:
            continue
        local, sign, _ = find_candidate_cycles(xrel[a:b], settings.video_fps, raw_clearance[a:b] if raw_clearance is not None else None)
        if settings.direction != 'auto':
            sign = 1 if settings.direction == 'positive' else -1
            from scipy.signal import find_peaks
            oriented = xrel[a:b] * sign
            spread = np.quantile(oriented, .95) - np.quantile(oriented, .05)
            peaks, _ = find_peaks(oriented, prominence=max(1e-8, .12 * spread), distance=max(3, round(.15 * settings.video_fps)))
            local = [(int(s), int(s + np.argmin(oriented[s:e+1])), int(e)) for s, e in zip(peaks[:-1], peaks[1:])]
            local = [(s,t,e) for s,t,e in local if t-s >= 3 and e-t >= 3]
        signs.append(sign)
        events.extend((a+s, a+t, a+e) for s,t,e in local)
    # Ground baseline comes only from candidates with sufficient raw coverage.
    stance = []
    for s, t, e in events:
        if raw_valid[s:e+1].mean() >= settings.min_coverage:
            stance.extend(range(s + round(.2*(t-s)), max(s + round(.2*(t-s)) + 1, s + round(.8*(t-s)))))
    clearance = np.full(len(work), np.nan)
    noise = threshold = np.nan
    if stance and raw_clearance is not None:
        clearance = raw_clearance - np.nanmedian(raw_clearance[stance])
        noise = robust_scale(clearance[stance])
        threshold = max(3*noise, .03*shank)
    work['foot_marker_clearance_px'] = clearance
    work['foot_marker_clearance_pct_shank'] = clearance / shank * 100
    if settings.px_per_cm:
        work['foot_marker_clearance_cm'] = clearance / settings.px_per_cm
    work['video_time_s'] = frames / settings.video_fps
    work['real_time_s'] = frames / settings.capture_fps if settings.capture_fps else np.nan
    cycles, curves, paths = [], [], []
    for i, (s,t,e) in enumerate(events, 1):
        reasons = []
        if raw_valid[s:e+1].mean() < settings.min_coverage:
            reasons.append('insufficient_raw_coverage')
        if any(interpolated[max(0,k-1):min(len(work),k+2)].any() for k in (s,t,e)):
            reasons.append('interpolated_event_boundary')
        if not np.isfinite(work[['knee_angle_deg','ankle_angle_deg']].iloc[s:e+1]).all().all():
            reasons.append('degenerate_joint_geometry')
        row = dict.fromkeys(CYCLE_COLUMNS, np.nan)
        row.update(cycle_id=f'C{i:04d}', source_frame_start=frames[s], source_frame_transition=frames[t], source_frame_end=frames[e],
                   automatic_include=not reasons, automatic_exclusion_reasons=';'.join(reasons), manual_include='', review_note='',
                   marker_coverage_fraction=float(raw_valid[s:e+1].mean()), cycle_frames=e-s,
                   cycle_duration_video_s=(e-s)/settings.video_fps,
                   cycle_duration_real_s=(e-s)/settings.capture_fps if settings.capture_fps else np.nan)
        for joint in ('knee', 'ankle'):
            v = work[f'{joint}_angle_deg'].iloc[s:e+1].to_numpy()
            row[f'{joint}_rom_p95_p5_deg'] = np.nanquantile(v,.95)-np.nanquantile(v,.05)
        if settings.protocol == 'treadmill':
            row.update(duty_factor=(t-s)/(e-s), stance_swing_ratio=(t-s)/(e-t),
                       stance_duration_real_s=(t-s)/settings.capture_fps if settings.capture_fps else np.nan,
                       swing_duration_real_s=(e-t)/settings.capture_fps if settings.capture_fps else np.nan)
            if np.isfinite(clearance[t:e]).all():
                ca, cb = t + round(.1*(e-t)), t + round(.9*(e-t))
                central = clearance[ca:max(ca+1,cb)]
                row['peak_clearance_pct_shank'] = np.quantile(clearance[t:e], .95)/shank*100
                row['drag_metric_reliable'] = bool(threshold <= .1*shank)
                row['swing_drag_pct'] = float(np.mean(central <= threshold)*100) if row['drag_metric_reliable'] else np.nan
            if settings.belt_cm_s and settings.capture_fps:
                row['belt_travel_per_cycle_cm'] = settings.belt_cm_s*(e-s)/settings.capture_fps
        cycles.append(row)
        if reasons:
            continue
        values = {name: normalize(work[name].iloc[s:e+1].to_numpy()) for name in angles
                  if np.isfinite(work[name].iloc[s:e+1]).all()}
        values['foot_marker_clearance_pct_shank'] = normalize(clearance[s:e+1]/shank*100)
        values['foot_x_relative_pct_shank'] = normalize(xrel[s:e+1]/shank*100)
        for phase in range(101):
            curves.append({'cycle_id': row['cycle_id'], 'cycle_pct': phase, **{k:v[phase] for k,v in values.items()}})
    cycle_table = pd.DataFrame(cycles) if cycles else pd.DataFrame(columns=CYCLE_COLUMNS)
    curve_table = pd.DataFrame(curves) if curves else pd.DataFrame(columns=['cycle_id','cycle_pct',*angles,'foot_marker_clearance_pct_shank'])
    n = int(cycle_table.automatic_include.sum()) if len(cycle_table) else 0
    qc = []
    if raw_valid.mean() < .95: qc.append('raw_coverage_below_95pct')
    if n < 3: qc.append('fewer_than_3_cycles')
    if any(robust_cv(work[k+'_length_px']) > limit for k,limit in [('thigh',10),('shank',10),('ankle_foot',15)]): qc.append('segment_length_variation')
    if threshold > .1*shank: qc.append('ground_noise_high')
    summary = {'protocol':settings.protocol,'frame_count':len(work),'raw_complete_fraction':raw_valid.mean(),
               'n_candidate_cycles':len(cycle_table),'n_automatic_cycles':n,
               'status':'FAIL' if n == 0 or raw_valid.mean() < .75 else 'REVIEW' if qc else 'PASS_AUTOMATIC',
               'qc_reasons':';'.join(qc),'median_shank_px':shank,'ground_noise_px':noise,
               'drag_threshold_pct_shank':threshold/shank*100,
               'timing':'capture_fps_confirmed' if settings.capture_fps else 'video_time_only',
               'directions_per_valid_segment':','.join(map(str,signs)),
               'engine':'local'}
    accepted = cycle_table.loc[cycle_table.automatic_include.astype(bool)]
    for col in ('knee_rom_p95_p5_deg','ankle_rom_p95_p5_deg','duty_factor','stance_swing_ratio','peak_clearance_pct_shank','swing_drag_pct'):
        summary[col+'_mean'] = pd.to_numeric(accepted[col],errors='coerce').mean()
    return work, cycle_table, curve_table, summary
