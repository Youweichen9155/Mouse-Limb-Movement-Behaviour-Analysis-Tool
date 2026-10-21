"""Velocity-based phase events and auditable per-frame phase exports."""
import numpy as np
import pandas as pd


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def derivative(values):
    """Pixels per source frame; never differentiate across a missing block."""
    values = np.asarray(values, float)
    result = np.full(len(values), np.nan)
    for a, b in runs(np.isfinite(values)):
        if b - a >= 3:
            result[a:b] = np.gradient(values[a:b])
    return result


def normalize_swing_regions(regions, frames):
    """Inclusive source-frame intervals; None and an empty list differ."""
    frames = np.asarray(frames, int)
    if not len(frames):
        raise ValueError('没有可划定的帧。')
    if len(regions) == 0:
        return []
    try:
        values = np.asarray(regions, float)
    except (TypeError, ValueError):
        raise ValueError('摆动区间需要整数起止帧。') from None
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all() or not (values == np.floor(values)).all():
        raise ValueError('摆动区间需要整数起止帧。')
    merged = []
    for a, b in sorted(values.astype(int).tolist()):
        if a > b or a < frames[0] or b > frames[-1]:
            raise ValueError('摆动区间起始帧不能大于结束帧，且需在视频范围内。')
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return merged


def region_cycles(regions, frames, valid):
    """Complete touchdown-to-touchdown cycles between consecutive swings."""
    lookup = {int(f): i for i, f in enumerate(frames)}
    events = []
    for previous, current in zip(regions[:-1], regions[1:]):
        bounds = previous[1] + 1, current[0], current[1] + 1
        if all(f in lookup for f in bounds):
            s, t, e = (lookup[f] for f in bounds)
            if s < t < e and valid[s:e + 1].all():
                events.append((s, t, e))
    return events


def velocity_cycles(x, fps, clearance=None, direction='auto'):
    """Touchdown -> toe-off -> touchdown from sustained velocity reversals.

    A 10% velocity dead band and three-frame confirmation reject jitter.
    Swing excursions must exceed 10% of the robust foot excursion to reject
    small reversals during support. Events use the last zero crossing before
    a confirmed reversal; a stationary foot can remain in stance.
    """
    x = np.asarray(x, float)
    dx = np.gradient(x)
    dead = max(1e-8, .10 * np.quantile(np.abs(dx), .90))
    if direction != 'auto':
        sign = 1 if direction == 'positive' else -1
    else:
        pos = np.asarray(clearance)[dx > dead] if clearance is not None else []
        neg = np.asarray(clearance)[dx < -dead] if clearance is not None else []
        if len(pos) and len(neg) and np.isfinite([np.median(pos), np.median(neg)]).all():
            sign = 1 if np.median(pos) >= np.median(neg) else -1
        else:
            sign = 1 if np.quantile(dx, .99) >= abs(np.quantile(dx, .01)) else -1
    velocity = dx * sign
    spread = np.quantile(x, .95) - np.quantile(x, .05)
    strong = np.where(velocity > dead, 1, np.where(velocity < -dead, -1, 0))
    confirmed = sorted((a, b, state) for state in (-1, 1)
                       for a, b in runs(strong == state) if b - a >= 3
                       and (state == -1 or sign * (x[b - 1] - x[a]) >= max(1e-8, .1 * spread)))
    bout = []
    for a, b, state in confirmed:
        if bout and bout[-1][2] == state:
            bout[-1][1] = b
        else:
            bout.append([a, b, state])
    events = []
    states = np.zeros(len(x), dtype=int)
    if bout:
        transitions = []
        for previous, current in zip(bout[:-1], bout[1:]):
            left, right = previous[1] - 1, current[0]
            opposite = np.flatnonzero(velocity[left:right + 1] * current[2] <= 0)
            boundary = left + int(opposite[-1]) + 1 if len(opposite) else right
            transitions.append((boundary, current[2]))
        begin, state = bout[0][0], bout[0][2]
        for boundary, next_state in transitions:
            states[begin:boundary] = state
            begin, state = boundary, next_state
        states[begin:bout[-1][1]] = state
        for i in range(len(transitions) - 2):
            (s, first), (t, second), (e, third) = transitions[i:i + 3]
            if (first, second, third) != (-1, 1, -1):
                continue
            amplitude = min(sign * (x[s] - x[t]), sign * (x[e] - x[t]))
            if t - s >= 3 and e - t >= 3 and e - s >= 10 and amplitude >= max(1e-8, .1 * spread):
                events.append((s, t, e))
    if events:
        median = np.median([e - s for s, _, e in events])
        events = [(s, t, e) for s, t, e in events if .45 * median <= e - s <= 2.25 * median]
    return events, sign, {'velocity_dead_band_px_per_frame': dead, 'states': states}


def manual_events(table, frames, valid):
    required = ['source_frame_start', 'source_frame_transition', 'source_frame_end']
    if not set(required).issubset(table):
        raise ValueError('相位表需要起始、摆动开始和结束源帧号。')
    values = table[required].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or not (values == np.floor(values)).all():
        raise ValueError('相位边界必须是整数源帧号，且至少包含一个周期。')
    lookup = {int(frame): i for i, frame in enumerate(frames)}
    result = []
    for s, t, e in values.astype(int):
        if not s < t < e or any(k not in lookup for k in (s, t, e)):
            raise ValueError('相位边界需要满足：起始帧 < 摆动开始帧 < 结束帧，且位于视频范围内。')
        a, b, c = (lookup[k] for k in (s, t, e))
        if not valid[a:c + 1].all():
            raise ValueError('相位周期跨越缺失坐标，请先校正轨迹或调整边界。')
        if result and a < result[-1][2]:
            raise ValueError('相位周期必须按时间排序，且不能重叠。')
        result.append((a, b, c))
    return result


def add_phase_columns(work, cycles, settings, source, swing_regions=None):
    """Shared display/export layer; ALMA keeps its own event definitions."""
    work = work.copy()
    frames = work.source_frame.to_numpy()
    x = work.foot_x_relative_to_hip_px
    y = work.get('foot_height_relative_to_hip_px', work.hip_y_px_smooth - work.foot_y_px_smooth)
    vx, vy = derivative(x), derivative(y)
    phase = np.full(len(work), 'unclassified', dtype=object)
    ids = np.full(len(work), '', dtype=object)
    accepted = np.zeros(len(work), bool)
    stance, swing = ('retraction', 'protraction') if settings.protocol == 'suspended' else ('stance', 'swing')
    if 'velocity_phase_state' in work and source in ('velocity', 'manual'):
        phase[work.velocity_phase_state.eq(-1)] = stance
        phase[work.velocity_phase_state.eq(1)] = swing
    for row in cycles.itertuples():
        if not row.source_frame_start < row.source_frame_transition < row.source_frame_end:
            continue
        a = (frames >= row.source_frame_start) & (frames < row.source_frame_transition)
        b = (frames >= row.source_frame_transition) & (frames < row.source_frame_end)
        # Excluded cycles stay visible but explicitly carry their QC decision.
        phase[a], phase[b] = stance, swing
        ids[a | b] = row.cycle_id
        accepted[a | b] = bool(row.automatic_include)
    valid = np.isfinite(work[[f'{j}_{a}_px_smooth' for j in ('hip', 'knee', 'ankle', 'foot')
                             for a in ('x', 'y')]]).all(axis=1)
    if swing_regions is not None:
        # A manual selection replaces every automatic colour, including partial
        # cycles. Empty selection deliberately means no swing anywhere.
        phase[:] = stance
        for a, b in normalize_swing_regions(swing_regions, frames):
            phase[(frames >= a) & (frames <= b)] = swing
    phase[~valid] = 'unclassified'
    if 'swing_direction_sign' in work:
        sign = work.swing_direction_sign.to_numpy(float)
    else:
        sign = np.full(len(work), np.nan)
        for a, b in runs(np.isfinite(vx)):
            swing_v = vx[a:b][phase[a:b] == swing]
            if len(swing_v):
                sign[a:b] = 1 if np.median(swing_v) >= 0 else -1
            elif settings.direction != 'auto':
                sign[a:b] = 1 if settings.direction == 'positive' else -1
        work['swing_direction_sign'] = sign
    work['foot_forward_velocity_px_per_frame'] = vx * sign
    work['foot_vertical_velocity_px_per_frame'] = vy
    work['foot_speed_px_per_frame'] = np.hypot(vx, vy)
    for name in ('foot_forward_velocity', 'foot_speed'):
        per_frame = work[name + '_px_per_frame']
        work[name + '_px_s'] = per_frame * settings.capture_fps if settings.capture_fps else np.nan
        work[name + '_cm_s'] = work[name + '_px_s'] / settings.px_per_cm if settings.px_per_cm else np.nan
    work['phase'] = phase
    work['phase_cycle_id'] = ids
    work['phase_cycle_automatic_include'] = accepted
    work['phase_source'] = np.where(ids != '', 'manual', 'velocity') if source == 'manual' and 'velocity_phase_state' in work else source
    return work
