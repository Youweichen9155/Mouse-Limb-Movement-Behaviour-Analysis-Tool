"""Explicit coordinate schemas; never silently compress missing source frames."""
from pathlib import Path
import csv
import numpy as np
import pandas as pd

JOINTS = ('hip', 'knee', 'ankle', 'foot')
ALIASES = {'hip': ['hip'], 'knee': ['knee'], 'ankle': ['ankle'],
           'foot': ['foot', 'toe'], 'mtp': ['mtp'],
           'crest': ['crest', 'iliac crest', 'iliac_crest', 'iliac']}

def canonical_name(name, side):
    name = str(name).lower().strip()
    for joint, aliases in ALIASES.items():
        for alias in aliases:
            allowed = [alias] if side == 'unsided' else [alias + side, alias + '_' + side]
            if name in allowed:
                return joint
    return None

def validate_frames(table):
    frame = pd.to_numeric(table['source_frame'], errors='coerce').to_numpy(float)
    if not len(frame) or not np.isfinite(frame).all() or (frame < 0).any() or (frame != np.floor(frame)).any():
        raise ValueError('source_frame 必须是非负整数原始帧号。')
    if len(np.unique(frame)) != len(frame) or (np.diff(frame) <= 0).any():
        raise ValueError('原始帧号必须唯一且递增；请先纠正重复或乱序数据。')
    if frame[-1] - frame[0] > 2_000_000:
        raise ValueError('帧号跨度超过 200 万，请分段导入。')
    table = table.copy()
    table['source_frame'] = frame.astype(int)
    return table.set_index('source_frame').reindex(range(int(frame[0]), int(frame[-1]) + 1)).rename_axis('source_frame').reset_index()

def load_coordinates(path, side='unsided', likelihood=0.9):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = []
        reader = csv.reader(handle)
        for _ in range(3): rows.append(next(reader, []))
    first = rows[0]
    is_dlc = bool(first and (first[0].strip().lower() == 'scorer' or (len(rows[2]) > 3 and set(rows[2][1:]).issubset({'x','y','likelihood'}))))
    if is_dlc:
        data = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        if len(set(data.columns.get_level_values(0))) != 1:
            raise ValueError('请一次导入一个 DLC scorer。')
        result = pd.DataFrame({'source_frame': data.index.to_numpy()})
        found = set()
        for bp in dict.fromkeys(data.columns.get_level_values(1)):
            joint = canonical_name(bp, side)
            if joint is None:
                continue
            if joint in found:
                raise ValueError(f'关节 {joint} 有多个候选列；请保留一个身体侧别。')
            found.add(joint)
            sub = data.xs(bp, axis=1, level=1)
            sub.columns = sub.columns.get_level_values(-1)
            if not {'x', 'y', 'likelihood'}.issubset(sub.columns):
                raise ValueError(f'DLC 关节 {bp} 缺少 x/y/likelihood。')
            confidence = pd.to_numeric(sub['likelihood'], errors='coerce').to_numpy()
            for axis in ('x', 'y'):
                values = pd.to_numeric(sub[axis], errors='coerce').to_numpy(float)
                values[~np.isfinite(confidence) | (confidence < likelihood)] = np.nan
                result[f'{joint}_{axis}_px'] = values
            result[f'{joint}_likelihood'] = confidence
    else:
        result = pd.read_csv(path)
        if 'source_frame' not in result:
            raise ValueError('缺少 source_frame。旧版 Frame 为删帧后编号，不能还原时间轴；请重新追踪原视频。')
        for joint in (*JOINTS, 'mtp', 'crest'):
            for axis in ('x', 'y'):
                col = f'{joint}_{axis}_px'
                if col in result:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
                    detected = f'{joint}_detected'
                    if detected in result:
                        good = result[detected].astype(str).str.lower().isin(['true', '1', '1.0'])
                        result.loc[~good, col] = np.nan
                    conf = f'{joint}_likelihood'
                    if conf in result:
                        good = pd.to_numeric(result[conf], errors='coerce') >= likelihood
                        manual = f'{joint}_manual_corrected'
                        if manual in result:
                            good |= result[manual].astype(str).str.lower().isin(['true','1','1.0'])
                        result.loc[~good, col] = np.nan
    missing = [f'{j}_{a}_px' for j in JOINTS for a in ('x', 'y') if f'{j}_{a}_px' not in result]
    if missing:
        raise ValueError('缺少关节坐标：' + ', '.join(missing) + '。DLC 双侧数据需选择 L/R。')
    for col in result:
        if col.endswith('_px'):
            result.loc[~np.isfinite(result[col].to_numpy(float)), col] = np.nan
    result = validate_frames(result)
    for j in JOINTS:
        result[f'{j}_detected'] = result[[f'{j}_x_px', f'{j}_y_px']].notna().all(axis=1)
    return result, {'coordinate_format': 'DLC' if is_dlc else 'framewise', 'side': side,
                    'likelihood_threshold': likelihood}
