"""Native annotation workspace; DeepLabCut itself runs in a separate interpreter."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
import os
import uuid
import cv2
import numpy as np
if __package__:
    from .video import open_video, COORDINATE_SYSTEM
else:  # Also imported by the standalone external worker.
    from video import open_video, COORDINATE_SYSTEM

BODY_PARTS = ('iliac_crest', 'hip', 'knee', 'ankle', 'mtp', 'toe')
BODY_LABELS = ('髂嵴', '髋', '膝', '踝', '跖趾关节', '趾尖')
DLC_VERSION = '3.0.2'

def save_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def load_project(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format') != 'mouselimblab.dlc.v1':
        raise ValueError('请选择本程序生成的 dlc_project.json。')
    return data

def fingerprint(data):
    values = {k: data[k] for k in ('videos', 'frames', 'labels', 'scorer', 'bodyparts')}
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def create_project(parent, name, scorer, videos, frames_per_video=20, progress=lambda _: None, cancelled=lambda: False):
    """Copy sampled pixels, never move, alter or copy whole source videos."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or not re.fullmatch(r'[A-Za-z0-9_-]+', scorer):
        raise ValueError('项目名和标注者请使用英文字母、数字、下划线或短横线。')
    if not videos:
        raise ValueError('请先添加视频。')
    if not 2 <= frames_per_video <= 1000:
        raise ValueError('每段视频抽帧数须为 2–1000。')
    stems = [Path(v['path']).stem.casefold() for v in videos]
    if len(set(stems)) != len(stems):
        raise ValueError('视频文件名不能重复（包括不同目录下的同名视频）。请先给副本设置不同名称。')
    root = Path(parent).expanduser().resolve() / f'{name}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
    root.mkdir(parents=True, exist_ok=False)
    data = {'format': 'mouselimblab.dlc.v1', 'name': name, 'scorer': scorer,
            'bodyparts': list(BODY_PARTS), 'videos': [], 'frames': [], 'labels': {},
            'created_utc': datetime.now(timezone.utc).isoformat(), 'sampling': 'uniform_original_frames',
            'coordinate_system': COORDINATE_SYSTEM}
    path = root / 'dlc_project.json'
    save_json(path, data)
    for v in videos:
        if cancelled():
            raise InterruptedError('已取消抽帧，已生成的项目和画面保留。')
        source = Path(v['path']).expanduser().resolve()
        cap = open_video(source)
        try:
            if not cap.isOpened():
                raise ValueError(f'无法读取视频：{source}')
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 2:
                raise ValueError(f'视频不足两帧：{source.name}')
            entry = {**v, 'path': str(source), 'id': source.stem, 'source_frames': total,
                     'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     'playback_fps': float(cap.get(cv2.CAP_PROP_FPS)),
                     'orientation_degrees': float(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
                     'source_bytes': source.stat().st_size, 'source_mtime_ns': source.stat().st_mtime_ns}
            data['videos'].append(entry)
            folder = root / 'labeled-data' / source.stem
            folder.mkdir(parents=True)
            for number in np.unique(np.linspace(0, total - 1, min(total, frames_per_video)).astype(int)):
                if cancelled():
                    raise InterruptedError('已取消抽帧。')
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(number))
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'{source.name} 第 {number} 帧无法解码。')
                target = folder / f'img{number:08d}.png'
                ok, encoded = cv2.imencode('.png', image)
                if not ok:
                    raise ValueError('保存抽帧图像失败。')
                target.write_bytes(encoded.tobytes())  # Unicode paths also work on Windows.
                rel = target.relative_to(root).as_posix()
                data['frames'].append({'image': rel, 'video_id': source.stem, 'source_frame': int(number),
                                       'width': image.shape[1], 'height': image.shape[0]})
                save_json(path, data)
            progress(f'{source.name}：已抽取 {min(total, frames_per_video)} 帧。')
        finally:
            cap.release()
    return path

def validate_annotations(data):
    """Return completed frame rows; split by animal, never adjacent frames."""
    animals = {}
    videos = {v['id']: v for v in data['videos']}
    for v in videos.values():
        animal = v.get('animal_id', '').strip()
        role = v.get('split')
        if not animal or role not in ('train', 'test'):
            raise ValueError('每段视频都需要动物编号和训练/测试分组。')
        if animal in animals and animals[animal] != role:
            raise ValueError(f'动物 {animal} 同时出现在训练集和测试集，请按动物分组。')
        animals[animal] = role
    completed = []
    coverage = {role: {p: 0 for p in BODY_PARTS} for role in ('train', 'test')}
    for frame in sorted(data['frames'], key=lambda f: tuple(f['image'].split('/'))):
        label = data['labels'].get(frame['image'], {})
        if not label.get('complete'):
            continue
        points = label.get('points', {})
        if set(points) != set(BODY_PARTS):
            raise ValueError('完成帧必须逐点标注或明确选择“不可见”。')
        visible = 0
        role = videos[frame['video_id']]['split']
        for part, point in points.items():
            if point is None:
                continue
            if len(point) != 2 or not np.isfinite(point).all() or not (0 <= point[0] < frame['width'] and 0 <= point[1] < frame['height']):
                raise ValueError(f'标注坐标超出画面：{frame["image"]} / {part}')
            visible += 1
            coverage[role][part] += 1
        if visible < 2:
            raise ValueError('完成帧至少需要两个可见关节点；请跳过无法辨认的画面。')
        completed.append({**frame, 'points': points, 'split': role, 'animal_id': videos[frame['video_id']]['animal_id']})
    for role, counts in coverage.items():
        missing = [part for part, count in counts.items() if count == 0]
        if missing:
            raise ValueError(f'{"训练" if role == "train" else "测试"}集缺少已完成的关节点标注：{", ".join(missing)}')
    return completed

def worker_script():
    return Path(__file__).resolve().parent / 'dlc_worker.py'

def child_environment():
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORM',
                'DYLD_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH'):
        env.pop(key, None)
    if 'LD_LIBRARY_PATH_ORIG' in env:
        env['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH_ORIG']
    else:
        env.pop('LD_LIBRARY_PATH', None)
    env.update({'DLClight': 'True', 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTORCH_ENABLE_MPS_FALLBACK': '1'})
    return env

def default_python():
    guesses = [Path.home() / 'MouseLimbLab_DLC' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python'),
               Path(__file__).resolve().parent.parent / '.dlc-venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')]
    return next((str(p) for p in guesses if p.is_file()), '')
