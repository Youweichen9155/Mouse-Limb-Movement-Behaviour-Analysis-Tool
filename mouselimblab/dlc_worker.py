"""Standalone worker executed by the user's isolated DeepLabCut Python.

Do not import this in the frozen application: it must not bundle PyTorch/DLC.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys
import traceback
import shutil
import hashlib
import uuid

os.environ.setdefault('DLClight', 'True')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

def choose_device(request, torch):
    device = request.get('device', 'auto')
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('此环境无法使用 Apple GPU，请选择 CPU。')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('此环境未检测到 CUDA。请安装对应 PyTorch GPU 版本，或选择 CPU。')
    if device not in ('cpu', 'mps', 'cuda:0'):
        raise ValueError('未知计算设备。')
    return device

def execute(request):
    import deeplabcut as dlc
    import torch
    import pandas as pd
    import numpy as np
    from deeplabcut.utils import auxiliaryfunctions as aux
    from dlc_project import BODY_PARTS, DLC_VERSION, load_project, save_json, fingerprint, validate_annotations
    from video import open_video, oriented_opencv_reads, COORDINATE_SYSTEM
    if dlc.__version__ != DLC_VERSION:
        raise ValueError(f'此版本已验证 DeepLabCut {DLC_VERSION}，检测到 {dlc.__version__}。请使用独立安装脚本。')
    device = choose_device(request, torch)
    print(f'DeepLabCut {dlc.__version__} / PyTorch {torch.__version__} / device={device}', flush=True)
    versions = {'deeplabcut': dlc.__version__, 'torch': torch.__version__, 'python': sys.version, 'device': device}
    action = request['action']
    if action == 'probe':
        return {**versions, 'cuda_available': torch.cuda.is_available(), 'mps_available': torch.backends.mps.is_available()}
    project = Path(request['project']).resolve()
    data = load_project(project)
    if data.get('coordinate_system') != COORDINATE_SYSTEM:
        for video in data['videos']:
            import cv2
            cap = open_video(video['path'])
            rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META);cap.release()
            if rotation % 360:
                raise ValueError('旧训练项目含旋转视频，标注坐标方向未记录。请创建新版项目并核对标注后训练；旧项目保持不变。')
    root = project.parent
    if action == 'prepare':
        rows = validate_annotations(data)
        net = request.get('net_type', 'resnet_50')
        if net not in ('resnet_50', 'resnet_101'):
            raise ValueError('支持的 PyTorch 网络为 ResNet-50 或 ResNet-101。')
        run = root / 'runs' / ('dataset_' + datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6])
        run.mkdir(parents=True)
        for name in ('training-datasets', 'dlc-models-pytorch', 'videos', 'labeled-data'):
            (run / name).mkdir()
        cfg, _ = aux.create_config_template()
        fraction = round(sum(r['split'] == 'train' for r in rows) / len(rows), 2)
        if not 0 < fraction < 1:
            raise ValueError('训练/测试比例过于失衡，请补充标注。')
        cfg.update({'Task': data['name'], 'scorer': data['scorer'], 'project_path': str(run),
                    'date': datetime.now().strftime('%b%d'), 'engine': 'pytorch', 'multianimalproject': False,
                    'bodyparts': list(BODY_PARTS), 'skeleton': [list(pair) for pair in zip(BODY_PARTS[:-1], BODY_PARTS[1:])],
                    'video_sets': {v['path']: {'crop': f'0, {v["width"]}, 0, {v["height"]}'} for v in data['videos']},
                    'default_net_type': net, 'default_augmenter': 'albumentations',
                    'cropping': False, 'start': 0, 'stop': 1, 'numframes2pick': 20, 'TrainingFraction': [fraction],
                    'iteration': 0, 'snapshotindex': -1, 'detector_snapshotindex': -1,
                    'batch_size': request.get('batch_size', 4), 'detector_batch_size': 1,
                    'x1': 0, 'x2': 640, 'y1': 0, 'y2': 480, 'corner2move2': [50, 50], 'move2corner': True,
                    'skeleton_color': 'black', 'pcutoff': request.get('pcutoff', 0.9),
                    'dotsize': 5, 'alphavalue': 0.7, 'colormap': 'rainbow'})
        config = run / 'config.yaml'
        aux.write_config(str(config), cfg)
        columns = pd.MultiIndex.from_product([[data['scorer']], BODY_PARTS, ['x', 'y']], names=['scorer', 'bodyparts', 'coords'])
        for video in data['videos']:
            subset = [r for r in rows if r['video_id'] == video['id']]
            if not subset:
                continue
            folder = run / 'labeled-data' / video['id']
            folder.mkdir()
            for row in subset:
                source = (root / row['image']).resolve()
                if not source.is_relative_to(root) or not source.is_file():
                    raise ValueError('标注图片路径无效。')
                shutil.copyfile(source, run / row['image'])
            index = pd.MultiIndex.from_tuples([tuple(r['image'].split('/')) for r in subset])
            values = [[xy for part in BODY_PARTS for xy in (r['points'][part] or [np.nan, np.nan])] for r in subset]
            frame = pd.DataFrame(values, index=index, columns=columns, dtype=float)
            frame.to_hdf(folder / f'CollectedData_{data["scorer"]}.h5', key='df_with_missing', mode='w')
            frame.to_csv(folder / f'CollectedData_{data["scorer"]}.csv')
        # DLC concatenates video annotations then sort_index(). Use precisely that
        # index to map the held-out animals to explicit train/test indices.
        merged_dir = run / aux.get_training_set_folder(cfg)
        merged_dir.mkdir(parents=True, exist_ok=True)
        from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
        merged = merge_annotateddatasets(cfg, merged_dir)
        role = {tuple(r['image'].split('/')): r['split'] for r in rows}
        train = [i for i, idx in enumerate(merged.index) if role[tuple(idx)] == 'train']
        test = [i for i, idx in enumerate(merged.index) if role[tuple(idx)] == 'test']
        dlc.create_training_dataset(str(config), Shuffles=[1], trainIndices=[train], testIndices=[test],
                                    net_type=net, augmenter_type='albumentations', engine=dlc.Engine.PYTORCH, userfeedback=False)
        manifest = {'fingerprint': fingerprint(data), 'versions': versions, 'net_type': net, 'train_indices': train,
                    'test_indices': test, 'rows': [{k: r[k] for k in ('image', 'animal_id', 'split', 'source_frame')} for r in rows],
                    'input_image_sha256': {r['image']: hashlib.sha256((run / r['image']).read_bytes()).hexdigest() for r in rows},
                    'training_status': 'not_started', 'evaluation_status': 'not_started'}
        save_json(run / 'dataset_manifest.json', manifest)
        data['active_run'] = str(config.relative_to(root))
        save_json(project, data)
        return {'config': str(config), 'train_frames': len(train), 'test_frames': len(test), **versions}
    if not data.get('active_run'):
        raise ValueError('请先完成标注并准备训练数据。')
    config = (root / data['active_run']).resolve()
    if not config.is_relative_to(root) or not config.is_file():
        raise ValueError('训练配置路径无效。')
    manifest_path = config.parent / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['fingerprint'] != fingerprint(data):
        raise ValueError('标注或分组已改变，请重新准备训练数据；旧模型仍保留。')
    # Relocating a project between Mac and Windows must not leave stale paths.
    cfg = aux.read_config(str(config))
    cfg['project_path'] = str(config.parent)
    aux.write_config(str(config), cfg)
    if action == 'train':
        epochs = int(request.get('epochs', 200))
        if not 1 <= epochs <= 10000:
            raise ValueError('训练轮数须为 1–10000。')
        resume = request.get('resume', False)
        snapshots = list(config.parent.rglob('snapshot-*.pt'))
        if snapshots and not resume:
            raise ValueError('该数据集已有训练快照，请选择“继续训练”，或重新准备新数据集。')
        snapshot = str(max(snapshots, key=lambda p: p.stat().st_mtime)) if snapshots and resume else None
        manifest.update(training_status='running', evaluation_status='not_started')
        save_json(manifest_path, manifest)
        dlc.train_network(str(config), shuffle=1, device=device, epochs=epochs,
                          batch_size=int(request.get('batch_size', 4)), save_epochs=max(1, min(10, epochs)),
                          max_snapshots_to_keep=5, display_iters=10, snapshot_path=snapshot)
        manifest['training_status'] = 'finished'
        save_json(manifest_path, manifest)
        return {'config': str(config), 'epochs': epochs, **versions}
    if action == 'evaluate':
        manifest['evaluation_status'] = 'running'
        save_json(manifest_path, manifest)
        dlc.evaluate_network(str(config), Shuffles=[1], plotting=False, device=device, pcutoff=request.get('pcutoff', .9))
        manifest['evaluation_status'] = 'finished'
        save_json(manifest_path, manifest)
        files = [str(p) for p in config.parent.rglob('*.csv') if 'evaluation' in str(p.parent)]
        return {'evaluation_files': files, 'config': str(config), **versions}
    if action == 'predict':
        if manifest.get('evaluation_status') != 'finished':
            raise ValueError('请先运行模型评估，查看测试误差后再预测视频。')
        videos = [str(Path(v).resolve()) for v in request.get('videos', [])]
        if not videos or not all(Path(v).is_file() for v in videos):
            raise ValueError('请选择要预测的视频。')
        outputs = []
        destination = root / 'predictions' / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6])
        for i, video in enumerate(videos):
            target = destination / f'{i:03d}_{Path(video).stem}'
            target.mkdir(parents=True)
            with oriented_opencv_reads():
                dlc.analyze_videos(str(config), [video], shuffle=1, save_as_csv=True, destfolder=str(target),
                                   device=device, batch_size=int(request.get('batch_size', 4)), cropping=None)
            files = list(target.glob('*.csv'))
            if len(files) != 1:
                raise RuntimeError(f'没有找到唯一的六点坐标 CSV：{target}')
            # Reject incomplete/misordered frame outputs before handing to ALMA.
            frame = pd.read_csv(files[0], header=[0, 1, 2], index_col=0)
            import cv2
            cap = open_video(video)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
            if len(frame) != count or not np.array_equal(frame.index.to_numpy(), np.arange(count)):
                raise ValueError('预测结果帧数或帧顺序与原视频不一致，未送入运动学分析。')
            required = {(part, coord) for part in BODY_PARTS for coord in ('x', 'y', 'likelihood')}
            if not required.issubset({(a[1], a[2]) for a in frame.columns}):
                raise ValueError('预测文件缺少六点坐标或置信度。')
            outputs.append({'video': video, 'csv': str(files[0]), 'source_frames': count, 'coordinate_system':COORDINATE_SYSTEM})
        save_json(destination / 'prediction_manifest.json', {'config': str(config), 'outputs': outputs, **versions})
        return {'outputs': outputs, **versions}
    raise ValueError('未知 DeepLabCut 操作。')

def main():
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding='utf-8'))
    try:
        result = {'ok': True, 'action': request['action'], 'result': execute(request)}
        code = 0
    except Exception as error:
        traceback.print_exc()
        # Preserve the failure state alongside the model without masking the
        # original error; an incomplete run must never look like completion.
        try:
            from dlc_project import load_project, save_json
            project = Path(request['project']).resolve()
            data = load_project(project)
            manifest_file = (project.parent / data['active_run']).parent / 'dataset_manifest.json'
            manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
            key = {'train': 'training_status', 'evaluate': 'evaluation_status'}.get(request['action'])
            if key and manifest.get(key) == 'running':
                manifest[key] = 'failed';save_json(manifest_file, manifest)
        except (KeyError, ValueError, OSError):
            pass
        result = {'ok': False, 'action': request.get('action'), 'error': str(error), 'traceback': traceback.format_exc()}
        code = 1
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return code

if __name__ == '__main__':
    sys.exit(main())
