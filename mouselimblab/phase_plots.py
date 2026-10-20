"""Phase-coloured stick sequences and their matching velocity curves."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from .phases import runs

COLOURS = {'stance': '#111111', 'swing': '#ef2929',
           'retraction': '#111111', 'protraction': '#ef2929',
           'unclassified': '#b3bac4'}


def phase_legend(settings):
    labels = (('retraction', 'Retraction'), ('protraction', 'Protraction')) if settings.protocol == 'suspended' else (
        ('stance', 'Stance'), ('swing', 'Swing'))
    return [Line2D([], [], color=COLOURS[k], lw=1, label=v) for k, v in
            (*labels, ('unclassified', 'Unclassified'))]


def save(fig, folder, stem):
    try:
        fig.savefig(folder / (stem + '.pdf'))
        fig.savefig(folder / (stem + '.png'), dpi=600)
    finally:
        plt.close(fig)


def plot_phases(work, cycles, settings, folder, engine='local'):
    frames = work.source_frame.to_numpy()
    phase = work.phase.to_numpy()
    colours = [COLOURS[p] for p in phase]
    columns = ['source_frame', 'video_time_s', 'real_time_s', 'phase', 'phase_source',
               'phase_cycle_id', 'phase_cycle_automatic_include', 'swing_direction_sign',
               'foot_forward_velocity_px_per_frame', 'foot_vertical_velocity_px_per_frame',
               'foot_speed_px_per_frame', 'foot_forward_velocity_px_s', 'foot_speed_px_s',
               'foot_forward_velocity_cm_s', 'foot_speed_cm_s', 'any_marker_interpolated']
    work[columns].to_csv(folder / 'phase_frames.csv', index=False)
    cycles[['cycle_id', 'source_frame_start', 'source_frame_transition', 'source_frame_end',
            'automatic_include']].to_csv(folder / 'phase_events.csv', index=False)
    suffix, unit = ('cm_s', 'cm/s') if settings.capture_fps and settings.px_per_cm else (
        ('px_s', 'px/s') if settings.capture_fps else ('px_per_frame', 'px/frame'))
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True, layout='constrained')
    for ax, metric, title in zip(axes, ['foot_forward_velocity', 'foot_speed'],
                                 ['Forward velocity', 'Speed']):
        values = work[metric + '_' + suffix].to_numpy()
        ax.plot(frames, values, color='#a1a7af', lw=.6)
        for name in COLOURS:
            mask = phase == name
            ax.plot(frames, np.where(mask, values, np.nan), color=COLOURS[name], lw=.85)
            if name in ('swing', 'protraction'):
                for a, b in runs(mask):
                    ax.axvspan(frames[a], frames[b - 1] + 1, color=COLOURS[name], alpha=.08, lw=0)
        for row in cycles.itertuples():
            ax.axvline(row.source_frame_start, color='#777777', lw=.35, ls=':')
            ax.axvline(row.source_frame_transition, color=COLOURS['swing'], lw=.4, ls='--')
        ax.set_ylabel(f'{title} ({unit})')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].axhline(0, color='#8a929e', lw=.4)
    axes[0].legend(handles=phase_legend(settings), frameon=False, ncol=3, loc='upper right')
    axes[0].set_title('Foot relative to hip', fontsize=10, loc='left')
    axes[-1].set_xlabel('Original source frame')
    save(fig, folder, 'phase_velocity')

    joints = ('crest', 'hip', 'knee', 'ankle', 'mtp', 'foot') if engine == 'alma' else (
        'hip', 'knee', 'ankle', 'foot')
    points = np.stack([work[[f'{j}_x_px_smooth', f'{j}_y_px_smooth']].to_numpy() for j in joints], axis=1)
    # Translate each skeleton horizontally; never scale X and Y independently.
    points[:, :, 0] -= work.hip_x_px_smooth.to_numpy()[:, None]
    points[:, :, 1] *= -1
    finite = points[np.isfinite(points).all(axis=(1, 2))]
    spans = np.ptp(finite[:, :, 0], axis=1) if len(finite) else [1.]
    width = max(1., float(np.median(spans)))
    chunks = np.array_split(np.arange(len(work)), min(4, max(1, int(np.ceil(len(work) / 100)))))
    fig, axes = plt.subplots(len(chunks), 1, figsize=(8, 1.8 * len(chunks) + .5),
                             squeeze=False, layout='constrained')
    source_rows = []
    for row_number, (ax, indices) in enumerate(zip(axes[:, 0], chunks), 1):
        stride = max(1, int(np.ceil(len(indices) / 250)))
        selected = indices[::stride]
        selected = np.unique(np.r_[selected, indices[-1]])
        segments, line_colours = [], []
        start, end = frames[indices[0]], frames[indices[-1]]
        for i in selected:
            if not np.isfinite(points[i]).all():
                continue
            offset = (frames[i] - start) / max(1, end - start) * width * 4
            pose = points[i].copy()
            pose[:, 0] += offset
            segments.append(pose)
            line_colours.append(colours[i])
            for joint, (x, y) in zip(joints, pose):
                source_rows.append({'row': row_number, 'source_frame': frames[i], 'joint': joint,
                                    'phase': phase[i], 'display_x_px': x, 'display_y_px': y,
                                    'time_offset_px': offset, 'frame_stride': stride})
        ax.add_collection(LineCollection(segments, colors=line_colours, linewidths=.35))
        ax.autoscale()
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_axis_off()
        ax.set_title(f'Frames {start}–{end}', loc='left', fontsize=8, color='#657080')
    fig.legend(handles=phase_legend(settings), frameon=False, ncol=3, loc='outside upper center')
    save(fig, folder, 'limb_sequence')
    pd.DataFrame(source_rows, columns=['row', 'source_frame', 'joint', 'phase', 'display_x_px',
                                      'display_y_px', 'time_offset_px', 'frame_stride']).to_csv(
                                          folder / 'limb_sequence_points.csv', index=False)
