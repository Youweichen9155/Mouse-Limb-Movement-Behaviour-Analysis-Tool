"""Adapted from the project analyze_video_v2.py; provenance in audit/."""
from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from .video import open_video, COORDINATE_SYSTEM
JOINTS = ("hip", "knee", "ankle", "foot")

def circular_hue_mean(hue: np.ndarray) -> float:
    angle = np.asarray(hue, dtype=float) * 2.0 * np.pi / 180.0
    return float(
        (np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()) % (2.0 * np.pi))
        * 180.0
        / (2.0 * np.pi)
    )

def hue_distance(hue: np.ndarray, reference: float) -> np.ndarray:
    return np.abs((np.asarray(hue, dtype=float) - reference + 90.0) % 180.0 - 90.0)

def colour_reference(hsv: np.ndarray, point: tuple[float, float], radius: int = 12) -> tuple[float, float, float]:
    x, y = (int(round(v)) for v in point)
    y0, y1 = max(0, y - radius), min(hsv.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(hsv.shape[1], x + radius + 1)
    yy, xx = np.ogrid[y0 - y : y1 - y, x0 - x : x1 - x]
    disk = (xx * xx + yy * yy) <= radius * radius
    pixels = hsv[y0:y1, x0:x1][disk]
    saturation_cut = np.quantile(pixels[:, 1], 0.55)
    selected = pixels[pixels[:, 1] >= saturation_cut]
    return (
        circular_hue_mean(selected[:, 0]),
        float(np.median(selected[:, 1])),
        float(np.median(selected[:, 2])),
    )

def locate_marker(
    hsv: np.ndarray,
    predicted: np.ndarray,
    radius: int,
    reference: tuple[float, float, float],
    anatomical_anchor: np.ndarray | None = None,
    expected_segment_length: float | None = None,
    wide: bool = False,
) -> tuple[np.ndarray, float, float] | None:
    """Return centre, mask area and distance for the best local colour blob."""
    h0, s0, v0 = reference
    x, y = (int(round(v)) for v in predicted)
    height, width = hsv.shape[:2]
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    roi = hsv[y0:y1, x0:x1]
    hue_width = (18 if s0 < 100 else 14 if s0 < 170 else 11) + (6 if wide else 0)
    saturation_min = max(18.0, s0 - (130.0 if wide else 100.0))
    value_min = max(20.0, v0 - (140.0 if wide else 110.0))
    mask = (
        (hue_distance(roi[:, :, 0], h0) <= hue_width)
        & (roi[:, :, 1] >= saturation_min)
        & (roi[:, :, 2] >= value_min)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    best: tuple[float, np.ndarray, float, float] | None = None
    for label in range(1, n_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 5 or area > 0.35 * mask.size:
            continue
        centre = np.asarray(centroids[label], dtype=float) + np.array([x0, y0], dtype=float)
        distance = float(np.linalg.norm(centre - predicted))
        if distance > 0.95 * radius:
            continue
        score = distance - 2.5 * math.sqrt(area)
        if anatomical_anchor is not None and expected_segment_length is not None:
            observed_length = float(np.linalg.norm(centre - anatomical_anchor))
            # Segment length is stable in a fixed 2-D view.  Keep this as a
            # graded penalty rather than a hard rejection because genuine
            # out-of-plane motion can shorten the projected distal segment.
            score += 1.5 * abs(observed_length - expected_segment_length)
        candidate = (score, centre, area, distance)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return None
    return best[1], best[2], best[3]

def track_video(video_path: Path, config, progress=None, cancelled=None) -> tuple[pd.DataFrame, dict[str, object]]:
    cap = open_video(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read first frame of {video_path}")
    if config.seed_frame != 0:
        cap.release()
        raise NotImplementedError("The current batch tracker expects first-frame seeds")
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("视频帧率无效，请先转换为固定帧率视频。")
    for xy in config.seeds.values():
        if len(xy) != 2 or not (0 <= xy[0] < width and 0 <= xy[1] < height):
            cap.release()
            raise ValueError("初始标记点必须在视频画面内。")
    first_hsv = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    references = {joint: colour_reference(first_hsv, config.seeds[joint]) for joint in JOINTS}
    previous = {joint: np.asarray(config.seeds[joint], dtype=float) for joint in JOINTS}
    expected_lengths = {
        joint: float(
            np.linalg.norm(
                np.asarray(config.seeds[joint], dtype=float)
                - np.asarray(config.seeds[JOINTS[index - 1]], dtype=float)
            )
        )
        for index, joint in enumerate(JOINTS)
        if index > 0
    }
    velocity = {joint: np.zeros(2, dtype=float) for joint in JOINTS}
    misses = {joint: 0 for joint in JOINTS}
    rows: list[dict[str, object]] = []
    frame_index = 0
    frame = first
    while ok:
        if cancelled and cancelled():
            cap.release()
            raise InterruptedError("已取消追踪")
        if progress and frame_index % 30 == 0:
            progress(frame_index, frame_count_meta)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        row: dict[str, object] = {
            "clip_id": config.clip_id,
            "protocol": config.protocol,
            "source_frame": frame_index,
            "time_s": frame_index / fps,
        }
        current_points: dict[str, np.ndarray] = {}
        for joint_index, joint in enumerate(JOINTS):
            if frame_index == 0:
                point = previous[joint].copy()
                area = float("nan")
                distance = 0.0
                detected = True
            else:
                predicted = previous[joint] + velocity[joint]
                search_radius = min(220, 70 + 15 * misses[joint])
                anchor = (
                    current_points.get(JOINTS[joint_index - 1], previous[JOINTS[joint_index - 1]])
                    if joint_index > 0
                    else None
                )
                expected_length = expected_lengths.get(joint)
                result = locate_marker(
                    hsv,
                    predicted,
                    search_radius,
                    references[joint],
                    anatomical_anchor=anchor,
                    expected_segment_length=expected_length,
                    wide=False,
                )
                if result is None:
                    result = locate_marker(
                        hsv,
                        predicted,
                        min(260, search_radius + 50),
                        references[joint],
                        anatomical_anchor=anchor,
                        expected_segment_length=expected_length,
                        wide=True,
                    )
                detected = result is not None
                if detected:
                    point, area, distance = result
                    step = point - previous[joint]
                    if np.linalg.norm(step) > 80 and misses[joint] < 2:
                        detected = False
                    else:
                        velocity[joint] = 0.65 * velocity[joint] + 0.35 * step
                        previous[joint] = point
                        misses[joint] = 0
                if not detected:
                    point = np.array([np.nan, np.nan])
                    area = float("nan")
                    distance = float("nan")
                    velocity[joint] *= 0.8
                    misses[joint] += 1
            row[f"{joint}_x_px"] = float(point[0])
            row[f"{joint}_y_px"] = float(point[1])
            row[f"{joint}_detected"] = bool(detected)
            row[f"{joint}_mask_area_px"] = area
            row[f"{joint}_prediction_distance_px"] = distance
            if detected:
                current_points[joint] = np.asarray(point, dtype=float)
        rows.append(row)
        ok, frame = cap.read()
        frame_index += 1
    cap.release()
    table = pd.DataFrame(rows)
    if frame_count_meta > len(table) + 1:
        raise ValueError(f"视频读取提前停止：{len(table)}/{frame_count_meta} 帧，请检查编码或文件。")
    metadata: dict[str, object] = {
        "fps_metadata": fps,
        "frame_count_metadata": frame_count_meta,
        "frame_count_read": len(table),
        "width_px": width,
        "height_px": height,
        "coordinate_system": COORDINATE_SYSTEM,
        "colour_references_hsv": {joint: list(references[joint]) for joint in JOINTS},
    }
    return table, metadata

def interpolate_short_gaps(values: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    result = values.copy()
    interpolated = np.zeros(len(values), dtype=bool)
    missing = ~np.isfinite(values)
    index = 0
    while index < len(values):
        if not missing[index]:
            index += 1
            continue
        end = index
        while end < len(values) and missing[end]:
            end += 1
        length = end - index
        if index > 0 and end < len(values) and length <= max_gap:
            result[index:end] = np.interp(
                np.arange(index, end),
                [index - 1, end],
                [result[index - 1], result[end]],
            )
            interpolated[index:end] = True
        index = end
    return result, interpolated

def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.nanmedian(values))
    return 1.4826 * float(np.nanmedian(np.abs(values - median)))

def robust_cv(values: np.ndarray) -> float:
    median = float(np.nanmedian(values))
    if not np.isfinite(median) or median <= 0:
        return float("nan")
    return 100.0 * robust_scale(values) / median

def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    numerator = np.einsum("ij,ij->i", ba, bc)
    denominator = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    cosine = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def find_candidate_cycles(
    x_relative: np.ndarray,
    fps: float,
    clearance_hint: np.ndarray | None,
) -> tuple[list[tuple[int, int, int]], int, dict[str, float]]:
    dx = np.gradient(x_relative)
    dead = max(1e-8, 0.10 * float(np.nanquantile(np.abs(dx), 0.90)))
    if clearance_hint is not None:
        pos = clearance_hint[dx > dead]
        neg = clearance_hint[dx < -dead]
        pos_level = float(np.nanmedian(pos)) if len(pos) else float("nan")
        neg_level = float(np.nanmedian(neg)) if len(neg) else float("nan")
        if np.isfinite(pos_level) and np.isfinite(neg_level):
            swing_sign = 1 if pos_level >= neg_level else -1
        else:
            swing_sign = 1 if np.nanquantile(dx, 0.99) >= abs(np.nanquantile(dx, 0.01)) else -1
    else:
        swing_sign = 1 if np.nanquantile(dx, 0.99) >= abs(np.nanquantile(dx, 0.01)) else -1
    oriented = swing_sign * x_relative
    robust_range = float(np.nanquantile(oriented, 0.95) - np.nanquantile(oriented, 0.05))
    distance = max(3, int(round(0.15 * fps)))
    prominence = max(1e-8, 0.12 * robust_range)
    peaks, _ = find_peaks(oriented, prominence=prominence, distance=distance)
    if len(peaks) < 3:
        peaks, _ = find_peaks(oriented, prominence=max(1e-8, 0.07 * robust_range), distance=distance)
    raw: list[tuple[int, int, int]] = []
    for start, end in zip(peaks[:-1], peaks[1:]):
        if end - start < 10:
            continue
        toeoff = int(start + np.nanargmin(oriented[start : end + 1]))
        if toeoff - start < 3 or end - toeoff < 3:
            continue
        amplitude = min(oriented[start] - oriented[toeoff], oriented[end] - oriented[toeoff])
        if amplitude < 0.10 * robust_range:
            continue
        raw.append((int(start), toeoff, int(end)))
    if raw:
        lengths = np.asarray([end - start for start, _, end in raw], dtype=float)
        median_length = float(np.nanmedian(lengths))
        raw = [
            event
            for event in raw
            if 0.45 * median_length <= event[2] - event[0] <= 2.25 * median_length
        ]
    return raw, swing_sign, {
        "velocity_dead_band_px_per_frame": dead,
        "x_relative_robust_range_px": robust_range,
        "peak_prominence_px": prominence,
        "peak_distance_frames": float(distance),
    }

def consecutive_true_max(values: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
