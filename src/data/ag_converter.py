"""Converts raw Action Genome annotations to SceneGraphBuilder input format.

Action Genome stores relations as object attributes:
    obj["contacting_relationship"] = ["holding", "touching"]
    obj["attention_relationship"] = ["looking_at"]

This module pivots them into pairwise (person, object, predicate) tuples
and structures frames for SceneGraphBuilder consumption.

Object IDs are tracked across frames using IoU-based Hungarian matching
so that SceneGraphBuilder can create temporal edges linking the same
physical object across consecutive frames.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_ag_annotations(pkl_path: str) -> Dict:
    """Load Action Genome object_bbox_and_relationship.pkl.

    Args:
        pkl_path: Path to the pickle file.

    Returns:
        Raw AG annotations dict keyed by frame filename.
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _compute_iou(bbox_a, bbox_b) -> float:
    """Compute IoU between two bounding boxes.

    Accepts any [c0, c1, c2, c3] format where (c0,c1) is one corner
    and (c2,c3) is the opposite corner.  Works for both [x1,y1,x2,y2]
    and [y1,x1,y2,x2] since IoU is symmetric over axes.

    Returns 0.0 if either bbox is None or degenerate.
    """
    if bbox_a is None or bbox_b is None:
        return 0.0
    try:
        a = [float(c) for c in bbox_a[:4]]
        b = [float(c) for c in bbox_b[:4]]
    except (TypeError, ValueError, IndexError):
        return 0.0

    inter_0 = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_1 = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter_area = inter_0 * inter_1

    area_a = abs(a[2] - a[0]) * abs(a[3] - a[1])
    area_b = abs(b[2] - b[0]) * abs(b[3] - b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _match_objects_across_frames(
    prev_by_class: Dict[str, List[Tuple[int, Any]]],
    curr_by_class: Dict[str, List[Tuple[int, Any]]],
    next_obj_id: int,
    iou_threshold: float = 0.3,
) -> Tuple[Dict[int, int], Dict[str, List[Tuple[int, Any]]], int]:
    """Match current-frame objects to previous-frame objects using IoU.

    For each class present in both frames, builds a cost matrix of
    negative IoU and runs the Hungarian algorithm to find the optimal
    one-to-one assignment.  Unmatched objects receive new IDs.

    Args:
        prev_by_class: Previous frame's objects grouped by class.
            class -> [(object_id, bbox), ...]
        curr_by_class: Current frame's objects grouped by class.
            class -> [(raw_index, bbox), ...]
        next_obj_id: Next available object ID.
        iou_threshold: Minimum IoU to accept a match.

    Returns:
        (assignments, new_prev_by_class, updated_next_obj_id)
        - assignments: raw_index -> assigned object_id
        - new_prev_by_class: updated dict for next iteration
        - updated_next_obj_id: incremented counter
    """
    assignments: Dict[int, int] = {}
    new_prev: Dict[str, List[Tuple[int, Any]]] = {}

    for cls_name, curr_items in curr_by_class.items():
        prev_items = prev_by_class.get(cls_name, [])

        if not prev_items:
            # No predecessors — assign fresh IDs
            cls_entries = []
            for raw_idx, bbox in curr_items:
                assignments[raw_idx] = next_obj_id
                cls_entries.append((next_obj_id, bbox))
                next_obj_id += 1
            new_prev[cls_name] = cls_entries
            continue

        n_prev = len(prev_items)
        n_curr = len(curr_items)

        # Check whether bboxes are available on both sides
        has_prev_bbox = any(bb is not None for _, bb in prev_items)
        has_curr_bbox = any(bb is not None for _, bb in curr_items)

        cls_entries = []

        if has_prev_bbox and has_curr_bbox:
            # Build IoU cost matrix (negative because we minimise)
            cost = np.zeros((n_curr, n_prev), dtype=np.float64)
            for ci, (_, c_bbox) in enumerate(curr_items):
                for pi, (_, p_bbox) in enumerate(prev_items):
                    cost[ci, pi] = -_compute_iou(c_bbox, p_bbox)

            row_ind, col_ind = linear_sum_assignment(cost)

            matched_curr: set = set()
            for r, c in zip(row_ind, col_ind):
                iou_val = -cost[r, c]
                if iou_val >= iou_threshold:
                    prev_id = prev_items[c][0]
                    raw_idx, bbox = curr_items[r]
                    assignments[raw_idx] = prev_id
                    cls_entries.append((prev_id, bbox))
                    matched_curr.add(r)

            # Unmatched current objects → new IDs
            for ci, (raw_idx, bbox) in enumerate(curr_items):
                if ci not in matched_curr:
                    assignments[raw_idx] = next_obj_id
                    cls_entries.append((next_obj_id, bbox))
                    next_obj_id += 1
        else:
            # No bboxes available — fall back to order-based 1-to-1 matching
            for ci, (raw_idx, bbox) in enumerate(curr_items):
                if ci < n_prev:
                    prev_id = prev_items[ci][0]
                    assignments[raw_idx] = prev_id
                    cls_entries.append((prev_id, bbox))
                else:
                    assignments[raw_idx] = next_obj_id
                    cls_entries.append((next_obj_id, bbox))
                    next_obj_id += 1

        new_prev[cls_name] = cls_entries

    return assignments, new_prev, next_obj_id


def convert_video(
    video_id: str,
    raw_annotations: Dict,
    frame_sample_rate: int = 3,
    class_map: Optional[Dict[int, str]] = None,
    iou_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Convert AG annotations for one video into SceneGraphBuilder format.

    AG raw format (per frame):
        [{"class": "cup", "bbox": [...],
          "contacting_relationship": ["holding"],
          "attention_relationship": ["looking_at"],
          "spatial_relationship": ["in_front_of"],
          "metadata": {"tag": ..., "set": ...}}, ...]

    SceneGraphBuilder expects:
        {"video_id": str, "frames": [
            {"frame_id": int, "objects": [{object_id, class, bbox}],
             "relations": [{subject_id, object_id, predicate}]}
        ]}

    In AG, the implicit subject of all relations is "person" (index 0).
    Objects are tracked across frames via IoU-based Hungarian matching
    so that the same physical object keeps a consistent object_id.

    Args:
        video_id: Video identifier (e.g., "001YG").
        raw_annotations: Full AG annotation dict (frame_filename -> object list).
        frame_sample_rate: Sample every N-th frame (1 = keep all).
        class_map: Optional mapping from class index to class name.
        iou_threshold: Min IoU for cross-frame matching (default 0.3).

    Returns:
        Annotation dict ready for SceneGraphBuilder.build().
    """
    # Collect frames belonging to this video
    prefix = video_id + ".mp4/"
    video_frames = sorted(
        [k for k in raw_annotations.keys() if k.startswith(prefix)]
    )

    if not video_frames:
        # Fallback: try without .mp4 extension
        prefix_bare = video_id + "/"
        video_frames = sorted(
            [k for k in raw_annotations.keys() if k.startswith(prefix_bare)]
        )

    # Sample frames
    if frame_sample_rate > 1:
        video_frames = video_frames[::frame_sample_rate]

    frames = []
    # Person always gets object_id 0 (exactly 1 person per frame in AG)
    person_id = 0
    next_obj_id = 1  # next available ID for non-person objects

    # Previous frame's objects for IoU matching (class -> [(obj_id, bbox)])
    prev_by_class: Dict[str, List[Tuple[int, Any]]] = {}

    for frame_idx, frame_key in enumerate(video_frames):
        objects_raw = raw_annotations[frame_key]
        frame_objects = []
        frame_relations = []

        # Person node (constant ID across all frames)
        frame_objects.append({
            "object_id": person_id,
            "class": "person",
            "bbox": None,
        })

        # ---------- group current frame's objects by class ----------
        curr_by_class: Dict[str, List[Tuple[int, Any]]] = {}
        resolved_classes: Dict[int, str] = {}  # raw_index -> class name

        for i, obj in enumerate(objects_raw):
            cls_name = obj.get("class", "unknown")
            if isinstance(cls_name, int) and class_map:
                cls_name = class_map.get(cls_name, f"object_{cls_name}")
            cls_name = str(cls_name)
            resolved_classes[i] = cls_name
            bbox = obj.get("bbox", None)
            curr_by_class.setdefault(cls_name, []).append((i, bbox))

        # ---------- match to previous frame ----------
        assignments, prev_by_class, next_obj_id = _match_objects_across_frames(
            prev_by_class, curr_by_class, next_obj_id, iou_threshold
        )

        # ---------- build frame output ----------
        for i, obj in enumerate(objects_raw):
            obj_id = assignments[i]
            cls_name = resolved_classes[i]
            bbox = obj.get("bbox", None)

            frame_objects.append({
                "object_id": obj_id,
                "class": cls_name,
                "bbox": bbox,
            })

            # Pivot relation attributes to pairwise tuples
            for rel_type in [
                "contacting_relationship",
                "attention_relationship",
                "spatial_relationship",
            ]:
                rels = obj.get(rel_type, [])
                if rels is None:
                    continue
                for pred in rels:
                    frame_relations.append({
                        "subject_id": person_id,
                        "object_id": obj_id,
                        "predicate": str(pred),
                    })

        frames.append({
            "frame_id": frame_idx,
            "objects": frame_objects,
            "relations": frame_relations,
        })

    return {
        "video_id": video_id,
        "frames": frames,
    }


def convert_all(
    raw_annotations: Dict,
    video_ids: Set[str],
    frame_sample_rate: int = 3,
    class_map: Optional[Dict[int, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Batch convert AG annotations for multiple videos.

    Args:
        raw_annotations: Full AG annotation dict.
        video_ids: Set of video IDs to convert.
        frame_sample_rate: Sample every N-th frame.
        class_map: Optional class index -> name mapping.

    Returns:
        Dict mapping video_id to SceneGraphBuilder-ready annotations.
    """
    results = {}
    for vid in sorted(video_ids):
        converted = convert_video(vid, raw_annotations, frame_sample_rate, class_map)
        if converted["frames"]:  # Only include videos with frames
            results[vid] = converted
    return results
