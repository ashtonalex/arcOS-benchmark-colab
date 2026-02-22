"""Converts raw Action Genome annotations to SceneGraphBuilder input format.

Action Genome stores relations as object attributes:
    obj["contacting_relationship"] = ["holding", "touching"]
    obj["attention_relationship"] = ["looking_at"]

This module pivots them into pairwise (person, object, predicate) tuples
and structures frames for SceneGraphBuilder consumption.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def load_ag_annotations(pkl_path: str) -> Dict:
    """Load Action Genome object_bbox_and_relationship.pkl.

    Args:
        pkl_path: Path to the pickle file.

    Returns:
        Raw AG annotations dict keyed by frame filename.
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def convert_video(
    video_id: str,
    raw_annotations: Dict,
    frame_sample_rate: int = 3,
    class_map: Optional[Dict[int, str]] = None,
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

    Args:
        video_id: Video identifier (e.g., "001YG").
        raw_annotations: Full AG annotation dict (frame_filename -> object list).
        frame_sample_rate: Sample every N-th frame (1 = keep all).
        class_map: Optional mapping from class index to class name.

    Returns:
        Annotation dict ready for SceneGraphBuilder.build().
    """
    # Collect frames belonging to this video
    prefix = video_id + "/"
    video_frames = sorted(
        [k for k in raw_annotations.keys() if k.startswith(prefix)]
    )

    if not video_frames:
        # Try without slash prefix (some AG versions use different key format)
        video_frames = sorted(
            [k for k in raw_annotations.keys() if video_id in k]
        )

    # Sample frames
    if frame_sample_rate > 1:
        video_frames = video_frames[::frame_sample_rate]

    frames = []
    global_obj_counter = 0

    for frame_idx, frame_key in enumerate(video_frames):
        objects_raw = raw_annotations[frame_key]
        frame_objects = []
        frame_relations = []

        # Object ID 0 is always the person
        person_id = global_obj_counter
        frame_objects.append({
            "object_id": person_id,
            "class": "person",
            "bbox": None,
        })
        global_obj_counter += 1

        for obj in objects_raw:
            obj_id = global_obj_counter
            global_obj_counter += 1

            # Resolve class name
            cls_name = obj.get("class", "unknown")
            if isinstance(cls_name, int) and class_map:
                cls_name = class_map.get(cls_name, f"object_{cls_name}")

            bbox = obj.get("bbox", None)
            frame_objects.append({
                "object_id": obj_id,
                "class": str(cls_name),
                "bbox": bbox,
            })

            # Pivot relation attributes to pairwise tuples
            for rel_type in ["contacting_relationship", "attention_relationship", "spatial_relationship"]:
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
        class_map: Optional class index → name mapping.

    Returns:
        Dict mapping video_id to SceneGraphBuilder-ready annotations.
    """
    results = {}
    for vid in sorted(video_ids):
        converted = convert_video(vid, raw_annotations, frame_sample_rate, class_map)
        if converted["frames"]:  # Only include videos with frames
            results[vid] = converted
    return results
