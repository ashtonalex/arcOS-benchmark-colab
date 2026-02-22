"""AGQA dataset loader for video QA benchmarking."""

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import random
from src.config import BenchmarkConfig


# AGQA 2.0 balanced split URLs (Stanford hosted)
AGQA_URLS = {
    "train": "https://storage.googleapis.com/agqa/AGQA2/Balanced/Train_Balanced.json",
    "val": "https://storage.googleapis.com/agqa/AGQA2/Balanced/Val_Balanced.json",
    "test": "https://storage.googleapis.com/agqa/AGQA2/Balanced/Test_Balanced.json",
}


class AGQALoader:
    """Loads and parses AGQA QA pairs."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.subset_size = config.agqa_subset_size

    def parse_sample(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "question": raw["question"],
            "answer": raw["answer"],
            "video_id": raw["video_id"],
            "program": raw.get("program", []),
        }

    def get_unique_video_ids(self, samples: List[Dict[str, Any]]) -> Set[str]:
        return {s["video_id"] for s in samples}

    def subsample(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(samples) <= self.subset_size:
            return samples
        rng = random.Random(self.config.seed)
        return rng.sample(samples, self.subset_size)

    @staticmethod
    def download_agqa(target_dir: str, splits: Optional[List[str]] = None) -> Dict[str, Path]:
        """Download AGQA 2.0 balanced JSON files.

        Args:
            target_dir: Directory to save downloaded files.
            splits: Which splits to download. Defaults to all.

        Returns:
            Dict mapping split name to local file path.
        """
        import urllib.request

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        splits = splits or list(AGQA_URLS.keys())
        paths = {}
        for split in splits:
            url = AGQA_URLS[split]
            filename = url.split("/")[-1]
            local_path = target / filename
            if local_path.exists():
                print(f"  {split}: already exists at {local_path}")
            else:
                print(f"  {split}: downloading from {url} ...")
                urllib.request.urlretrieve(url, str(local_path))
                print(f"  {split}: saved to {local_path}")
            paths[split] = local_path
        return paths

    def load_from_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Load AGQA JSON and parse all entries.

        AGQA JSON format: {question_id: {question, answer, video_id, ...}, ...}

        Args:
            filepath: Path to AGQA JSON file.

        Returns:
            List of parsed sample dicts.
        """
        with open(filepath, "r") as f:
            raw_data = json.load(f)

        samples = []
        for qid, entry in raw_data.items():
            # Ensure video_id is present (AGQA stores it as a field)
            if "video_id" not in entry:
                continue
            parsed = self.parse_sample(entry)
            parsed["question_id"] = qid
            samples.append(parsed)
        return samples

    def split(
        self,
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Deterministic split by video_id to prevent data leakage.

        All QA pairs for a given video stay in the same split.

        Args:
            samples: List of parsed samples.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation. Test gets the remainder.

        Returns:
            (train_samples, val_samples, test_samples)
        """
        # Group by video_id
        video_groups: Dict[str, List[Dict[str, Any]]] = {}
        for s in samples:
            video_groups.setdefault(s["video_id"], []).append(s)

        # Sort video_ids for determinism, then shuffle with seed
        video_ids = sorted(video_groups.keys())
        rng = random.Random(self.config.seed)
        rng.shuffle(video_ids)

        n = len(video_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_vids = set(video_ids[:n_train])
        val_vids = set(video_ids[n_train : n_train + n_val])
        test_vids = set(video_ids[n_train + n_val :])

        train = [s for vid in video_ids if vid in train_vids for s in video_groups[vid]]
        val = [s for vid in video_ids if vid in val_vids for s in video_groups[vid]]
        test = [s for vid in video_ids if vid in test_vids for s in video_groups[vid]]

        return train, val, test
