"""AGQA dataset loader for video QA benchmarking."""

from typing import Any, Dict, List, Set
import random
from src.config import BenchmarkConfig


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
            "program": raw["program"],
        }

    def get_unique_video_ids(self, samples: List[Dict[str, Any]]) -> Set[str]:
        return {s["video_id"] for s in samples}

    def subsample(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(samples) <= self.subset_size:
            return samples
        rng = random.Random(self.config.seed)
        return rng.sample(samples, self.subset_size)
