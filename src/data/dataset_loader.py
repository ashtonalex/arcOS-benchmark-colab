"""
HuggingFace dataset interface for RoG-WebQSP.

Handles dataset loading with Google Drive caching and schema validation.
"""

import os
from pathlib import Path
from typing import Optional, Union
from datasets import load_dataset, DatasetDict, Dataset


class RoGWebQSPLoader:
    """
    Loader for the RoG-WebQSP dataset from HuggingFace.

    This dataset contains question-answer pairs with Freebase subgraphs.
    Each example includes:
    - id: Unique identifier
    - question: Natural language question
    - answer: List of answer entities
    - q_entity: Question topic entity
    - a_entity: Answer entity
    - graph: List of [subject, relation, object] triples
    """

    EXPECTED_FIELDS = ["id", "question", "answer", "q_entity", "a_entity", "graph"]

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the dataset loader.

        Args:
            cache_dir: Directory for caching downloaded datasets (uses Google Drive in Colab)
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache_dir)
            print(f"✓ HuggingFace cache directory: {cache_dir}")

    def load(
        self,
        dataset_name: str = "rmanluo/RoG-webqsp",
        split: Optional[str] = None
    ) -> Union[DatasetDict, Dataset]:
        """
        Load the RoG-WebQSP dataset from HuggingFace.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Specific split to load (train/validation/test), or None for all splits

        Returns:
            DatasetDict if split is None, else single Dataset
        """
        print(f"Loading dataset: {dataset_name}")
        if split:
            print(f"Split: {split}")

        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

        if split:
            print(f"✓ Loaded {len(dataset)} examples from {split} split")
        else:
            print(f"✓ Loaded dataset with splits: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                print(f"  - {split_name}: {len(split_data)} examples")

        return dataset

    def inspect_schema(self, dataset: Union[DatasetDict, Dataset], num_examples: int = 1):
        """
        Print dataset schema and sample examples.

        Args:
            dataset: Dataset or DatasetDict to inspect
            num_examples: Number of sample examples to display
        """
        print("\n" + "=" * 60)
        print("Dataset Schema Inspection")
        print("=" * 60)

        # Get a sample dataset
        if isinstance(dataset, DatasetDict):
            sample_dataset = dataset[list(dataset.keys())[0]]
            print(f"Inspecting first split: {list(dataset.keys())[0]}")
        else:
            sample_dataset = dataset

        # Print fields
        print("\nFields:")
        for field_name, field_type in sample_dataset.features.items():
            print(f"  - {field_name}: {field_type}")

        # Validate expected fields
        actual_fields = set(sample_dataset.features.keys())
        expected_fields = set(self.EXPECTED_FIELDS)
        missing_fields = expected_fields - actual_fields
        extra_fields = actual_fields - expected_fields

        if missing_fields:
            print(f"\n⚠ Warning: Missing expected fields: {missing_fields}")
        if extra_fields:
            print(f"\nℹ Additional fields found: {extra_fields}")
        if not missing_fields:
            print("\n✓ All expected fields present")

        # Print sample examples
        print(f"\nSample Examples (first {num_examples}):")
        for i in range(min(num_examples, len(sample_dataset))):
            example = sample_dataset[i]
            print(f"\n--- Example {i} ---")
            print(f"ID: {example['id']}")
            print(f"Question: {example['question']}")
            print(f"Answer: {example['answer']}")
            print(f"Question Entity: {example['q_entity']}")
            print(f"Answer Entity: {example['a_entity']}")
            print(f"Graph: {len(example['graph'])} triples")
            if example['graph']:
                print(f"  Sample triple: {example['graph'][0]}")

        print("=" * 60)

    def compute_statistics(self, dataset: Union[DatasetDict, Dataset]):
        """
        Compute and print dataset statistics.

        Args:
            dataset: Dataset or DatasetDict to analyze
        """
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)

        datasets_to_analyze = {}
        if isinstance(dataset, DatasetDict):
            datasets_to_analyze = dataset
        else:
            datasets_to_analyze = {"dataset": dataset}

        for split_name, split_data in datasets_to_analyze.items():
            print(f"\n--- {split_name} ---")
            print(f"Total examples: {len(split_data)}")

            # Compute graph size statistics
            graph_sizes = [len(example['graph']) for example in split_data]
            avg_graph_size = sum(graph_sizes) / len(graph_sizes) if graph_sizes else 0
            min_graph_size = min(graph_sizes) if graph_sizes else 0
            max_graph_size = max(graph_sizes) if graph_sizes else 0

            print(f"Graph size (triples):")
            print(f"  - Average: {avg_graph_size:.1f}")
            print(f"  - Min: {min_graph_size}")
            print(f"  - Max: {max_graph_size}")

        print("=" * 60)

    def validate_split_counts(
        self,
        dataset: DatasetDict,
        expected_train: int = 2830,
        expected_val: int = 246,
        expected_test: int = 1630
    ) -> bool:
        """
        Validate that dataset splits have expected sizes.

        Args:
            dataset: DatasetDict to validate
            expected_train: Expected training set size
            expected_val: Expected validation set size
            expected_test: Expected test set size

        Returns:
            True if all splits match expected sizes, False otherwise
        """
        print("\n" + "=" * 60)
        print("Dataset Split Validation")
        print("=" * 60)

        actual_train = len(dataset["train"])
        actual_val = len(dataset["validation"])
        actual_test = len(dataset["test"])

        train_match = actual_train == expected_train
        val_match = actual_val == expected_val
        test_match = actual_test == expected_test

        print(f"Train: {actual_train} {'✓' if train_match else f'✗ (expected {expected_train})'}")
        print(f"Validation: {actual_val} {'✓' if val_match else f'✗ (expected {expected_val})'}")
        print(f"Test: {actual_test} {'✓' if test_match else f'✗ (expected {expected_test})'}")

        all_match = train_match and val_match and test_match
        if all_match:
            print("\n✓ All splits have expected sizes")
        else:
            print("\n✗ Split size mismatch detected")

        print("=" * 60)
        return all_match
