"""
mock_python.py

This is a mock Python script used for demonstrating
File-Based Node Parsers with CodeSplitter in LlamaIndex.
"""

import math
from typing import List, Dict


# -----------------------------
# Utility Functions
# -----------------------------

def calculate_average(numbers: List[float]) -> float:
    """
    Calculate the average of a list of numbers.
    """
    if not numbers:
        return 0.0

    total = sum(numbers)
    return total / len(numbers)


def normalize_numbers(numbers: List[float]) -> List[float]:
    """
    Normalize a list of numbers using min-max normalization.
    """
    if not numbers:
        return []

    min_val = min(numbers)
    max_val = max(numbers)

    if min_val == max_val:
        return [0.0 for _ in numbers]

    return [(n - min_val) / (max_val - min_val) for n in numbers]


# -----------------------------
# Data Processing Class
# -----------------------------

class DataProcessor:
    """
    A simple data processing class for demonstration purposes.
    """

    def __init__(self, name: str):
        self.name = name
        self.records: List[Dict[str, float]] = []

    def add_record(self, record: Dict[str, float]) -> None:
        """
        Add a single record to the processor.
        """
        self.records.append(record)

    def compute_statistics(self) -> Dict[str, float]:
        """
        Compute basic statistics from stored records.
        """
        values = [r["value"] for r in self.records if "value" in r]

        return {
            "count": len(values),
            "average": calculate_average(values),
            "max": max(values) if values else 0.0,
            "min": min(values) if values else 0.0,
        }


# -----------------------------
# Example Workflow
# -----------------------------

def run_demo():
    """
    Run a simple demo workflow.
    """
    processor = DataProcessor("demo_processor")

    sample_values = [10, 20, 30, 40, 50]

    for v in sample_values:
        processor.add_record({"value": v})

    stats = processor.compute_statistics()
    normalized = normalize_numbers(sample_values)

    print("Statistics:", stats)
    print("Normalized values:", normalized)


if __name__ == "__main__":
    run_demo()
