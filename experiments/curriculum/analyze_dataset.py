#!/usr/bin/env python3
"""
Analyze task difficulty distribution from a dataset file.

Usage (run from repository root):
    python3 experiments/curriculum/analyze_dataset.py --dataset train.txt
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def load_dataset(dataset_name: str) -> List[str]:
    """Load task IDs from dataset file."""
    dataset_file = Path("data/datasets") / dataset_name

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    with open(dataset_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    return task_ids


def get_task_difficulty(task_id: str) -> Tuple[str, int]:
    """Get difficulty level for a task from its metadata.json file."""
    metadata_path = Path("data/tasks") / task_id / "ground_truth" / "metadata.json"

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return (task_id, metadata.get('difficulty', -1))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read metadata for task {task_id}: {e}")
        return (task_id, -1)


def analyze_difficulty_distribution(task_ids: List[str]) -> List[Tuple[str, int]]:
    """Analyze difficulty distribution and return task-difficulty pairs."""
    task_difficulty_pairs = []

    for task_id in task_ids:
        task_id, difficulty = get_task_difficulty(task_id)
        task_difficulty_pairs.append((task_id, difficulty))

    return task_difficulty_pairs


def print_distribution(task_difficulty_pairs: List[Tuple[str, int]]):
    """Print difficulty distribution statistics."""
    # Count by difficulty
    difficulty_counts = {}
    total_tasks = len(task_difficulty_pairs)

    for _, difficulty in task_difficulty_pairs:
        if difficulty >= 0:
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    print("\n" + "=" * 60)
    print("DIFFICULTY DISTRIBUTION")
    print("=" * 60)

    print(f"\n{'Difficulty':<12} {'Count':<10} {'Percentage':<12} {'Bar'}")
    print("-" * 60)

    for difficulty in sorted(difficulty_counts.keys()):
        count = difficulty_counts[difficulty]
        percentage = (count / total_tasks) * 100
        bar = "█" * int(percentage / 2)
        print(f"Level {difficulty:<6} {count:<10} {percentage:>6.2f}%     {bar}")

    print("-" * 60)
    print(f"{'Total':<12} {total_tasks:<10} {'100.00%':<12}")
    print()


def print_task_list(task_difficulty_pairs: List[Tuple[str, int]]):
    """Print tasks in original order with difficulty level appended."""
    print("\n" + "=" * 60)
    print("TASKS IN ORIGINAL ORDER (with difficulty)")
    print("=" * 60)
    print()

    for task_id, difficulty in task_difficulty_pairs:
        diff_str = f"Level {difficulty}" if difficulty >= 0 else "Unknown"
        print(f"{task_id:<20} -> {diff_str}")

    print()


def save_analysis(task_difficulty_pairs: List[Tuple[str, int]], dataset_name: str):
    """Save formatted analysis to JSON file in datasets folder."""
    output_file = Path("data/datasets") / f"{dataset_name}_analysis.json"

    # Count by difficulty
    difficulty_counts = {}
    total_tasks = len(task_difficulty_pairs)

    for _, difficulty in task_difficulty_pairs:
        if difficulty >= 0:
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    # Build distribution
    distribution = {}
    for difficulty in sorted(difficulty_counts.keys()):
        count = difficulty_counts[difficulty]
        percentage = (count / total_tasks) * 100
        distribution[f"level_{difficulty}"] = {
            "count": count,
            "percentage": round(percentage, 2)
        }

    # Filter out unknown difficulty tasks
    tasks_with_difficulty = [
        {"task_id": task_id, "difficulty": difficulty}
        for task_id, difficulty in task_difficulty_pairs
        if difficulty >= 0
    ]

    # Create output data
    output_data = {
        "dataset": dataset_name,
        "total_tasks": total_tasks,
        "distribution": distribution,
        "tasks": tasks_with_difficulty
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nAnalysis saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze task difficulty distribution from a dataset file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset filename (e.g., train.txt, test.txt)"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    task_ids = load_dataset(args.dataset)
    print(f"Found {len(task_ids)} tasks")

    # Analyze difficulty distribution
    print("Analyzing difficulty levels...")
    task_difficulty_pairs = analyze_difficulty_distribution(task_ids)

    # Print results
    print_distribution(task_difficulty_pairs)
    print_task_list(task_difficulty_pairs)

    # Save analysis
    dataset_name = Path(args.dataset).stem
    save_analysis(task_difficulty_pairs, dataset_name)


if __name__ == "__main__":
    main()
