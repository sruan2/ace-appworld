#!/usr/bin/env python3
"""
Select and reorder tasks from a dataset file based on difficulty and size.

Usage (run from repository root):
    python3 experiments/curriculum/data_selector.py --dataset train.txt --output train_subset.txt --size 50 --difficulty balanced --order easy-to-hard
"""

import argparse
import json
import random
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


def get_task_difficulty(task_id: str) -> int:
    """Get difficulty level for a task from its metadata.json file."""
    metadata_path = Path("data/tasks") / task_id / "ground_truth" / "metadata.json"

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return metadata.get('difficulty', -1)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read metadata for task {task_id}: {e}")
        return -1


def get_tasks_with_difficulty(task_ids: List[str]) -> List[Tuple[str, int]]:
    """Get list of (task_id, difficulty) tuples."""
    task_difficulty_pairs = []
    for task_id in task_ids:
        difficulty = get_task_difficulty(task_id)
        task_difficulty_pairs.append((task_id, difficulty))
    return task_difficulty_pairs


def filter_by_difficulty(
    task_pairs: List[Tuple[str, int]],
    difficulty_mode: str
) -> List[Tuple[str, int]]:
    """Filter tasks based on difficulty mode."""
    if difficulty_mode == "all":
        return [p for p in task_pairs if p[1] >= 0]
    elif difficulty_mode == "easy":
        return [p for p in task_pairs if p[1] == 1]
    elif difficulty_mode == "medium":
        return [p for p in task_pairs if p[1] == 2]
    elif difficulty_mode == "hard":
        return [p for p in task_pairs if p[1] == 3]
    elif difficulty_mode in ["balanced", "custom"]:
        # Keep all difficulties for balanced/custom selection
        return [p for p in task_pairs if p[1] >= 0]
    else:
        raise ValueError(f"Unknown difficulty mode: {difficulty_mode}")


def select_tasks(
    task_pairs: List[Tuple[str, int]],
    size: int,
    difficulty_mode: str,
    ratio: str = None
) -> List[Tuple[str, int]]:
    """Select tasks based on size and difficulty distribution."""
    if difficulty_mode in ["balanced", "custom"]:
        # Group by difficulty
        by_difficulty = {1: [], 2: [], 3: []}
        for task_id, diff in task_pairs:
            if diff in by_difficulty:
                by_difficulty[diff].append((task_id, diff))

        # Determine ratio based on mode
        if difficulty_mode == "balanced":
            # Strictly equal distribution: 1:1:1
            ratio_easy = ratio_medium = ratio_hard = 1
        else:  # difficulty_mode == "custom"
            # Custom ratio from --ratio argument
            if ratio:
                try:
                    parts = [int(x) for x in ratio.split(':')]
                    if len(parts) != 3:
                        raise ValueError("Ratio must have 3 parts (easy:medium:hard)")
                    ratio_easy, ratio_medium, ratio_hard = parts
                except ValueError as e:
                    print(f"Warning: Invalid ratio '{ratio}', using equal distribution. Error: {e}")
                    ratio_easy = ratio_medium = ratio_hard = 1
            else:
                # Default for custom mode if no ratio specified
                ratio_easy = ratio_medium = ratio_hard = 1

        # Calculate how many tasks per difficulty level based on ratio
        total_ratio = ratio_easy + ratio_medium + ratio_hard
        count_easy = int(size * ratio_easy / total_ratio)
        count_medium = int(size * ratio_medium / total_ratio)
        count_hard = size - count_easy - count_medium  # Ensure we hit exact size

        selected = []

        # Select tasks according to ratio
        selected.extend(by_difficulty[1][:min(count_easy, len(by_difficulty[1]))])
        selected.extend(by_difficulty[2][:min(count_medium, len(by_difficulty[2]))])
        selected.extend(by_difficulty[3][:min(count_hard, len(by_difficulty[3]))])

        return selected[:size]  # Ensure we don't exceed size
    else:
        # For non-balanced modes, just take first N tasks
        return task_pairs[:size]


def order_tasks(
    task_pairs: List[Tuple[str, int]],
    order_mode: str,
    random_seed: int = None
) -> List[Tuple[str, int]]:
    """Order tasks based on the specified mode."""
    if order_mode == "original":
        return task_pairs
    elif order_mode == "easy-to-hard":
        return sorted(task_pairs, key=lambda x: (x[1], x[0]))
    elif order_mode == "hard-to-easy":
        return sorted(task_pairs, key=lambda x: (-x[1], x[0]))
    elif order_mode == "random":
        if random_seed is not None:
            random.seed(random_seed)
        shuffled = task_pairs.copy()
        random.shuffle(shuffled)
        return shuffled
    else:
        raise ValueError(f"Unknown order mode: {order_mode}")


def save_dataset(task_ids: List[str], output_name: str):
    """Save selected task IDs to output file."""
    output_file = Path("data/datasets") / output_name

    with open(output_file, 'w') as f:
        for task_id in task_ids:
            f.write(f"{task_id}\n")

    print(f"Saved {len(task_ids)} tasks to: {output_file}")


def print_summary(task_pairs: List[Tuple[str, int]]):
    """Print summary of selected tasks."""
    total = len(task_pairs)
    if total == 0:
        print("No tasks selected!")
        return

    # Count by difficulty
    counts = {1: 0, 2: 0, 3: 0}
    for _, diff in task_pairs:
        if diff in counts:
            counts[diff] += 1

    print("\nSelection Summary:")
    print(f"Total tasks: {total}")
    print(f"  Easy (Level 1):   {counts[1]:3d} ({counts[1]/total*100:5.1f}%)")
    print(f"  Medium (Level 2): {counts[2]:3d} ({counts[2]/total*100:5.1f}%)")
    print(f"  Hard (Level 3):   {counts[3]:3d} ({counts[3]/total*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Select and reorder tasks from a dataset file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Input dataset filename (e.g., train.txt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output dataset filename (e.g., train_subset.txt)"
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Number of tasks to select"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard", "balanced", "custom", "all"],
        default="all",
        help="Difficulty filter: easy/medium/hard (single level), balanced (equal 1:1:1), custom (custom ratio), all (default: all)"
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["original", "easy-to-hard", "hard-to-easy", "random"],
        default="original",
        help="Task ordering (default: original)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (only used with --order random)"
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default=None,
        help="Difficulty ratio for custom mode (e.g., '1:2:1' for easy:medium:hard)"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    task_ids = load_dataset(args.dataset)
    print(f"Found {len(task_ids)} tasks")

    # Get difficulty information
    print("Reading task metadata...")
    task_pairs = get_tasks_with_difficulty(task_ids)

    # Filter by difficulty
    print(f"Filtering by difficulty: {args.difficulty}")
    filtered_pairs = filter_by_difficulty(task_pairs, args.difficulty)
    print(f"After filtering: {len(filtered_pairs)} tasks")

    # Select tasks
    print(f"Selecting {args.size} tasks...")
    if args.difficulty == "custom":
        if args.ratio:
            print(f"Using custom ratio: {args.ratio}")
        else:
            print("Warning: --ratio not specified for custom mode")
    elif args.difficulty == "balanced":
        print("Using balanced distribution (1:1:1)")
    selected_pairs = select_tasks(filtered_pairs, args.size, args.difficulty, args.ratio)

    if len(selected_pairs) < args.size:
        print(f"Warning: Only {len(selected_pairs)} tasks available, requested {args.size}")

    # Order tasks
    print(f"Ordering tasks: {args.order}")
    ordered_pairs = order_tasks(selected_pairs, args.order, args.seed)

    # Extract task IDs
    selected_task_ids = [task_id for task_id, _ in ordered_pairs]

    # Print summary
    print_summary(ordered_pairs)

    # Save output
    save_dataset(selected_task_ids, args.output)


if __name__ == "__main__":
    main()
