#!/usr/bin/env python3
"""
Similarity metrics for comparing text instructions.

This module provides various text similarity metrics that can be used
for clustering and comparing task instructions.

Available metrics:
- Jaccard similarity: Word-based set similarity
- TF-IDF Cosine similarity: Weighted term similarity with IDF scoring
- Levenshtein similarity: Character-level edit distance similarity
- Oracle similarity: Ground truth similarity based on task family ID
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional


def extract_task_family_id(task_id: str) -> str:
    """Extract the family ID from a task ID.

    Args:
        task_id: Task ID in format like "76f2c72_2" or "76f2c72_1"

    Returns:
        Family ID (e.g., "76f2c72")
    """
    # Split by underscore and take the first part
    parts = task_id.split('_')
    return parts[0] if parts else task_id


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    # Convert to lowercase and split by non-alphanumeric characters
    return re.findall(r'\b\w+\b', text.lower())


def calculate_jaccard_similarity(s1: str, s2: str) -> float:
    """Calculate Jaccard similarity between two strings based on word tokens.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not s1 or not s2:
        return 0.0

    words1 = set(tokenize(s1))
    words2 = set(tokenize(s2))

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def calculate_cosine_similarity(s1: str, s2: str, idf_scores: Dict[str, float]) -> float:
    """Calculate TF-IDF cosine similarity between two strings.

    Args:
        s1: First string
        s2: Second string
        idf_scores: Dictionary mapping terms to their IDF scores

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not s1 or not s2:
        return 0.0

    tokens1 = tokenize(s1)
    tokens2 = tokenize(s2)

    # Calculate TF-IDF vectors
    def get_tfidf_vector(tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total_terms = len(tokens)
        tfidf = {}
        for term, count in tf.items():
            tf_score = count / total_terms if total_terms > 0 else 0
            idf_score = idf_scores.get(term, 0)
            tfidf[term] = tf_score * idf_score
        return tfidf

    vec1 = get_tfidf_vector(tokens1)
    vec2 = get_tfidf_vector(tokens2)

    # Calculate cosine similarity
    all_terms = set(vec1.keys()) | set(vec2.keys())

    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
    magnitude1 = math.sqrt(sum(v * v for v in vec1.values()))
    magnitude2 = math.sqrt(sum(v * v for v in vec2.values()))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def calculate_levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate normalized Levenshtein (edit distance) similarity.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not s1 or not s2:
        return 0.0

    # Levenshtein distance using dynamic programming
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return 0.0 if len2 > 0 else 1.0
    if len2 == 0:
        return 0.0

    # Create distance matrix
    distances = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        distances[i][0] = i
    for j in range(len2 + 1):
        distances[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,      # deletion
                distances[i][j - 1] + 1,      # insertion
                distances[i - 1][j - 1] + cost  # substitution
            )

    max_len = max(len1, len2)
    return 1.0 - (distances[len1][len2] / max_len)


def calculate_oracle_similarity(task_id1: str, task_id2: str) -> float:
    """Calculate oracle similarity based on task family ID.

    This is a "ground truth" similarity metric that returns 1.0 if two tasks
    belong to the same family (same prefix before underscore), 0.0 otherwise.

    Args:
        task_id1: First task ID (e.g., "76f2c72_2")
        task_id2: Second task ID (e.g., "76f2c72_3")

    Returns:
        1.0 if same family, 0.0 otherwise
    """
    if not task_id1 or not task_id2:
        return 0.0

    family1 = extract_task_family_id(task_id1)
    family2 = extract_task_family_id(task_id2)

    return 1.0 if family1 == family2 else 0.0


def compute_idf_scores(texts: List[str]) -> Dict[str, float]:
    """Compute IDF (Inverse Document Frequency) scores for all terms in a text corpus.

    Args:
        texts: List of text documents

    Returns:
        Dictionary mapping terms to their IDF scores
    """
    num_docs = len(texts)
    if num_docs == 0:
        return {}

    # Count document frequency for each term
    doc_freq = defaultdict(int)
    for text in texts:
        unique_terms = set(tokenize(text))
        for term in unique_terms:
            doc_freq[term] += 1

    # Calculate IDF scores
    idf_scores = {}
    for term, freq in doc_freq.items():
        idf_scores[term] = math.log(num_docs / freq) if freq > 0 else 0

    return idf_scores


def calculate_similarity(
    s1: str,
    s2: str,
    metric: str = "jaccard",
    idf_scores: Dict[str, float] = None,
    task_id1: Optional[str] = None,
    task_id2: Optional[str] = None
) -> float:
    """Calculate similarity between two strings using the specified metric.

    Args:
        s1: First string (instruction text or task ID for oracle)
        s2: Second string (instruction text or task ID for oracle)
        metric: Similarity metric to use ('jaccard', 'cosine', 'levenshtein', 'oracle')
        idf_scores: IDF scores (required for cosine similarity)
        task_id1: First task ID (required for oracle similarity)
        task_id2: Second task ID (required for oracle similarity)

    Returns:
        Similarity score between 0.0 and 1.0

    Raises:
        ValueError: If an unknown metric is specified or if required parameters are missing
    """
    if metric == "jaccard":
        return calculate_jaccard_similarity(s1, s2)
    elif metric == "cosine":
        if idf_scores is None:
            raise ValueError("IDF scores required for cosine similarity")
        return calculate_cosine_similarity(s1, s2, idf_scores)
    elif metric == "levenshtein":
        return calculate_levenshtein_similarity(s1, s2)
    elif metric == "oracle":
        if task_id1 is None or task_id2 is None:
            raise ValueError("Task IDs required for oracle similarity")
        return calculate_oracle_similarity(task_id1, task_id2)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
