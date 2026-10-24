"""NaviTrace scoring and trace parsing, copied unchanged from the official evaluation notebook.

Source: https://github.com/leggedrobotics/navitrace_evaluation (src/run_evaluation.ipynb),
MIT License, see LICENSE in this folder. Only the data file paths were adjusted so they
resolve next to this module.
"""

import re
from typing import List


def parse_trace(text: str) -> List[List[float]]:
    """
    Parse point trace from model response.
    Expected format: [[x1, y1], [x2, y2], ...] or similar variations.
    Returns a list of [x, y] coordinate pairs, or an empty list if parsing fails.
    """

    try:
        # Try to find JSON-like array in response
        patterns = [
            r"\[\s*\[[\d\s,.-]+\]\s*(?:,\s*\[[\d\s,.-]+\]\s*)*\]",  # [[x,y], [x,y], ...]
            r"\(\s*\([\d\s,.-]+\)\s*(?:,\s*\([\d\s,.-]+\)\s*)*\)",  # ((x,y), (x,y), ...)
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Parse the first match
                match = matches[0]
                # Convert to proper JSON format
                match = match.replace("(", "[").replace(")", "]")
                points = json.loads(match)
                # Validate format
                if all(isinstance(p, list) and len(p) == 2 for p in points):
                    return [[float(p[0]), float(p[1])] for p in points]

        # If no pattern matches, try to extract numbers and pair them
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if len(numbers) >= 2 and len(numbers) % 2 == 0:
            points = []
            for i in range(0, len(numbers), 2):
                points.append([float(numbers[i]), float(numbers[i + 1])])
            return points

    except (json.JSONDecodeError, ValueError, IndexError) as e:
        print(f"Failed to parse trace: {e}")
        return []




import ast
import functools
import json
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from skimage.draw import line_aa
from skimage.draw import line as sk_line
from tqdm import tqdm

PENALTY_SCORES_PATH = str(Path(__file__).with_name("category_penalty.tsv"))
M2F_CONFIG_PATH = str(Path(__file__).with_name("mask2former_config.json"))

@functools.lru_cache(maxsize=4)
def create_penalty_lookup(embodiment: str) -> Dict[int, float]:
    """Creates a direct mapping from a category ID (`label_id`) to its penalty factor."""

    # Load fixed penalty values
    penalty_values_df = pd.read_csv(PENALTY_SCORES_PATH, sep="\t")

    # Load Mask2Former mapping from IDs to labels
    with open(M2F_CONFIG_PATH, "r") as f:
        config = json.load(f)
    id2label = {int(k): v for k, v in config["id2label"].items()}

    label_id_to_penalty = {}
    for label_id, category_name in id2label.items():
        
        # Look up the penalty value
        row = penalty_values_df[
            penalty_values_df["category"] == category_name
        ]
        penalty_value = float(row.iloc[0][embodiment]) * 0.8  # Adjust scale
        label_id_to_penalty[label_id] = penalty_value

    return label_id_to_penalty

def rasterize_gt_trace(
    gt_trace: List[List[float]], height: int, width: int
) -> np.ndarray:
    """Converts a line trace into a dense array of pixel coordinates."""

    gt_trace_np = np.array(gt_trace)
    gt_line_pixels = []
    if len(gt_trace_np) > 1:
        for i in range(len(gt_trace_np) - 1):
            p1, p2 = gt_trace_np[i], gt_trace_np[i + 1]
            r0, c0, r1, c1 = (
                int(round(p1[1])),
                int(round(p1[0])),
                int(round(p2[1])),
                int(round(p2[0])),
            )
            rr, cc, _ = line_aa(r0, c0, r1, c1)
            valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            gt_line_pixels.extend(zip(rr[valid], cc[valid]))
    elif len(gt_trace_np) == 1:
        r, c = int(round(gt_trace_np[0][1])), int(round(gt_trace_np[0][0]))
        if 0 <= r < height and 0 <= c < width:
            gt_line_pixels.append((r, c))

    return np.array(gt_line_pixels)

def create_penalty_mask(
    segmentation_mask: np.ndarray,
    gt_trace: List[List[float]],
    embodiment: str,
    distance_threshold: float = 35,
) -> np.ndarray:

    # Initialize mask with default no penalty
    height, width = segmentation_mask.shape
    penalty_mask = np.full((height, width), 0, dtype=float)

    # Create a KDTree from ground truth pixels for efficient distance queries
    gt_line_pixels = rasterize_gt_trace(gt_trace, height, width)
    gt_tree = KDTree(gt_line_pixels)

    # Create a more efficient lookup for segment info and penalty values
    label_id_to_penalty = create_penalty_lookup(embodiment)

    # Get label IDs for all pixels
    all_label_ids = segmentation_mask.ravel()

    # Identify pixels that belong to undesired segments
    undesired_mask = np.isin(all_label_ids, list(label_id_to_penalty.keys()))
    undesired_indices = np.where(undesired_mask)[0]
    if undesired_indices.size == 0:
        return penalty_mask

    # Map indices to coordinates
    rows, cols = np.unravel_index(undesired_indices, (height, width))
    undesired_coords = np.vstack((rows, cols)).T

    # Perform a single batch query for distances for all undesired pixels
    distances, _ = gt_tree.query(undesired_coords)

    # Filter for pixels that are beyond the distance threshold
    coords_to_penalize = undesired_coords[distances > distance_threshold]

    if coords_to_penalize.size > 0:
        # Apply penalties
        rows_pen, cols_pen = coords_to_penalize[:, 0], coords_to_penalize[:, 1]
        label_ids_to_penalize = segmentation_mask[rows_pen, cols_pen]
        penalties = np.vectorize(label_id_to_penalty.get)(label_ids_to_penalize)
        penalty_mask[rows_pen, cols_pen] = penalties

    return penalty_mask

BAD_SCORE_THRESHOLD = 3234.75

def resample_to_match_length(
    trace_1: np.ndarray, trace_2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    if len(trace_1) == 0 or len(trace_2) == 0:
        raise ValueError("One of the traces is empty")
    if len(trace_1) == len(trace_2):
        return trace_1, trace_2
    elif len(trace_1) > len(trace_2):
        longer, shorter = (trace_1, trace_2)
    else:
        shorter, longer = (trace_1, trace_2)
    if len(shorter) == 1:
        return shorter * len(longer), longer

    # Parameterize shorter trajectory by cumulative distance
    dists = np.cumsum(
        [0]
        + [np.linalg.norm(shorter[i] - shorter[i - 1]) for i in range(1, len(shorter))]
    )
    dists = dists / dists[-1]  # Normalize to [0,1]

    # Create new parameter values matching longer trajectory length
    new_params = np.linspace(0, 1, len(longer))

    # Interpolate x and y coordinates separately
    new_x = np.interp(new_params, dists, shorter[:, 0])
    new_y = np.interp(new_params, dists, shorter[:, 1])
    shorter = np.column_stack([new_x, new_y])

    if len(trace_1) > len(trace_2):
        return longer, shorter
    else:
        return shorter, longer

def calculate_semantic_penalty(
    prediction: np.ndarray, penalty_mask: np.ndarray
) -> List[float]:

    penalties = []
    for i in range(len(prediction) - 1):
        x1, y1 = int(round(prediction[i][0])), int(round(prediction[i][1]))
        x2, y2 = int(round(prediction[i + 1][0])), int(round(prediction[i + 1][1]))

        # Use scikit-image's optimized line drawing algorithm
        rr, cc = sk_line(y1, x1, y2, x2)

        # Access mask using (y, x) coordinates
        height, width = penalty_mask.shape
        valid_indices = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        penalties.extend(penalty_mask[rr[valid_indices], cc[valid_indices]].tolist())

    return np.mean(penalties)

def calculate_fde(prediction: np.ndarray, ground_truth: np.ndarray):

    return np.linalg.norm(prediction[-1] - ground_truth[-1])

def calculate_dtw(prediction: np.ndarray, ground_truth: np.ndarray):

    # Create cost matrix
    n, m = len(prediction), len(ground_truth)
    cost_matrix = np.full((n + 1, m + 1), np.inf)
    cost_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            euclidean_distance = np.linalg.norm(prediction[i - 1] - ground_truth[j - 1])

            # Find the minimum from the three possible previous cells
            min_prev_cost = min(
                cost_matrix[i - 1, j],  # Insertion
                cost_matrix[i, j - 1],  # Deletion
                cost_matrix[i - 1, j - 1],  # Match
            )

            cost_matrix[i, j] = euclidean_distance + min_prev_cost

    return cost_matrix[n, m]

def normalize_score(score: float) -> float:

    # Normalize score so that a perferct score is at 100 and a score worse than the avg. performance of predicting a vertical line through the center is < 0
    return (BAD_SCORE_THRESHOLD - score) / BAD_SCORE_THRESHOLD * 100

def score(
    prediction: List[List[float]],
    ground_truths: List[List[List[float]]],
    segmentation_mask: np.ndarray,
    embodiment: str,
):
    
    # Iterate over all ground-truths
    scores = []
    for ground_truth in ground_truths:
        
        # Create penalty mask
        penalty_mask = create_penalty_mask(segmentation_mask, ground_truth, embodiment)

        # Convert to NumPy
        prediction, ground_truth = np.array(prediction), np.array(ground_truth)

        # Resample if necessary
        if len(prediction) != len(ground_truth):
            prediction, ground_truth = resample_to_match_length(prediction, ground_truth)

        # Calculate score function
        sem_penalty = calculate_semantic_penalty(prediction, penalty_mask)
        fde = calculate_fde(prediction, ground_truth)
        dtw = calculate_dtw(prediction, ground_truth)
        scores.append(dtw + fde + sem_penalty)
        
    # Select the best score
    score = min(scores)

    # Normalize
    return normalize_score(score)
