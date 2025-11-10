"""
Text Block Detection Module
===========================
Identifies blocks of text using horizontal alignment and vertical proximity.

This module is part of the document forgery detection system.
See ALIGNMENT_AND_COLON_LOGIC_SUMMARY.md for detailed documentation.

Two approaches available:
1. Hierarchical clustering (spatial grid) - Original approach
2. DBSCAN - Density-based clustering considering x,y coordinates together
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
from statistics import median
import numpy as np

from spatial_grid import SpatialGrid

# Try to import sklearn for clustering algorithms
try:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
    DBSCAN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    DBSCAN_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

# Text block detection thresholds
LINE_CLUSTER_TOLERANCE = 1.5           # pt - tolerance for grouping cells into the same line
MIN_BLOCK_LINE_COUNT = 2               # require at least 2 lines to form a block
BASELINE_SNAP_TOLERANCE = 5.0          # pt - max distance to snap block edge to baseline

# DBSCAN parameters (density-based clustering - single stage)
# Based on smoke test: eps=12.0, min_samples=3, h_scale=0.2 produces ~7 blocks (optimal)
# h_scale=0.2 allows wide horizontal grouping while maintaining tight vertical clustering
DBSCAN_EPS = 12.0                      # pt - maximum distance between two samples to be neighbors
DBSCAN_MIN_SAMPLES = 3                 # minimum samples in neighborhood for core point
DBSCAN_HORIZONTAL_SCALE = 0.2          # scale factor for x-coordinates (< 1.0 to allow wider horizontal clustering)

# Two-stage clustering parameters (vertical then horizontal)
# Testing h_eps reduction: dbscan+dbscan with v_eps=12.0, h_eps=40.0 for better column separation
TWO_STAGE_VERTICAL_METHOD = "dbscan"   # "dbscan" or "agglomerative"
TWO_STAGE_HORIZONTAL_METHOD = "dbscan"  # "dbscan" or "agglomerative" (changed to dbscan)
TWO_STAGE_VERTICAL_EPS = 12.0          # pt - vertical clustering threshold
TWO_STAGE_HORIZONTAL_EPS = 40.0        # pt - horizontal clustering threshold (REDUCED from 70.0 to 40.0)
TWO_STAGE_MIN_SAMPLES = 2              # minimum samples for DBSCAN core point

# Iterative clustering parameters (alternating vertical/horizontal until convergence)
ITERATIVE_VERTICAL_EPS = 12.0          # pt - vertical clustering threshold
ITERATIVE_HORIZONTAL_EPS = 40.0        # pt - horizontal clustering threshold
ITERATIVE_MIN_SAMPLES = 2              # minimum samples for DBSCAN core point
ITERATIVE_MAX_ITERATIONS = 3           # maximum iterations (typical convergence: 3-5)
ITERATIVE_MIN_BLOCK_CELLS = 2          # minimum cells per block to prevent over-fragmentation

# Clustering mode selection
USE_ITERATIVE = True                   # Use iterative multi-pass clustering
USE_TWO_STAGE = False                  # Use two-stage clustering (vertical then horizontal)
USE_DBSCAN = False                     # Use single-stage DBSCAN

# Spatial grid tuning
GRID_ALIGN_FRACTION = 0.15             # fraction of overlap required along orthogonal axis (REDUCED from 0.35 to allow looser grouping)
GRID_LOOK_RADIUS = 1                   # how many neighboring cells to inspect within the grid
GRID_SPACING_MULTIPLIER = 1.35         # typical spacing multiplier for cluster cutoffs
GRID_FALLBACK_GAP = 36.0               # pt - fallback gap if no spacing statistics are available
MIN_VERTICAL_CUTOFF = 8.0              # pt - minimum cutoff for vertical clustering (prevents over-fragmentation)
CLUSTER_COLUMN_SPACING_MULTIPLIER = 1.15  # controls secondary horizontal clustering tightness
MAX_CLUSTER_WIDTH_STD = 18.0           # pt - reject clusters with width stddev above this
MIN_DOMINANT_ORIENTATION_SHARE = 0.55  # min share for a dominant orientation per cluster


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _classify_cell_alignment_for_blocks(
    cell: Dict[str, Any],
    left_baselines: List[float],
    right_baselines: List[float],
    tolerance: float = 3.0
) -> Dict[str, Any]:
    """
    Classify a cell's horizontal alignment relative to detected baselines.
    """
    bbox = cell["bbox"]
    left_edge = bbox["x0"]
    right_edge = bbox["x1"]
    center_x = (left_edge + right_edge) / 2

    best_left: Optional[Tuple[float, float]] = None
    if left_baselines:
        best_left = min(((abs(left_edge - value), value) for value in left_baselines), default=None)

    best_right: Optional[Tuple[float, float]] = None
    if right_baselines:
        best_right = min(((abs(right_edge - value), value) for value in right_baselines), default=None)

    orientation = "center"
    baseline_value: Optional[float] = None
    baseline_diff: Optional[float] = None

    if best_left and best_left[0] <= tolerance:
        orientation = "left"
        baseline_diff, baseline_value = best_left
    elif best_right and best_right[0] <= tolerance:
        orientation = "right"
        baseline_diff, baseline_value = best_right
    else:
        orientation = "center"
        baseline_value = center_x

    return {
        "orientation": orientation,
        "baseline_value": baseline_value,
        "baseline_diff": baseline_diff,
        "anchor_x": left_edge if orientation == "left" else (right_edge if orientation == "right" else center_x)
    }


def _std_dev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((val - mean) ** 2 for val in values) / len(values)
    return variance ** 0.5


def _cluster_lines_for_blocks(cell_infos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group cells into lines based on vertical proximity.
    """
    if not cell_infos:
        return []

    lines: List[Dict[str, Any]] = []

    for cell in sorted(cell_infos, key=lambda c: c["center_y"]):
        assigned_line: Optional[Dict[str, Any]] = None
        for line in lines:
            if abs(cell["center_y"] - line["center_y"]) <= LINE_CLUSTER_TOLERANCE:
                assigned_line = line
                break

        if assigned_line is None:
            assigned_line = {
                "cells": [],
                "centers": [],
                "top": cell["bbox"]["y0"],
                "bottom": cell["bbox"]["y1"],
                "left": cell["bbox"]["x0"],
                "right": cell["bbox"]["x1"],
                "orientation_counts": Counter(),
                "alignment_values": []
            }
            lines.append(assigned_line)

        assigned_line["cells"].append(cell)
        assigned_line["centers"].append(cell["center_y"])
        assigned_line["top"] = min(assigned_line["top"], cell["bbox"]["y0"])
        assigned_line["bottom"] = max(assigned_line["bottom"], cell["bbox"]["y1"])
        assigned_line["left"] = min(assigned_line["left"], cell["bbox"]["x0"])
        assigned_line["right"] = max(assigned_line["right"], cell["bbox"]["x1"])

        orientation = cell["alignment"]["orientation"]
        assigned_line["orientation_counts"][orientation] += 1

        baseline_value = cell["alignment"].get("baseline_value")
        if baseline_value is not None and orientation in ("left", "right"):
            assigned_line["alignment_values"].append(baseline_value)

        # Update line center for future comparisons
        assigned_line["center_y"] = sum(assigned_line["centers"]) / len(assigned_line["centers"])

    # Finalize dominant orientation
    for line in lines:
        orientation_counts = line["orientation_counts"]
        if orientation_counts:
            total = sum(orientation_counts.values())
            dominant_orientation, dominant_count = orientation_counts.most_common(1)[0]
            if total > 0 and dominant_count / total < 0.6:
                line["dominant_orientation"] = "mixed"
            else:
                line["dominant_orientation"] = dominant_orientation
        else:
            line["dominant_orientation"] = "unknown"

    return sorted(lines, key=lambda l: l["center_y"])


def _split_cluster_into_columns(cluster_indices: List[int], boxes: List[Tuple[float, float, float, float]]) -> List[List[int]]:
    """
    Re-run clustering in horizontal mode within a vertical cluster to prevent
    multi-column blobs from being treated as a single block.
    """
    if len(cluster_indices) <= 1:
        return [cluster_indices]

    sub_boxes = [boxes[i] for i in cluster_indices]
    sub_grid = SpatialGrid(sub_boxes)
    sub_clusters = sub_grid.cluster_text(
        direction="horizontal",
        cutoff=None,
        spacing_multiplier=CLUSTER_COLUMN_SPACING_MULTIPLIER,
        align_frac=GRID_ALIGN_FRACTION,
        look_cells_radius=GRID_LOOK_RADIUS
    )

    if len(sub_clusters) <= 1:
        return [cluster_indices]

    mapped: List[List[int]] = []
    for sub_cluster in sub_clusters:
        if not sub_cluster:
            continue
        mapped.append([cluster_indices[idx] for idx in sub_cluster])
    return mapped or [cluster_indices]


def _cluster_is_cohesive(
    cluster_indices: List[int],
    cell_infos: List[Dict[str, Any]]
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Apply sanity checks (width variance + orientation dominance) to ensure
    a cluster represents a single column/run of text.

    Returns:
        (accepted, reason, stats)
    """
    stats = {}
    if not cluster_indices:
        return False, "empty_cluster", stats

    cluster_cells = [cell_infos[i] for i in cluster_indices]

    # CRITICAL FIX: Check actual LINES, not just cell count
    # Group cells into lines first to get accurate line count
    lines = _cluster_lines_for_blocks(cluster_cells)
    stats["line_count"] = len(lines)
    stats["cell_count"] = len(cluster_cells)

    if len(lines) < MIN_BLOCK_LINE_COUNT:
        return False, "below_min_lines", stats

    widths = [(cell["bbox"]["x1"] - cell["bbox"]["x0"]) for cell in cluster_cells]
    width_std = _std_dev(widths)
    stats["width_std"] = width_std

    orientation_counts = Counter(cell["alignment"]["orientation"] for cell in cluster_cells)
    stats["orientation_counts"] = orientation_counts
    if not orientation_counts:
        return False, "no_alignment_data", stats

    total = sum(orientation_counts.values())
    dominant_orientation, dominant_count = orientation_counts.most_common(1)[0]
    stats["dominant_orientation"] = dominant_orientation
    stats["dominance_share"] = dominant_count / total if total else 0.0

    passes_orientation = (
        stats["dominance_share"] >= MIN_DOMINANT_ORIENTATION_SHARE or len(cluster_cells) <= 2
    )
    passes_width = width_std <= MAX_CLUSTER_WIDTH_STD or len(cluster_cells) <= 2

    if not passes_orientation and not passes_width:
        return False, "orientation_and_width_variance", stats
    if not passes_orientation:
        return False, "orientation_variance", stats
    if not passes_width:
        return False, "width_variance", stats

    return True, None, stats



def _finalize_text_block(
    block: Dict[str, Any],
    left_baselines: List[float],
    right_baselines: List[float]
) -> Optional[Dict[str, Any]]:
    """
    Compute summary statistics for a text block and snap edges to baselines.
    """
    if not block or len(block["lines"]) < MIN_BLOCK_LINE_COUNT:
        return None

    spacing_values = block["line_spacing"]
    spacing_mean = round(sum(spacing_values) / len(spacing_values), 3) if spacing_values else 0.0

    if spacing_values:
        rounded = [round(val * 2) / 2 for val in spacing_values]
        mode_value = Counter(rounded).most_common(1)[0][0]
    else:
        mode_value = 0.0

    orientation_counts = block["orientation_counts"]
    if orientation_counts:
        total = sum(orientation_counts.values())
        dominant_orientation, dominant_count = orientation_counts.most_common(1)[0]
        if total > 0 and dominant_count / total < 0.6:
            dominant_orientation = "mixed"
    else:
        dominant_orientation = "unknown"

    baseline_value = None
    if block["alignment_values"]:
        baseline_value = float(median(block["alignment_values"]))

    line_centers = [line["center_y"] for line in block["lines"]]

    # Snap block edges to baselines based on alignment
    left_edge = block["left"]
    right_edge = block["right"]

    if dominant_orientation == "left" and left_baselines:
        # Find nearest left baseline within tolerance
        nearest_baseline = None
        min_distance = float('inf')
        for baseline in left_baselines:
            distance = abs(left_edge - baseline)
            if distance <= BASELINE_SNAP_TOLERANCE and distance < min_distance:
                nearest_baseline = baseline
                min_distance = distance
        if nearest_baseline is not None:
            left_edge = nearest_baseline

    elif dominant_orientation == "right" and right_baselines:
        # Find nearest right baseline within tolerance
        nearest_baseline = None
        min_distance = float('inf')
        for baseline in right_baselines:
            distance = abs(right_edge - baseline)
            if distance <= BASELINE_SNAP_TOLERANCE and distance < min_distance:
                nearest_baseline = baseline
                min_distance = distance
        if nearest_baseline is not None:
            right_edge = nearest_baseline

    return {
        "block_id": block["block_id"],
        "bbox": {
            "x0": left_edge,
            "y0": block["top"],
            "x1": right_edge,
            "y1": block["bottom"]
        },
        "line_count": len(block["lines"]),
        "line_spacing_values": [round(val, 3) for val in spacing_values],
        "line_spacing_mean": spacing_mean,
        "line_spacing_mode": mode_value,
        "dominant_alignment": dominant_orientation,
        "alignment_baseline": baseline_value,
        "line_centers": [round(val, 3) for val in line_centers]
    }


def _build_block_from_cluster(
    cluster_indices: List[int],
    cell_infos: List[Dict[str, Any]],
    left_baselines: List[float],
    right_baselines: List[float],
    block_id: int
) -> Optional[Dict[str, Any]]:
    """
    Convert a set of clustered cell indices into a finalized text block summary.
    """
    if not cluster_indices:
        return None

    cluster_cells = [cell_infos[i] for i in cluster_indices]
    lines = _cluster_lines_for_blocks(cluster_cells)
    if len(lines) < MIN_BLOCK_LINE_COUNT:
        return None

    # Compute spacing between line centers
    line_spacing: List[float] = []
    for idx in range(len(lines) - 1):
        gap = lines[idx + 1]["center_y"] - lines[idx]["center_y"]
        line_spacing.append(max(0.0, gap))

    orientation_counts = Counter()
    alignment_values: List[float] = []
    for line in lines:
        orientation_counts.update(line.get("orientation_counts", {}))
        alignment_values.extend(line.get("alignment_values", []))

    block = {
        "block_id": block_id,
        "lines": lines,
        "line_spacing": line_spacing,
        "orientation_counts": orientation_counts,
        "alignment_values": alignment_values,
        "top": min(line["top"] for line in lines),
        "bottom": max(line["bottom"] for line in lines),
        "left": min(line["left"] for line in lines),
        "right": max(line["right"] for line in lines),
    }
    return _finalize_text_block(block, left_baselines, right_baselines)


# =============================================================================
# MAIN TEXT BLOCK IDENTIFICATION
# =============================================================================

def identify_text_blocks(
    cells: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]],
    debug: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Identify blocks of text by clustering cells along vertical runs using
    the shared spatial grid. This replaces the previous ad-hoc segmentation
    logic with direction-aware agglomerative clustering.
    """
    if debug is not None:
        debug.clear()
        debug["inputs"] = {
            "cell_count": len(cells),
            "baseline_count": len(baselines),
        }
        debug["clusters"] = []
        debug["final_blocks"] = []

    if not cells:
        return []

    left_baselines = [
        baseline["value"]
        for baseline in baselines
        if baseline.get("orientation") == "left" and baseline.get("count", 0) >= 3
    ]
    right_baselines = [
        baseline["value"]
        for baseline in baselines
        if baseline.get("orientation") == "right" and baseline.get("count", 0) >= 3
    ]

    cell_infos: List[Dict[str, Any]] = []
    boxes: List[Tuple[float, float, float, float]] = []
    for cell in cells:
        bbox = cell.get("bbox") or {}
        if not bbox:
            continue
        center_y = (bbox["y0"] + bbox["y1"]) / 2
        alignment = _classify_cell_alignment_for_blocks(cell, left_baselines, right_baselines)
        cell_infos.append({
            "cell": cell,
            "bbox": bbox,
            "center_y": center_y,
            "alignment": alignment
        })
        boxes.append((bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]))

    if len(cell_infos) < MIN_BLOCK_LINE_COUNT:
        return []

    grid = SpatialGrid(boxes)
    typical_spacing = grid.estimate_typical_spacing(
        direction="vertical",
        align_frac=GRID_ALIGN_FRACTION,
        look_cells_radius=GRID_LOOK_RADIUS
    )
    if debug is not None:
        debug["inputs"]["typical_spacing_estimate"] = typical_spacing
        debug["inputs"]["GRID_ALIGN_FRACTION"] = GRID_ALIGN_FRACTION
        debug["inputs"]["GRID_SPACING_MULTIPLIER"] = GRID_SPACING_MULTIPLIER
        debug["inputs"]["GRID_FALLBACK_GAP"] = GRID_FALLBACK_GAP

    cutoff = GRID_FALLBACK_GAP
    if typical_spacing > 0:
        cutoff = GRID_SPACING_MULTIPLIER * typical_spacing
        # Apply minimum threshold to prevent over-fragmentation
        if cutoff < MIN_VERTICAL_CUTOFF:
            if debug is not None:
                debug["inputs"]["cutoff_before_min"] = cutoff
                debug["inputs"]["WARNING"] = f"Computed cutoff {cutoff:.2f}pt is too small (< {MIN_VERTICAL_CUTOFF}pt). Using minimum threshold to prevent over-fragmentation."
            cutoff = MIN_VERTICAL_CUTOFF
        if debug is not None:
            debug["inputs"]["cutoff_calculation"] = f"max({GRID_SPACING_MULTIPLIER} * {typical_spacing}, {MIN_VERTICAL_CUTOFF}) = {cutoff}"
            debug["inputs"]["cutoff_used"] = cutoff
    else:
        if debug is not None:
            debug["inputs"]["cutoff_used"] = cutoff
            debug["inputs"]["cutoff_source"] = "fallback (typical_spacing was 0)"

    clusters = grid.cluster_text(
        direction="vertical",
        cutoff=cutoff,
        align_frac=GRID_ALIGN_FRACTION,
        look_cells_radius=GRID_LOOK_RADIUS
    )

    if not clusters:
        return []

    refined_clusters: List[List[int]] = []
    for cluster in clusters:
        cluster_record = {
            "source_indices": cluster,
            "splits": [],
        }
        split_clusters = _split_cluster_into_columns(cluster, boxes)
        for column_cluster in split_clusters:
            accepted, reason, stats = _cluster_is_cohesive(column_cluster, cell_infos)
            bbox_summary = None
            texts = []
            if column_cluster:
                xs0 = min(boxes[i][0] for i in column_cluster)
                ys0 = min(boxes[i][1] for i in column_cluster)
                xs1 = max(boxes[i][2] for i in column_cluster)
                ys1 = max(boxes[i][3] for i in column_cluster)
                bbox_summary = {"x0": xs0, "y0": ys0, "x1": xs1, "y1": ys1}

                # Extract text content from cells for debugging
                for idx in column_cluster:
                    cell = cell_infos[idx]["cell"]
                    text = cell.get("text", "")
                    if text:
                        texts.append(text)

            cluster_record["splits"].append({
                "indices": column_cluster,
                "texts": texts,  # Add text content for debugging
                "accepted": accepted,
                "reason": reason,
                "stats": stats,
                "bbox": bbox_summary,
            })

            if accepted:
                refined_clusters.append(column_cluster)

        if debug is not None:
            debug["clusters"].append(cluster_record)

    if not refined_clusters:
        return []

    blocks: List[Dict[str, Any]] = []
    block_id = 1

    def cluster_top(indices: List[int]) -> float:
        return min(boxes[i][1] for i in indices)

    for cluster in sorted(refined_clusters, key=cluster_top):
        finalized = _build_block_from_cluster(
            cluster,
            cell_infos,
            left_baselines,
            right_baselines,
            block_id
        )
        if finalized:
            blocks.append(finalized)
            block_id += 1

    if debug is not None:
        debug["final_blocks"] = [
            {
                "block_id": block["block_id"],
                "bbox": block["bbox"],
                "line_count": block["line_count"],
                "dominant_alignment": block["dominant_alignment"],
                "alignment_baseline": block.get("alignment_baseline"),
                "line_spacing_mean": block["line_spacing_mean"],
                "line_spacing_values": block["line_spacing_values"],
            }
            for block in blocks
        ]

        # Add summary statistics
        total_splits = sum(len(cluster_rec["splits"]) for cluster_rec in debug["clusters"])
        accepted_splits = sum(
            1 for cluster_rec in debug["clusters"]
            for split in cluster_rec["splits"]
            if split["accepted"]
        )
        rejected_splits = total_splits - accepted_splits

        # Count rejection reasons
        rejection_reasons = {}
        for cluster_rec in debug["clusters"]:
            for split in cluster_rec["splits"]:
                if not split["accepted"] and split["reason"]:
                    reason = split["reason"]
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        debug["summary"] = {
            "configuration": {
                "MIN_BLOCK_LINE_COUNT": MIN_BLOCK_LINE_COUNT,
                "LINE_CLUSTER_TOLERANCE": LINE_CLUSTER_TOLERANCE,
                "BASELINE_SNAP_TOLERANCE": BASELINE_SNAP_TOLERANCE,
                "GRID_SPACING_MULTIPLIER": GRID_SPACING_MULTIPLIER,
                "GRID_FALLBACK_GAP": GRID_FALLBACK_GAP,
                "MIN_VERTICAL_CUTOFF": MIN_VERTICAL_CUTOFF,
                "MAX_CLUSTER_WIDTH_STD": MAX_CLUSTER_WIDTH_STD,
                "MIN_DOMINANT_ORIENTATION_SHARE": MIN_DOMINANT_ORIENTATION_SHARE,
                "computed_cutoff": cutoff,
            },
            "statistics": {
                "total_input_cells": len(cells),
                "total_baselines": len(baselines),
                "left_baselines_count": len(left_baselines),
                "right_baselines_count": len(right_baselines),
                "initial_clusters": len(clusters),
                "total_splits_analyzed": total_splits,
                "accepted_splits": accepted_splits,
                "rejected_splits": rejected_splits,
                "final_blocks_created": len(blocks),
            },
            "rejection_reasons": rejection_reasons,
        }

    return blocks


# =============================================================================
# DBSCAN-BASED TEXT BLOCK IDENTIFICATION
# =============================================================================

def identify_text_blocks_dbscan(
    cells: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]],
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    horizontal_scale: float = DBSCAN_HORIZONTAL_SCALE,
    debug: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Identify blocks of text using DBSCAN density-based clustering.

    This approach considers both x and y coordinates simultaneously,
    making it better at finding natural text groupings compared to
    hierarchical vertical-then-horizontal clustering.

    Uses anisotropic distance by scaling x-coordinates before clustering,
    allowing wider horizontal grouping while keeping tight vertical clustering.

    Parameters
    ----------
    cells : List[Dict]
        List of cell dictionaries with bbox and text
    baselines : List[Dict]
        Detected alignment baselines
    eps : float
        Maximum distance between samples for DBSCAN neighborhood (default: 12pt)
    min_samples : int
        Minimum samples in neighborhood for core point (default: 3)
    horizontal_scale : float
        Scale factor for x-coordinates (< 1.0 to allow wider horizontal clustering)
    debug : Optional[Dict]
        Debug info dictionary to populate

    Returns
    -------
    List[Dict]
        List of text block summaries with bbox, line_count, etc.
    """
    if not DBSCAN_AVAILABLE:
        if debug is not None:
            debug["error"] = "sklearn not available, falling back to hierarchical"
        return []

    if debug is not None:
        debug.clear()
        debug["approach"] = "dbscan"
        debug["inputs"] = {
            "cell_count": len(cells),
            "baseline_count": len(baselines),
            "eps": eps,
            "min_samples": min_samples,
            "horizontal_scale": horizontal_scale,
            "note": f"x-coordinates scaled by {horizontal_scale} to allow wider horizontal clustering"
        }
        debug["clusters"] = []
        debug["final_blocks"] = []

    if not cells:
        return []

    # Extract baselines
    left_baselines = [
        baseline["value"]
        for baseline in baselines
        if baseline.get("orientation") == "left" and baseline.get("count", 0) >= 3
    ]
    right_baselines = [
        baseline["value"]
        for baseline in baselines
        if baseline.get("orientation") == "right" and baseline.get("count", 0) >= 3
    ]

    # Prepare data: use center points for clustering
    cell_infos: List[Dict[str, Any]] = []
    points: List[Tuple[float, float]] = []

    for cell in cells:
        bbox = cell.get("bbox") or {}
        if not bbox:
            continue

        center_x = (bbox["x0"] + bbox["x1"]) / 2
        center_y = (bbox["y0"] + bbox["y1"]) / 2

        alignment = _classify_cell_alignment_for_blocks(
            cell, left_baselines, right_baselines
        )

        cell_infos.append({
            "cell": cell,
            "bbox": bbox,
            "center_x": center_x,
            "center_y": center_y,
            "alignment": alignment
        })
        # Scale x-coordinate for anisotropic clustering (allows wider horizontal grouping)
        points.append((center_x * horizontal_scale, center_y))

    if len(cell_infos) < MIN_BLOCK_LINE_COUNT:
        return []

    # Run DBSCAN clustering on scaled coordinates
    # Note: scaled x allows cells on same baseline but horizontally distant to cluster together
    X = np.array(points)
    clustering = SklearnDBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(X)

    # Group cells by cluster label
    clusters_dict: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:  # Noise points
            continue
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(idx)

    if debug is not None:
        debug["inputs"]["clusters_found"] = len(clusters_dict)
        debug["inputs"]["noise_points"] = int(np.sum(labels == -1))

    # Process each cluster into text blocks
    blocks: List[Dict[str, Any]] = []
    block_id = 1

    for cluster_label, cluster_indices in sorted(clusters_dict.items()):
        cluster_cells = [cell_infos[i] for i in cluster_indices]

        # Extract texts for debugging
        texts = [cell["cell"].get("text", "") for cell in cluster_cells if cell["cell"].get("text")]

        # Group into lines
        lines = _cluster_lines_for_blocks(cluster_cells)

        # Check if meets minimum line count
        if len(lines) < MIN_BLOCK_LINE_COUNT:
            if debug is not None:
                debug["clusters"].append({
                    "cluster_label": int(cluster_label),
                    "cell_count": len(cluster_cells),
                    "line_count": len(lines),
                    "texts": texts,
                    "accepted": False,
                    "reason": "below_min_lines"
                })
            continue

        # Build the block
        finalized = _build_block_from_cluster(
            cluster_indices,
            cell_infos,
            left_baselines,
            right_baselines,
            block_id
        )

        if finalized:
            blocks.append(finalized)
            if debug is not None:
                debug["clusters"].append({
                    "cluster_label": int(cluster_label),
                    "cell_count": len(cluster_cells),
                    "line_count": len(lines),
                    "texts": texts,
                    "accepted": True,
                    "reason": None
                })
            block_id += 1
        else:
            if debug is not None:
                debug["clusters"].append({
                    "cluster_label": int(cluster_label),
                    "cell_count": len(cluster_cells),
                    "line_count": len(lines),
                    "texts": texts,
                    "accepted": False,
                    "reason": "failed_finalization"
                })

    if debug is not None:
        debug["final_blocks"] = [
            {
                "block_id": block["block_id"],
                "bbox": block["bbox"],
                "line_count": block["line_count"],
                "dominant_alignment": block["dominant_alignment"],
                "alignment_baseline": block.get("alignment_baseline"),
                "line_spacing_mean": block["line_spacing_mean"],
                "line_spacing_values": block["line_spacing_values"],
            }
            for block in blocks
        ]

        # Add summary
        debug["summary"] = {
            "configuration": {
                "approach": "dbscan",
                "eps": eps,
                "min_samples": min_samples,
                "MIN_BLOCK_LINE_COUNT": MIN_BLOCK_LINE_COUNT,
                "LINE_CLUSTER_TOLERANCE": LINE_CLUSTER_TOLERANCE,
            },
            "statistics": {
                "total_input_cells": len(cells),
                "dbscan_clusters_found": len(clusters_dict),
                "noise_points": int(np.sum(labels == -1)),
                "accepted_clusters": len(blocks),
                "final_blocks_created": len(blocks),
            }
        }

    return blocks


# =============================================================================
# TABLE-AWARE CLUSTERING UTILITIES
# =============================================================================

def extract_table_metadata(docling_payload: Dict[str, Any], page_no: int) -> List[Dict[str, Any]]:
    """
    Extract table structure metadata from Docling payload for a specific page.

    Returns a list of table metadata dictionaries, each containing:
    - table_id: Unique identifier for the table
    - page: Page number
    - bbox: Table bounding box
    - num_rows: Number of rows in table
    - num_cols: Number of columns in table
    - cells: List of cell dictionaries with bbox, row_idx, col_idx, text

    Parameters
    ----------
    docling_payload : Dict
        The Docling payload containing table information
    page_no : int
        Page number (0-indexed)

    Returns
    -------
    List[Dict]
        List of table metadata for the specified page
    """
    tables_metadata = []

    if not isinstance(docling_payload, dict):
        return tables_metadata

    doc_struct = docling_payload.get("docling_document")
    if not isinstance(doc_struct, dict):
        return tables_metadata

    tables = doc_struct.get("tables", [])
    if not isinstance(tables, list):
        return tables_metadata

    table_id = 0
    for table in tables:
        if not isinstance(table, dict):
            continue

        # Check if table belongs to this page
        prov_entries = table.get("prov", [])
        table_page = None

        if isinstance(prov_entries, list):
            for prov in prov_entries:
                if not isinstance(prov, dict):
                    continue
                prov_page_no = prov.get("page_no")
                if prov_page_no is None:
                    continue
                try:
                    prov_page_no = int(prov_page_no)
                except (TypeError, ValueError):
                    continue
                # Convert to 0-indexed
                table_page = prov_page_no - 1 if prov_page_no > 0 else prov_page_no
                break

        if table_page != page_no:
            continue

        # Extract table data
        data = table.get("data", {})
        num_rows = data.get("num_rows", 0)
        num_cols = data.get("num_cols", 0)
        grid = data.get("grid", [])

        if not isinstance(grid, list) or not grid:
            continue

        # Extract cells with row/column indices
        table_cells = []
        table_bbox = None

        for row_idx, row in enumerate(grid):
            if not isinstance(row, list):
                continue

            for cell in row:
                if not isinstance(cell, dict):
                    continue

                bbox = cell.get("bbox")
                if not isinstance(bbox, dict):
                    continue

                # Get row and column indices from the cell
                start_row = cell.get("start_row_offset_idx", row_idx)
                end_row = cell.get("end_row_offset_idx", row_idx + 1)
                start_col = cell.get("start_col_offset_idx", 0)
                end_col = cell.get("end_col_offset_idx", 1)

                # Convert bbox to standard format
                try:
                    # Docling uses TOPLEFT origin with l, t, r, b
                    x0 = float(bbox.get("l", 0))
                    y0 = float(bbox.get("t", 0))
                    x1 = float(bbox.get("r", 0))
                    y1 = float(bbox.get("b", 0))

                    cell_bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

                    # Update table bbox
                    if table_bbox is None:
                        table_bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
                    else:
                        table_bbox["x0"] = min(table_bbox["x0"], x0)
                        table_bbox["y0"] = min(table_bbox["y0"], y0)
                        table_bbox["x1"] = max(table_bbox["x1"], x1)
                        table_bbox["y1"] = max(table_bbox["y1"], y1)

                    table_cells.append({
                        "bbox": cell_bbox,
                        "text": cell.get("text", ""),
                        "start_row": start_row,
                        "end_row": end_row,
                        "start_col": start_col,
                        "end_col": end_col,
                    })

                except (TypeError, ValueError):
                    continue

        if table_cells:
            tables_metadata.append({
                "table_id": table_id,
                "page": page_no,
                "bbox": table_bbox,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "cells": table_cells,
            })
            table_id += 1

    return tables_metadata


def enrich_cells_with_table_info(
    cells: List[Dict[str, Any]],
    tables: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enrich cells with table membership and column information.

    For each cell, adds:
    - table_id: ID of table containing this cell (None if not in table)
    - column_id: Column index within table (None if not in table)
    - row_id: Row index within table (None if not in table)

    Parameters
    ----------
    cells : List[Dict]
        List of cell dictionaries with bbox
    tables : List[Dict]
        List of table metadata from extract_table_metadata

    Returns
    -------
    List[Dict]
        Enriched cells with table metadata
    """
    enriched_cells = []

    for cell in cells:
        enriched_cell = cell.copy()
        enriched_cell["table_id"] = None
        enriched_cell["column_id"] = None
        enriched_cell["row_id"] = None

        cell_bbox = cell.get("bbox")
        if not cell_bbox:
            enriched_cells.append(enriched_cell)
            continue

        cell_x0 = cell_bbox["x0"]
        cell_y0 = cell_bbox["y0"]
        cell_x1 = cell_bbox["x1"]
        cell_y1 = cell_bbox["y1"]

        # Find which table this cell belongs to
        for table in tables:
            table_bbox = table.get("bbox")
            if not table_bbox:
                continue

            # Check if cell is within table bounds (with small tolerance)
            tolerance = 2.0
            if (cell_x0 >= table_bbox["x0"] - tolerance and
                cell_x1 <= table_bbox["x1"] + tolerance and
                cell_y0 >= table_bbox["y0"] - tolerance and
                cell_y1 <= table_bbox["y1"] + tolerance):

                # Find which table cell this matches
                table_cells = table.get("cells", [])
                best_match = None
                best_iou = 0.0

                for table_cell in table_cells:
                    tc_bbox = table_cell["bbox"]

                    # Calculate IoU (Intersection over Union)
                    x_overlap = max(0, min(cell_x1, tc_bbox["x1"]) - max(cell_x0, tc_bbox["x0"]))
                    y_overlap = max(0, min(cell_y1, tc_bbox["y1"]) - max(cell_y0, tc_bbox["y0"]))
                    intersection = x_overlap * y_overlap

                    cell_area = (cell_x1 - cell_x0) * (cell_y1 - cell_y0)
                    tc_area = (tc_bbox["x1"] - tc_bbox["x0"]) * (tc_bbox["y1"] - tc_bbox["y0"])
                    union = cell_area + tc_area - intersection

                    iou = intersection / union if union > 0 else 0

                    # Require at least 50% IoU to consider a match
                    if iou > 0.5 and iou > best_iou:
                        best_iou = iou
                        best_match = table_cell

                if best_match:
                    enriched_cell["table_id"] = table["table_id"]
                    enriched_cell["column_id"] = best_match["start_col"]
                    enriched_cell["row_id"] = best_match["start_row"]
                    break  # Found a match, stop searching

        enriched_cells.append(enriched_cell)

    return enriched_cells


def table_aware_distance(
    cell_a: Dict[str, Any],
    cell_b: Dict[str, Any],
    dimension: str
) -> float:
    """
    Calculate distance between cells with table-awareness.

    If both cells are in the same table but different columns:
    - Return INFINITY for horizontal clustering (prevent cross-column clustering)
    - Return normal distance for vertical clustering (allow rows across columns)

    Parameters
    ----------
    cell_a : Dict
        First cell with bbox and optional table metadata
    cell_b : Dict
        Second cell with bbox and optional table metadata
    dimension : str
        "vertical" or "horizontal"

    Returns
    -------
    float
        Distance between cells (np.inf for blocked cross-column clustering)
    """
    table_a = cell_a.get("table_id")
    table_b = cell_b.get("table_id")
    col_a = cell_a.get("column_id")
    col_b = cell_b.get("column_id")

    # Both cells in same table?
    if table_a is not None and table_a == table_b:
        # Different columns?
        if col_a is not None and col_b is not None and col_a != col_b:
            # For horizontal clustering: prevent cross-column grouping
            if dimension == "horizontal":
                return np.inf
            # For vertical clustering: allow (to detect rows)
            # Fall through to normal distance

    # Normal euclidean distance
    bbox_a = cell_a["bbox"]
    bbox_b = cell_b["bbox"]

    if dimension == "vertical":
        center_a = (bbox_a["y0"] + bbox_a["y1"]) / 2.0
        center_b = (bbox_b["y0"] + bbox_b["y1"]) / 2.0
        return abs(center_a - center_b)
    else:  # horizontal
        center_a = (bbox_a["x0"] + bbox_a["x1"]) / 2.0
        center_b = (bbox_b["x0"] + bbox_b["x1"]) / 2.0
        return abs(center_a - center_b)


# =============================================================================
# TWO-STAGE CLUSTERING (VERTICAL THEN HORIZONTAL)
# =============================================================================

def identify_text_blocks_two_stage(
    cells: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]],
    vertical_method: str = TWO_STAGE_VERTICAL_METHOD,
    horizontal_method: str = TWO_STAGE_HORIZONTAL_METHOD,
    vertical_eps: float = TWO_STAGE_VERTICAL_EPS,
    horizontal_eps: float = TWO_STAGE_HORIZONTAL_EPS,
    min_samples: int = TWO_STAGE_MIN_SAMPLES,
    debug: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Identify text blocks using two-stage clustering: vertical first, then horizontal.

    Stage 1 (Vertical): Cluster cells by y-coordinates to find horizontal "bands"
    Stage 2 (Horizontal): Within each band, cluster by x-coordinates to separate columns

    Parameters
    ----------
    cells : List[Dict]
        List of cell dictionaries with bbox and text
    baselines : List[Dict]
        Detected alignment baselines
    vertical_method : str
        Method for vertical clustering: "dbscan" or "agglomerative"
    horizontal_method : str
        Method for horizontal clustering: "dbscan" or "agglomerative"
    vertical_eps : float
        Vertical clustering threshold (default: 12pt)
    horizontal_eps : float
        Horizontal clustering threshold (default: 50pt)
    min_samples : int
        Minimum samples for DBSCAN core point (default: 2)
    debug : Optional[Dict]
        Debug info dictionary to populate

    Returns
    -------
    List[Dict]
        List of text block summaries with bbox, line_count, etc.
    """
    if not SKLEARN_AVAILABLE:
        if debug is not None:
            debug["error"] = "sklearn not available"
        return []

    if debug is not None:
        debug.clear()
        debug["approach"] = "two_stage"
        debug["inputs"] = {
            "cell_count": len(cells),
            "baseline_count": len(baselines),
            "vertical_method": vertical_method,
            "horizontal_method": horizontal_method,
            "vertical_eps": vertical_eps,
            "horizontal_eps": horizontal_eps,
            "min_samples": min_samples,
        }
        debug["vertical_clusters"] = []
        debug["final_blocks"] = []

    if not cells:
        return []

    # Extract baselines
    left_baselines = [
        baseline["value"]
        for baseline in baselines
        if baseline.get("orientation") == "left" and baseline.get("count", 0) >= 3
    ]
    right_baselines = [
        baseline["value"]
        for baseline in baselines
        if baseline.get("orientation") == "right" and baseline.get("count", 0) >= 3
    ]

    # Prepare cell data
    cell_infos: List[Dict[str, Any]] = []
    for cell in cells:
        bbox = cell.get("bbox") or {}
        if not bbox:
            continue

        center_x = (bbox["x0"] + bbox["x1"]) / 2
        center_y = (bbox["y0"] + bbox["y1"]) / 2

        alignment = _classify_cell_alignment_for_blocks(
            cell, left_baselines, right_baselines
        )

        cell_infos.append({
            "cell": cell,
            "bbox": bbox,
            "center_x": center_x,
            "center_y": center_y,
            "alignment": alignment
        })

    if len(cell_infos) < MIN_BLOCK_LINE_COUNT:
        return []

    # STAGE 1: Vertical clustering (cluster by y-coordinate)
    y_coords = np.array([[info["center_y"]] for info in cell_infos])

    if vertical_method == "dbscan":
        vert_clustering = SklearnDBSCAN(eps=vertical_eps, min_samples=min_samples, metric='euclidean')
        vert_labels = vert_clustering.fit_predict(y_coords)
    elif vertical_method == "agglomerative":
        vert_clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=vertical_eps, linkage='single'
        )
        vert_labels = vert_clustering.fit_predict(y_coords)
    else:
        raise ValueError(f"Unknown vertical method: {vertical_method}")

    # Group cells by vertical cluster
    vert_clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(vert_labels):
        if label == -1:  # Noise (only for DBSCAN)
            continue
        if label not in vert_clusters:
            vert_clusters[label] = []
        vert_clusters[label].append(idx)

    if debug is not None:
        debug["inputs"]["vertical_clusters_found"] = len(vert_clusters)
        debug["inputs"]["vertical_noise_points"] = int(np.sum(vert_labels == -1))

    # STAGE 2: Horizontal clustering within each vertical cluster
    all_blocks: List[Dict[str, Any]] = []
    block_id = 1

    for vert_label, vert_indices in vert_clusters.items():
        if len(vert_indices) < MIN_BLOCK_LINE_COUNT:
            continue

        # Extract x-coordinates for this vertical cluster
        x_coords = np.array([[cell_infos[idx]["center_x"]] for idx in vert_indices])

        if horizontal_method == "dbscan":
            horiz_clustering = SklearnDBSCAN(eps=horizontal_eps, min_samples=min_samples, metric='euclidean')
            horiz_labels = horiz_clustering.fit_predict(x_coords)
        elif horizontal_method == "agglomerative":
            horiz_clustering = AgglomerativeClustering(
                n_clusters=None, distance_threshold=horizontal_eps, linkage='single'
            )
            horiz_labels = horiz_clustering.fit_predict(x_coords)
        else:
            raise ValueError(f"Unknown horizontal method: {horizontal_method}")

        # Group by horizontal cluster
        horiz_clusters: Dict[int, List[int]] = {}
        for local_idx, label in enumerate(horiz_labels):
            if label == -1:  # Noise (only for DBSCAN)
                continue
            if label not in horiz_clusters:
                horiz_clusters[label] = []
            horiz_clusters[label].append(vert_indices[local_idx])

        # Create blocks from horizontal clusters
        for horiz_label, cell_indices in horiz_clusters.items():
            cluster_cells = [cell_infos[i] for i in cell_indices]

            # Group into lines
            lines = _cluster_lines_for_blocks(cluster_cells)

            if len(lines) < MIN_BLOCK_LINE_COUNT:
                continue

            # Calculate block bounding box
            x0 = min(c["bbox"]["x0"] for c in cluster_cells)
            x1 = max(c["bbox"]["x1"] for c in cluster_cells)
            y0 = min(c["bbox"]["y0"] for c in cluster_cells)
            y1 = max(c["bbox"]["y1"] for c in cluster_cells)

            # Determine dominant alignment
            alignments = [c["alignment"]["orientation"] for c in cluster_cells]
            alignment_counts = Counter(alignments)
            dominant_alignment = alignment_counts.most_common(1)[0][0]

            # Calculate line spacing
            if len(lines) > 1:
                y_centers = sorted([
                    (line["cells"][0]["bbox"]["y0"] + line["cells"][0]["bbox"]["y1"]) / 2
                    for line in lines
                ])
                spacings = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]
                spacing_mean = round(sum(spacings) / len(spacings), 3)
                spacing_values = [round(s, 3) for s in spacings]
            else:
                spacing_mean = 0.0
                spacing_values = []

            # Snap to baselines
            alignment_baseline = None
            if dominant_alignment == "left" and left_baselines:
                closest_left = min(left_baselines, key=lambda b: abs(b - x0))
                if abs(closest_left - x0) < BASELINE_SNAP_TOLERANCE:
                    alignment_baseline = closest_left
            elif dominant_alignment == "right" and right_baselines:
                closest_right = min(right_baselines, key=lambda b: abs(b - x1))
                if abs(closest_right - x1) < BASELINE_SNAP_TOLERANCE:
                    alignment_baseline = closest_right

            block = {
                "block_id": block_id,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "line_count": len(lines),
                "dominant_alignment": dominant_alignment,
                "alignment_baseline": alignment_baseline,
                "line_spacing_mean": spacing_mean,
                "line_spacing_values": spacing_values,
            }

            all_blocks.append(block)
            block_id += 1

            if debug is not None:
                texts = [c["cell"].get("text", "") for c in cluster_cells if c["cell"].get("text")]
                debug["vertical_clusters"].append({
                    "vertical_label": int(vert_label),
                    "horizontal_label": int(horiz_label),
                    "cell_count": len(cluster_cells),
                    "line_count": len(lines),
                    "texts": texts[:10],  # First 10 texts
                    "block_id": block_id - 1,
                })

    if debug is not None:
        debug["final_blocks"] = [
            {
                "block_id": block["block_id"],
                "bbox": block["bbox"],
                "line_count": block["line_count"],
                "dominant_alignment": block["dominant_alignment"],
                "alignment_baseline": block.get("alignment_baseline"),
                "line_spacing_mean": block["line_spacing_mean"],
                "line_spacing_values": block["line_spacing_values"],
            }
            for block in all_blocks
        ]

        debug["summary"] = {
            "configuration": {
                "approach": "two_stage",
                "vertical_method": vertical_method,
                "horizontal_method": horizontal_method,
                "vertical_eps": vertical_eps,
                "horizontal_eps": horizontal_eps,
                "min_samples": min_samples,
            },
            "statistics": {
                "total_input_cells": len(cells),
                "vertical_clusters_found": len(vert_clusters),
                "final_blocks_created": len(all_blocks),
            }
        }

    return all_blocks


def identify_text_blocks_iterative(
    cells: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]],
    tables: Optional[List[Dict[str, Any]]] = None,
    vertical_eps: float = ITERATIVE_VERTICAL_EPS,
    horizontal_eps: float = ITERATIVE_HORIZONTAL_EPS,
    min_samples: int = ITERATIVE_MIN_SAMPLES,
    max_iterations: int = ITERATIVE_MAX_ITERATIONS,
    min_block_cells: int = ITERATIVE_MIN_BLOCK_CELLS,
    use_table_awareness: bool = True,
    debug: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Identify text blocks using iterative multi-pass clustering.

    Alternates between vertical (y-coordinate) and horizontal (x-coordinate)
    DBSCAN clustering until no new splits occur (convergence).

    Supports table-aware clustering to respect table column boundaries.

    Algorithm:
    ---------
    1. Start with all cells in one block
    2. If tables provided and use_table_awareness=True:
       - Enrich cells with table metadata (table_id, column_id, row_id)
    3. Iteration loop:
       - Odd iterations: Cluster by y-coordinate (vertical)
       - Even iterations: Cluster by x-coordinate (horizontal)
       - For each current block:
         - Apply DBSCAN clustering
         - If table-aware and horizontal: Use precomputed distance matrix
           with INFINITY penalty for cross-column clustering
         - If multiple sub-blocks created → split occurred
       - Track if ANY splits occurred this iteration
    4. Stop when:
       - No splits occurred (converged), OR
       - Max iterations reached, OR
       - All blocks below minimum size

    Parameters
    ----------
    cells : List[Dict]
        List of cell dictionaries with bbox and text
    baselines : List[Dict]
        Detected alignment baselines
    tables : Optional[List[Dict]]
        List of table metadata from extract_table_metadata (default: None)
    vertical_eps : float
        Vertical clustering threshold (default: 12pt)
    horizontal_eps : float
        Horizontal clustering threshold (default: 40pt)
    min_samples : int
        Minimum samples for DBSCAN core point (default: 2)
    max_iterations : int
        Maximum iterations before stopping (default: 3)
    min_block_cells : int
        Minimum cells per block to prevent over-fragmentation (default: 2)
    use_table_awareness : bool
        Use table-aware distance to prevent cross-column clustering (default: True)
    debug : Optional[Dict]
        Debug info dictionary to populate

    Returns
    -------
    List[Dict]
        List of text block summaries with bbox, line_count, etc.
    """
    if not SKLEARN_AVAILABLE:
        if debug is not None:
            debug["error"] = "sklearn not available"
        return []

    if debug is not None:
        debug.clear()
        debug["approach"] = "iterative"
        debug["inputs"] = {
            "cell_count": len(cells),
            "baseline_count": len(baselines),
            "table_count": len(tables) if tables else 0,
            "use_table_awareness": use_table_awareness,
            "vertical_eps": vertical_eps,
            "horizontal_eps": horizontal_eps,
            "min_samples": min_samples,
            "max_iterations": max_iterations,
            "min_block_cells": min_block_cells,
        }
        debug["iterations"] = []

    if not cells:
        return []

    # Enrich cells with table metadata if tables provided and table-awareness enabled
    if tables and use_table_awareness:
        cells = enrich_cells_with_table_info(cells, tables)
        if debug is not None:
            # Count how many cells are in tables
            cells_in_tables = sum(1 for cell in cells if cell.get("table_id") is not None)
            debug["inputs"]["cells_in_tables"] = cells_in_tables
            debug["inputs"]["cells_not_in_tables"] = len(cells) - cells_in_tables

    # Extract baselines
    left_baselines = [
        baseline["value"]
        for baseline in baselines
        if baseline.get("orientation") == "left" and baseline.get("count", 0) >= 3
    ]
    right_baselines = [
        baseline["value"]
        for baseline in baselines
        if baseline.get("orientation") == "right" and baseline.get("count", 0) >= 3
    ]

    # Prepare cell data
    cell_infos: List[Dict[str, Any]] = []
    for cell in cells:
        bbox = cell.get("bbox") or {}
        if not bbox:
            continue

        center_x = (bbox["x0"] + bbox["x1"]) / 2
        center_y = (bbox["y0"] + bbox["y1"]) / 2

        alignment = _classify_cell_alignment_for_blocks(
            cell, left_baselines, right_baselines
        )

        cell_infos.append({
            "cell": cell,
            "bbox": bbox,
            "center_x": center_x,
            "center_y": center_y,
            "alignment": alignment
        })

    if len(cell_infos) < min_block_cells:
        return []

    # Initialize: Start with all cells as one "block"
    current_blocks = [list(range(len(cell_infos)))]  # List of lists of cell indices

    iteration = 0
    converged = False
    convergence_reason = None

    while iteration < max_iterations:
        iteration += 1
        dimension = "vertical" if iteration % 2 == 1 else "horizontal"
        eps = vertical_eps if dimension == "vertical" else horizontal_eps

        new_blocks = []
        splits_occurred = False
        blocks_split_this_iteration = 0

        for block_indices in current_blocks:
            # Skip blocks below minimum size
            if len(block_indices) < min_block_cells:
                new_blocks.append(block_indices)
                continue

            # Apply DBSCAN clustering
            # Use table-aware distance for horizontal clustering if enabled
            use_table_aware = (tables and use_table_awareness and dimension == "horizontal")

            if use_table_aware:
                # Build precomputed distance matrix with table awareness
                n = len(block_indices)
                dist_matrix = np.zeros((n, n))

                for i in range(n):
                    for j in range(i + 1, n):
                        cell_i = cell_infos[block_indices[i]]
                        cell_j = cell_infos[block_indices[j]]
                        dist = table_aware_distance(cell_i, cell_j, dimension)
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist

                clustering = SklearnDBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                labels = clustering.fit_predict(dist_matrix)
            else:
                # Standard euclidean distance clustering
                if dimension == "vertical":
                    coords = np.array([[cell_infos[idx]["center_y"]] for idx in block_indices])
                else:  # horizontal
                    coords = np.array([[cell_infos[idx]["center_x"]] for idx in block_indices])

                clustering = SklearnDBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                labels = clustering.fit_predict(coords)

            # Group by cluster label
            sub_blocks: Dict[int, List[int]] = {}
            noise_indices = []

            for local_idx, label in enumerate(labels):
                global_idx = block_indices[local_idx]
                if label == -1:  # Noise point
                    noise_indices.append(global_idx)
                else:
                    if label not in sub_blocks:
                        sub_blocks[label] = []
                    sub_blocks[label].append(global_idx)

            # Handle noise points: keep them in the original block (conservative approach)
            # Create a "remainder" block for noise if there are any
            if noise_indices and len(noise_indices) >= min_block_cells:
                new_blocks.append(noise_indices)

            # Check if block was split
            if len(sub_blocks) > 1:
                # Block split into multiple sub-blocks
                for sub_block_indices in sub_blocks.values():
                    if len(sub_block_indices) >= min_block_cells:
                        new_blocks.append(sub_block_indices)
                splits_occurred = True
                blocks_split_this_iteration += 1
            elif len(sub_blocks) == 1:
                # No split - keep original block (add back noise if any)
                combined_indices = list(sub_blocks.values())[0]
                if noise_indices and len(noise_indices) < min_block_cells:
                    # Add noise back to the main block
                    combined_indices.extend(noise_indices)
                new_blocks.append(combined_indices)
            else:
                # All noise (shouldn't happen with min_samples=2, but handle it)
                if noise_indices:
                    new_blocks.append(noise_indices)

        # Update blocks for next iteration
        blocks_before = len(current_blocks)
        current_blocks = new_blocks
        blocks_after = len(current_blocks)

        # Record iteration debug info
        if debug is not None:
            debug["iterations"].append({
                "iteration": iteration,
                "dimension": dimension,
                "eps": eps,
                "blocks_before": blocks_before,
                "blocks_after": blocks_after,
                "splits_occurred": splits_occurred,
                "blocks_split": blocks_split_this_iteration,
            })

        # Check convergence
        if not splits_occurred:
            converged = True
            convergence_reason = "no_splits"
            break

    # Check if stopped due to max iterations
    if not converged:
        convergence_reason = "max_iterations"

    # Finalize blocks: Convert indices to full block dictionaries
    final_blocks = []
    eliminated_blocks = []  # Track blocks that were filtered out
    block_id = 1
    candidate_id = 1

    for block_indices in current_blocks:
        # Check minimum cell count
        if len(block_indices) < min_block_cells:
            cluster_cells = [cell_infos[i] for i in block_indices]
            x0 = min(c["bbox"]["x0"] for c in cluster_cells)
            x1 = max(c["bbox"]["x1"] for c in cluster_cells)
            y0 = min(c["bbox"]["y0"] for c in cluster_cells)
            y1 = max(c["bbox"]["y1"] for c in cluster_cells)
            texts = [c["cell"].get("text", "") for c in cluster_cells if c["cell"].get("text")]

            eliminated_blocks.append({
                "candidate_id": candidate_id,
                "cell_count": len(cluster_cells),
                "line_count": 0,  # Not calculated yet
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "texts": texts[:5],
                "elimination_reason": f"too_few_cells (< {min_block_cells})",
            })
            candidate_id += 1
            continue

        cluster_cells = [cell_infos[i] for i in block_indices]

        # Group into lines
        lines = _cluster_lines_for_blocks(cluster_cells)

        if len(lines) < MIN_BLOCK_LINE_COUNT:
            x0 = min(c["bbox"]["x0"] for c in cluster_cells)
            x1 = max(c["bbox"]["x1"] for c in cluster_cells)
            y0 = min(c["bbox"]["y0"] for c in cluster_cells)
            y1 = max(c["bbox"]["y1"] for c in cluster_cells)
            texts = [c["cell"].get("text", "") for c in cluster_cells if c["cell"].get("text")]

            eliminated_blocks.append({
                "candidate_id": candidate_id,
                "cell_count": len(cluster_cells),
                "line_count": len(lines),
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "texts": texts[:5],
                "elimination_reason": f"too_few_lines (< {MIN_BLOCK_LINE_COUNT})",
            })
            candidate_id += 1
            continue

        # Calculate initial block bounding box
        x0 = min(c["bbox"]["x0"] for c in cluster_cells)
        x1 = max(c["bbox"]["x1"] for c in cluster_cells)
        y0 = min(c["bbox"]["y0"] for c in cluster_cells)
        y1 = max(c["bbox"]["y1"] for c in cluster_cells)

        # Analyze alignment (UC-001 / Phase 2)
        alignment_analysis = analyze_block_alignment(
            cluster_cells, left_baselines, right_baselines, BASELINE_SNAP_TOLERANCE
        )

        # Extract alignment metadata
        alignment_type = alignment_analysis["alignment_type"]
        alignment_confidence = alignment_analysis["confidence"]
        alignment_baseline = alignment_analysis["baseline_value"]
        alignment_deviations = alignment_analysis["deviations"]

        # Baseline-aware snapping (UC-001 / Phase 2)
        # If alignment is strong (≥70% confidence), snap block edge to baseline
        if alignment_baseline is not None and alignment_confidence >= 0.70:
            if alignment_type == "right":
                # Snap right edge to baseline instead of using max(x1)
                x1 = alignment_baseline
            elif alignment_type == "left":
                # Snap left edge to baseline instead of using min(x0)
                x0 = alignment_baseline

        # Determine dominant alignment (legacy - for backwards compatibility)
        # Use alignment_type from analysis
        alignments = [c["alignment"]["orientation"] for c in cluster_cells]
        alignment_counts = Counter(alignments)
        dominant_alignment_legacy = alignment_counts.most_common(1)[0][0]

        # Calculate line spacing
        if len(lines) > 1:
            y_centers = sorted([
                (line["cells"][0]["bbox"]["y0"] + line["cells"][0]["bbox"]["y1"]) / 2
                for line in lines
            ])
            spacings = [y_centers[i+1] - y_centers[i] for i in range(len(y_centers)-1)]
            spacing_mean = round(sum(spacings) / len(spacings), 3)
            spacing_values = [round(s, 3) for s in spacings]
        else:
            spacing_mean = 0.0
            spacing_values = []

        # Extract text samples
        texts = [c["cell"].get("text", "") for c in cluster_cells if c["cell"].get("text")]

        # Build block with enhanced alignment metadata (UC-001 / Phase 2)
        block = {
            "block_id": block_id,
            "candidate_id": candidate_id,  # Track which candidate this was
            "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "line_count": len(lines),
            "cell_count": len(cluster_cells),
            "dominant_alignment": alignment_type,  # Use analyzed alignment type
            "alignment_baseline": alignment_baseline,
            "line_spacing_mean": spacing_mean,
            "line_spacing_values": spacing_values,
            "texts": texts[:10],  # First 10 texts for debugging
            # UC-001 / Phase 2 enhancements
            "alignment_metadata": {
                "alignment_type": alignment_type,
                "confidence": alignment_confidence,
                "baseline_value": alignment_baseline,
                "aligned_count": alignment_analysis["aligned_count"],
                "total_count": alignment_analysis["total_count"],
                "deviation_count": len(alignment_deviations),
                "deviations": alignment_deviations,  # Full deviation details
            }
        }

        final_blocks.append(block)
        block_id += 1
        candidate_id += 1

    # Add convergence info to debug
    if debug is not None:
        debug["convergence"] = {
            "converged": converged,
            "iterations_used": iteration,
            "final_blocks": len(final_blocks),
            "convergence_reason": convergence_reason,
        }

        debug["eliminated_blocks"] = eliminated_blocks

        debug["summary"] = {
            "configuration": {
                "approach": "iterative",
                "vertical_eps": vertical_eps,
                "horizontal_eps": horizontal_eps,
                "min_samples": min_samples,
                "max_iterations": max_iterations,
                "min_block_cells": min_block_cells,
            },
            "statistics": {
                "total_input_cells": len(cells),
                "iterations_used": iteration,
                "converged": converged,
                "candidate_blocks": len(current_blocks),
                "eliminated_blocks": len(eliminated_blocks),
                "final_blocks_created": len(final_blocks),
            }
        }

    return final_blocks


# =============================================================================
# ALIGNMENT ANALYSIS AND ANOMALY DETECTION (UC-001 / PHASE 2)
# =============================================================================

def calculate_alignment_confidence(
    cluster_cells: List[Dict[str, Any]],
    alignment_type: str,
    baseline_value: float,
    tolerance: float = BASELINE_SNAP_TOLERANCE
) -> float:
    """
    Calculate confidence score for alignment (percentage of cells aligned to baseline).

    For a given alignment type and baseline, determine what percentage of cells
    in the cluster align to that baseline within tolerance.

    Parameters
    ----------
    cluster_cells : List[Dict]
        List of cells in the cluster (with bbox and alignment info)
    alignment_type : str
        "left", "right", "center", or "mixed"
    baseline_value : float
        The baseline coordinate value to check against
    tolerance : float
        Maximum deviation to consider aligned (default: BASELINE_SNAP_TOLERANCE)

    Returns
    -------
    float
        Confidence score from 0.0 to 1.0 (percentage aligned)
    """
    if not cluster_cells or alignment_type in ("center", "mixed"):
        # Center/mixed alignments don't have a clear baseline
        return 0.0

    aligned_count = 0

    for cell in cluster_cells:
        bbox = cell.get("bbox", {})

        if alignment_type == "left":
            edge_value = bbox.get("x0", 0.0)
        elif alignment_type == "right":
            edge_value = bbox.get("x1", 0.0)
        else:
            continue

        # Check if cell aligns to baseline
        if abs(edge_value - baseline_value) <= tolerance:
            aligned_count += 1

    confidence = aligned_count / len(cluster_cells) if cluster_cells else 0.0
    return round(confidence, 3)


def detect_baseline_deviations(
    cluster_cells: List[Dict[str, Any]],
    alignment_type: str,
    baseline_value: float,
    tolerance: float = BASELINE_SNAP_TOLERANCE
) -> List[Dict[str, Any]]:
    """
    Detect cells that deviate from the expected baseline alignment.

    For a given alignment type and baseline, find all cells that DON'T align
    to that baseline (deviation > tolerance).

    Parameters
    ----------
    cluster_cells : List[Dict]
        List of cells in the cluster (with bbox, text, and alignment info)
    alignment_type : str
        "left", "right", "center", or "mixed"
    baseline_value : float
        The baseline coordinate value to check against
    tolerance : float
        Maximum deviation to consider aligned (default: BASELINE_SNAP_TOLERANCE)

    Returns
    -------
    List[Dict]
        List of deviation dictionaries with:
        - cell_text: Text content of deviating cell
        - expected_edge: Expected baseline value
        - actual_edge: Actual edge coordinate
        - deviation: Signed deviation (positive = extends beyond baseline)
        - cell_bbox: Full bbox of deviating cell
    """
    deviations = []

    if alignment_type not in ("left", "right"):
        # Only left/right alignments have clear baselines
        return deviations

    for cell in cluster_cells:
        bbox = cell.get("bbox", {})
        text = cell.get("cell", {}).get("text", "") if "cell" in cell else cell.get("text", "")

        if alignment_type == "left":
            edge_value = bbox.get("x0", 0.0)
        else:  # right
            edge_value = bbox.get("x1", 0.0)

        # Calculate deviation
        deviation_value = edge_value - baseline_value

        # Flag if deviation exceeds tolerance
        if abs(deviation_value) > tolerance:
            deviations.append({
                "cell_text": text,
                "expected_edge": round(baseline_value, 2),
                "actual_edge": round(edge_value, 2),
                "deviation": round(deviation_value, 2),
                "cell_bbox": bbox,
            })

    return deviations


def analyze_block_alignment(
    cluster_cells: List[Dict[str, Any]],
    left_baselines: List[float],
    right_baselines: List[float],
    tolerance: float = BASELINE_SNAP_TOLERANCE
) -> Dict[str, Any]:
    """
    Analyze alignment type, confidence, and deviations for a block of cells.

    Determines whether the block is left-aligned, right-aligned, center-aligned,
    or mixed. Calculates confidence based on percentage of cells aligned to the
    dominant baseline, and identifies cells that deviate from expected alignment.

    Algorithm:
    1. For each cell, check if left edge aligns to any left baseline
    2. For each cell, check if right edge aligns to any right baseline
    3. Count: left_aligned, right_aligned, neither (center)
    4. Determine dominant type:
       - If ≥70% align right → "right"
       - If ≥70% align left → "left"
       - If <30% align to either → "center"
       - Otherwise → "mixed"
    5. For dominant type, find most common baseline
    6. Calculate confidence (percentage aligned to dominant baseline)
    7. Detect deviations (cells not aligned to dominant baseline)

    Parameters
    ----------
    cluster_cells : List[Dict]
        List of cells in the cluster (with bbox, text, and alignment info)
    left_baselines : List[float]
        Detected left alignment baselines (x-coordinates)
    right_baselines : List[float]
        Detected right alignment baselines (x-coordinates)
    tolerance : float
        Maximum deviation to consider aligned (default: BASELINE_SNAP_TOLERANCE)

    Returns
    -------
    Dict
        {
            "alignment_type": "left" | "right" | "center" | "mixed",
            "confidence": 0.0-1.0,
            "baseline_value": float | None,
            "aligned_count": int,
            "total_count": int,
            "deviations": [
                {
                    "cell_text": str,
                    "expected_edge": float,
                    "actual_edge": float,
                    "deviation": float,
                    "cell_bbox": Dict
                }
            ]
        }
    """
    if not cluster_cells:
        return {
            "alignment_type": "mixed",
            "confidence": 0.0,
            "baseline_value": None,
            "aligned_count": 0,
            "total_count": 0,
            "deviations": []
        }

    # Count alignments for each cell
    left_aligned_cells = []
    right_aligned_cells = []
    neither_aligned_cells = []

    # Track which baseline each cell aligns to (for finding most common)
    left_baseline_counts: Dict[float, int] = {}
    right_baseline_counts: Dict[float, int] = {}

    for cell in cluster_cells:
        bbox = cell.get("bbox", {})
        left_edge = bbox.get("x0", 0.0)
        right_edge = bbox.get("x1", 0.0)

        left_aligned = False
        right_aligned = False

        # Check left alignment
        if left_baselines:
            for baseline in left_baselines:
                if abs(left_edge - baseline) <= tolerance:
                    left_aligned = True
                    left_baseline_counts[baseline] = left_baseline_counts.get(baseline, 0) + 1
                    break

        # Check right alignment
        if right_baselines:
            for baseline in right_baselines:
                if abs(right_edge - baseline) <= tolerance:
                    right_aligned = True
                    right_baseline_counts[baseline] = right_baseline_counts.get(baseline, 0) + 1
                    break

        # Categorize cell
        if left_aligned:
            left_aligned_cells.append(cell)
        if right_aligned:
            right_aligned_cells.append(cell)
        if not left_aligned and not right_aligned:
            neither_aligned_cells.append(cell)

    total_count = len(cluster_cells)
    left_count = len(left_aligned_cells)
    right_count = len(right_aligned_cells)
    neither_count = len(neither_aligned_cells)

    left_pct = left_count / total_count if total_count > 0 else 0.0
    right_pct = right_count / total_count if total_count > 0 else 0.0

    # Determine alignment type
    if right_pct >= 0.70:
        alignment_type = "right"
        # Find most common right baseline
        if right_baseline_counts:
            baseline_value = max(right_baseline_counts.items(), key=lambda x: x[1])[0]
        else:
            baseline_value = None
    elif left_pct >= 0.70:
        alignment_type = "left"
        # Find most common left baseline
        if left_baseline_counts:
            baseline_value = max(left_baseline_counts.items(), key=lambda x: x[1])[0]
        else:
            baseline_value = None
    elif left_pct < 0.30 and right_pct < 0.30:
        alignment_type = "center"
        baseline_value = None
    else:
        alignment_type = "mixed"
        # For mixed, choose the more dominant baseline
        if right_pct > left_pct and right_baseline_counts:
            baseline_value = max(right_baseline_counts.items(), key=lambda x: x[1])[0]
        elif left_baseline_counts:
            baseline_value = max(left_baseline_counts.items(), key=lambda x: x[1])[0]
        else:
            baseline_value = None

    # Calculate confidence and detect deviations
    if baseline_value is not None:
        confidence = calculate_alignment_confidence(
            cluster_cells, alignment_type, baseline_value, tolerance
        )
        deviations = detect_baseline_deviations(
            cluster_cells, alignment_type, baseline_value, tolerance
        )
        aligned_count = int(confidence * total_count)
    else:
        confidence = 0.0
        deviations = []
        aligned_count = 0

    return {
        "alignment_type": alignment_type,
        "confidence": confidence,
        "baseline_value": baseline_value,
        "aligned_count": aligned_count,
        "total_count": total_count,
        "deviations": deviations,
    }


def detect_column_alignment_anomalies(
    blocks: List[Dict[str, Any]],
    severity_thresholds: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Detect alignment anomalies across text blocks.

    Analyzes blocks for alignment deviations and flags anomalies with severity ratings.
    An anomaly is flagged when:
    - Block has strong alignment (confidence ≥ 70%)
    - One or more cells deviate from the baseline
    - Deviation magnitude exceeds threshold (> 5.0pt)

    Severity Rating:
    - HIGH: Large deviation (>10pt) OR multiple deviations (≥2) with high confidence (≥0.9)
    - MEDIUM: Moderate deviation (5-10pt) OR single deviation with medium confidence (0.7-0.9)
    - LOW: Small deviation (<5pt detected during analysis)

    Parameters
    ----------
    blocks : List[Dict]
        List of text blocks with alignment_metadata
    severity_thresholds : Optional[Dict]
        Custom thresholds for severity rating (default: None uses built-in thresholds)

    Returns
    -------
    List[Dict]
        List of anomaly dictionaries with:
        - block_id: ID of block with anomaly
        - anomaly_type: "column_alignment_deviation"
        - severity: "HIGH" | "MEDIUM" | "LOW"
        - alignment_type: Block alignment type
        - baseline_value: Expected baseline
        - confidence: Alignment confidence score
        - deviation_count: Number of deviating cells
        - deviations: List of deviation details
        - description: Human-readable description
    """
    # Default severity thresholds
    if severity_thresholds is None:
        severity_thresholds = {
            "high_deviation_magnitude": 10.0,  # pt
            "medium_deviation_magnitude": 5.0,  # pt
            "high_confidence": 0.9,
            "medium_confidence": 0.7,
            "multiple_deviations": 2,
        }

    anomalies = []

    for block in blocks:
        alignment_metadata = block.get("alignment_metadata", {})

        # Skip blocks without alignment metadata
        if not alignment_metadata:
            continue

        alignment_type = alignment_metadata.get("alignment_type", "unknown")
        confidence = alignment_metadata.get("confidence", 0.0)
        baseline_value = alignment_metadata.get("baseline_value")
        deviations = alignment_metadata.get("deviations", [])
        deviation_count = len(deviations)

        # Skip if no deviations or weak alignment
        if deviation_count == 0:
            continue

        # Skip center/mixed alignments (deviations are expected)
        if alignment_type in ("center", "mixed"):
            continue

        # Only flag anomalies for strong alignments (confidence ≥ 70%)
        if confidence < severity_thresholds["medium_confidence"]:
            continue

        # Analyze deviation magnitudes
        max_deviation = max(abs(d["deviation"]) for d in deviations) if deviations else 0.0
        avg_deviation = sum(abs(d["deviation"]) for d in deviations) / len(deviations) if deviations else 0.0

        # Determine severity
        if (max_deviation > severity_thresholds["high_deviation_magnitude"] or
            (deviation_count >= severity_thresholds["multiple_deviations"] and
             confidence >= severity_thresholds["high_confidence"])):
            severity = "HIGH"
        elif max_deviation > severity_thresholds["medium_deviation_magnitude"]:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Build anomaly description
        if deviation_count == 1:
            deviation_desc = f"1 cell deviates by {max_deviation:+.1f}pt"
        else:
            deviation_desc = f"{deviation_count} cells deviate (max: {max_deviation:+.1f}pt, avg: {avg_deviation:.1f}pt)"

        description = (
            f"{alignment_type.capitalize()}-aligned block has alignment anomaly: "
            f"{deviation_desc} from baseline at {baseline_value:.1f}pt "
            f"(confidence: {confidence:.1%})"
        )

        # Create anomaly record
        anomaly = {
            "block_id": block.get("block_id"),
            "anomaly_type": "column_alignment_deviation",
            "severity": severity,
            "alignment_type": alignment_type,
            "baseline_value": baseline_value,
            "confidence": confidence,
            "deviation_count": deviation_count,
            "max_deviation": round(max_deviation, 2),
            "avg_deviation": round(avg_deviation, 2),
            "deviations": deviations,
            "description": description,
            "block_bbox": block.get("bbox"),
            "block_texts": block.get("texts", [])[:5],  # First 5 texts for context
        }

        anomalies.append(anomaly)

    return anomalies


# =============================================================================
# PHASE 3 (UC-005): BLOCK REFINEMENT AND CONFLICT RESOLUTION
# =============================================================================


def calculate_bbox_overlap(
    bbox1: Dict[str, float],
    bbox2: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate overlap metrics between two bounding boxes.

    Parameters
    ----------
    bbox1, bbox2 : Dict[str, float]
        Bounding boxes with keys: x0, y0, x1, y1

    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - intersection_area: Area of overlap
        - union_area: Total area covered by both boxes
        - iou: Intersection over Union (0.0 to 1.0)
        - overlap_ratio_1: Intersection / area(bbox1)
        - overlap_ratio_2: Intersection / area(bbox2)
    """
    # Calculate intersection rectangle
    x0_inter = max(bbox1["x0"], bbox2["x0"])
    y0_inter = max(bbox1["y0"], bbox2["y0"])
    x1_inter = min(bbox1["x1"], bbox2["x1"])
    y1_inter = min(bbox1["y1"], bbox2["y1"])

    # Check if there's any overlap
    if x0_inter >= x1_inter or y0_inter >= y1_inter:
        return {
            "intersection_area": 0.0,
            "union_area": 0.0,
            "iou": 0.0,
            "overlap_ratio_1": 0.0,
            "overlap_ratio_2": 0.0,
        }

    # Calculate areas
    intersection_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)

    area1 = (bbox1["x1"] - bbox1["x0"]) * (bbox1["y1"] - bbox1["y0"])
    area2 = (bbox2["x1"] - bbox2["x0"]) * (bbox2["y1"] - bbox2["y0"])

    union_area = area1 + area2 - intersection_area

    # Calculate metrics
    iou = intersection_area / union_area if union_area > 0 else 0.0
    overlap_ratio_1 = intersection_area / area1 if area1 > 0 else 0.0
    overlap_ratio_2 = intersection_area / area2 if area2 > 0 else 0.0

    return {
        "intersection_area": round(intersection_area, 2),
        "union_area": round(union_area, 2),
        "iou": round(iou, 3),
        "overlap_ratio_1": round(overlap_ratio_1, 3),
        "overlap_ratio_2": round(overlap_ratio_2, 3),
    }


def calculate_block_confidence(
    block: Dict[str, Any],
    tables: Optional[List[Dict[str, Any]]] = None,
    page_height: Optional[float] = None
) -> float:
    """
    Calculate confidence score for a text block (UC-005 / Phase 3).

    Combines multiple evidence metrics to assess block quality:
    - Cell count (25%): More cells = higher confidence
    - Alignment coherence (25%): Percentage of cells matching dominant alignment
    - Cell density (15%): Cells per area
    - Vertical span (15%): Cells per vertical distance
    - Table membership (20% bonus): Inside Docling table

    Parameters
    ----------
    block : Dict[str, Any]
        Text block with bbox, cells, alignment_metadata
    tables : List[Dict[str, Any]], optional
        List of Docling table bboxes for table membership check
    page_height : float, optional
        Page height for normalization

    Returns
    -------
    float
        Confidence score from 0.0 to 1.0

    Algorithm
    ---------
    1. Cell count score (25%):
       - Normalized by 20 cells (20+ cells = full score)
       - Score = min(cell_count / 20, 1.0) * 0.25

    2. Alignment coherence (25%):
       - From alignment_metadata.confidence
       - Score = alignment_confidence * 0.25

    3. Cell density (15%):
       - cells / area (pt^2)
       - Typical: 0.0005 cells/pt^2
       - Score = min(density / 0.0005, 1.0) * 0.15

    4. Vertical span (15%):
       - cells / vertical_distance (pt)
       - Typical: 0.1 cells/pt
       - Score = min(vertical_span / 0.1, 1.0) * 0.15

    5. Table membership bonus (20%):
       - If IoU >= 0.7 with any Docling table: +0.20
       - Otherwise: +0.0

    Example
    -------
    Strong block (in table, 30 cells, 95% alignment):
    - Cell count: min(30/20, 1.0) * 0.25 = 0.25
    - Alignment: 0.95 * 0.25 = 0.24
    - Density: assume full = 0.15
    - Vertical span: assume full = 0.15
    - Table bonus: 0.20
    - Total: 0.99

    Weak block (isolated, 3 cells, 50% alignment):
    - Cell count: min(3/20, 1.0) * 0.25 = 0.04
    - Alignment: 0.50 * 0.25 = 0.13
    - Density: assume 0.5x typical = 0.08
    - Vertical span: assume 0.5x typical = 0.08
    - Table bonus: 0.0
    - Total: 0.33
    """
    # Extract block properties
    bbox = block.get("bbox", {})
    cells = block.get("cells", [])
    alignment_metadata = block.get("alignment_metadata", {})

    # Metric 1: Cell count (25%)
    cell_count = len(cells)
    cell_count_score = min(cell_count / 20.0, 1.0) * 0.25

    # Metric 2: Alignment coherence (25%)
    alignment_confidence = alignment_metadata.get("confidence", 0.0)
    alignment_score = alignment_confidence * 0.25

    # Metric 3: Cell density (15%)
    # cells / area (pt^2), normalized by typical value 0.0005
    width = bbox.get("x1", 0.0) - bbox.get("x0", 0.0)
    height = bbox.get("y1", 0.0) - bbox.get("y0", 0.0)
    area = width * height

    if area > 0:
        cell_density = cell_count / area
        cell_density_score = min(cell_density / 0.0005, 1.0) * 0.15
    else:
        cell_density_score = 0.0

    # Metric 4: Vertical span (15%)
    # cells / vertical_distance (pt), normalized by typical value 0.1
    if height > 0:
        vertical_span = cell_count / height
        vertical_span_score = min(vertical_span / 0.1, 1.0) * 0.15
    else:
        vertical_span_score = 0.0

    # Metric 5: Table membership bonus (20%)
    table_bonus = 0.0
    if tables:
        for table in tables:
            table_bbox = table.get("bbox", {})
            if not table_bbox:
                continue

            overlap_metrics = calculate_bbox_overlap(bbox, table_bbox)
            iou = overlap_metrics.get("iou", 0.0)

            if iou >= 0.7:
                table_bonus = 0.20
                break

    # Combine all scores
    total_confidence = (
        cell_count_score +
        alignment_score +
        cell_density_score +
        vertical_span_score +
        table_bonus
    )

    return round(total_confidence, 3)


def merge_adjacent_blocks(
    blocks: List[Dict[str, Any]],
    tables: Optional[List[Dict[str, Any]]] = None,
    vertical_gap_threshold: float = 15.0,
    horizontal_center_tolerance: float = 20.0
) -> List[Dict[str, Any]]:
    """
    Merge adjacent blocks with same alignment (UC-005 / Phase 3 Increment 2).

    Merges blocks that meet ALL criteria:
    1. Same alignment type (both left OR both right)
    2. Adjacent vertically (gap < vertical_gap_threshold)
    3. Same table membership (both in same table OR both not in any table)
    4. Centers aligned horizontally (within horizontal_center_tolerance)

    Parameters
    ----------
    blocks : List[Dict[str, Any]]
        List of text blocks with bbox, alignment_metadata, cells
    tables : List[Dict[str, Any]], optional
        List of Docling table bboxes for table membership checking
    vertical_gap_threshold : float
        Maximum vertical gap to consider blocks adjacent (default: 15.0pt)
    horizontal_center_tolerance : float
        Maximum horizontal center offset to consider blocks aligned (default: 20.0pt)

    Returns
    -------
    List[Dict[str, Any]]
        Merged list of blocks (some may be merged)

    Algorithm
    ---------
    1. Sort blocks by vertical position (top to bottom)
    2. For each block, check if it can merge with the next block:
       - Same alignment type (left/right)
       - Vertical gap < threshold
       - Same table membership
       - Horizontal centers aligned
    3. If mergeable, combine cells and recalculate bbox
    4. Continue until no more merges possible

    Example Use Case
    ----------------
    Multi-line address in PTC-006:
    - Block 1: "123 Main Street" (left-aligned, y=200-215)
    - Block 2: "Apartment 4B" (left-aligned, y=220-235)
    - Block 3: "Springfield" (left-aligned, y=240-255)

    Gap between blocks: 5pt < 15pt threshold
    Same left alignment, same table
    => Merge into single address block
    """
    if not blocks:
        return []

    # Helper: Determine which table a block belongs to
    def get_block_table_id(block_bbox: Dict[str, float]) -> Optional[int]:
        """Return table index if block overlaps table (IoU >= 0.5), else None."""
        if not tables:
            return None

        for table_idx, table in enumerate(tables):
            table_bbox = table.get("bbox", {})
            if not table_bbox:
                continue

            overlap = calculate_bbox_overlap(block_bbox, table_bbox)
            if overlap["iou"] >= 0.5:
                return table_idx

        return None

    # Helper: Calculate vertical gap between two blocks
    def vertical_gap(block1: Dict[str, Any], block2: Dict[str, Any]) -> float:
        """Calculate vertical gap between bottom of block1 and top of block2."""
        bbox1 = block1.get("bbox", {})
        bbox2 = block2.get("bbox", {})

        bottom1 = bbox1.get("y1", 0.0)
        top2 = bbox2.get("y0", 0.0)

        return top2 - bottom1

    # Helper: Calculate horizontal center offset
    def horizontal_center_offset(block1: Dict[str, Any], block2: Dict[str, Any]) -> float:
        """Calculate horizontal offset between block centers."""
        bbox1 = block1.get("bbox", {})
        bbox2 = block2.get("bbox", {})

        center1 = (bbox1.get("x0", 0.0) + bbox1.get("x1", 0.0)) / 2
        center2 = (bbox2.get("x0", 0.0) + bbox2.get("x1", 0.0)) / 2

        return abs(center2 - center1)

    # Helper: Check if two blocks can be merged
    def can_merge(block1: Dict[str, Any], block2: Dict[str, Any]) -> bool:
        """Check if block1 and block2 meet all merge criteria."""
        # Get alignment types
        alignment1 = block1.get("alignment_metadata", {}).get("alignment_type", "unknown")
        alignment2 = block2.get("alignment_metadata", {}).get("alignment_type", "unknown")

        # Criterion 1: Same alignment type (left or right only)
        if alignment1 not in ("left", "right") or alignment2 not in ("left", "right"):
            return False
        if alignment1 != alignment2:
            return False

        # Criterion 2: Adjacent vertically
        gap = vertical_gap(block1, block2)
        if gap < 0 or gap > vertical_gap_threshold:
            return False

        # Criterion 3: Same table membership
        table_id1 = get_block_table_id(block1.get("bbox", {}))
        table_id2 = get_block_table_id(block2.get("bbox", {}))
        if table_id1 != table_id2:
            return False

        # Criterion 4: Horizontal centers aligned
        center_offset = horizontal_center_offset(block1, block2)
        if center_offset > horizontal_center_tolerance:
            return False

        return True

    # Helper: Merge two blocks
    def merge_two_blocks(block1: Dict[str, Any], block2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge block2 into block1, returning new merged block."""
        # Combine cells
        cells1 = block1.get("cells", [])
        cells2 = block2.get("cells", [])
        merged_cells = cells1 + cells2

        # Recalculate bbox (union of both blocks)
        bbox1 = block1.get("bbox", {})
        bbox2 = block2.get("bbox", {})

        merged_bbox = {
            "x0": min(bbox1.get("x0", 0.0), bbox2.get("x0", 0.0)),
            "y0": min(bbox1.get("y0", 0.0), bbox2.get("y0", 0.0)),
            "x1": max(bbox1.get("x1", 0.0), bbox2.get("x1", 0.0)),
            "y1": max(bbox1.get("y1", 0.0), bbox2.get("y1", 0.0)),
        }

        # Combine texts
        texts1 = block1.get("texts", [])
        texts2 = block2.get("texts", [])
        merged_texts = texts1 + texts2

        # Keep alignment metadata from block1 (dominant block)
        alignment_metadata = block1.get("alignment_metadata", {})

        # Create merged block
        merged_block = {
            "bbox": merged_bbox,
            "cells": merged_cells,
            "texts": merged_texts,
            "alignment_metadata": alignment_metadata,
            "dominant_alignment": block1.get("dominant_alignment"),
            "merged_from": [
                block1.get("block_id", "unknown"),
                block2.get("block_id", "unknown")
            ]
        }

        return merged_block

    # Sort blocks by vertical position (top to bottom)
    sorted_blocks = sorted(blocks, key=lambda b: b.get("bbox", {}).get("y0", 0.0))

    # Iteratively merge adjacent blocks until no more merges possible
    merged = True
    while merged:
        merged = False
        new_blocks = []
        i = 0

        while i < len(sorted_blocks):
            current_block = sorted_blocks[i]

            # Check if current block can merge with next block
            if i + 1 < len(sorted_blocks):
                next_block = sorted_blocks[i + 1]

                if can_merge(current_block, next_block):
                    # Merge current and next
                    merged_block = merge_two_blocks(current_block, next_block)
                    new_blocks.append(merged_block)
                    i += 2  # Skip both blocks
                    merged = True
                    continue

            # No merge, keep current block
            new_blocks.append(current_block)
            i += 1

        sorted_blocks = new_blocks

    return sorted_blocks


def detect_nested_blocks(
    blocks: List[Dict[str, Any]],
    tables: Optional[List[Dict[str, Any]]] = None,
    iou_threshold: float = 0.8,
    area_ratio_threshold: float = 0.5,
    confidence_difference_threshold: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Remove nested/engulfed blocks based on IoU and confidence (UC-005 / Phase 3 Increment 3).

    Identifies blocks that are nested inside other blocks and removes the weaker one
    based on confidence scores.

    Parameters
    ----------
    blocks : List[Dict[str, Any]]
        List of text blocks with bbox, confidence, cells
    tables : List[Dict[str, Any]], optional
        List of Docling table bboxes (for confidence calculation if needed)
    iou_threshold : float
        IoU threshold for considering blocks nested (default: 0.8)
    area_ratio_threshold : float
        Maximum area ratio (small/large) for nesting (default: 0.5)
    confidence_difference_threshold : float
        Minimum confidence difference to remove nested block (default: 0.20)

    Returns
    -------
    List[Dict[str, Any]]
        Filtered list with nested blocks removed

    Algorithm
    ---------
    1. Calculate IoU for all block pairs
    2. If IoU > threshold AND area_small < 0.5 × area_large:
       - Consider smaller block nested in larger
       - Calculate confidence for both blocks (if not already present)
       - If confidence_large - confidence_small >= 0.20:
         - Remove smaller block
       - Else:
         - Keep both (ambiguous case)

    Example
    -------
    Large table block: 400x300pt, 50 cells, confidence = 0.85
    Small nested block: 100x50pt, 3 cells, confidence = 0.35
    IoU = 1.0 (small fully inside large)
    Area ratio = 0.04 < 0.5
    Confidence diff = 0.50 > 0.20
    => Remove small block
    """
    if not blocks:
        return []

    # Calculate confidence for all blocks if not present
    for block in blocks:
        if "confidence" not in block:
            block["confidence"] = calculate_block_confidence(block, tables, page_height=None)

    # Helper: Calculate area of bbox
    def bbox_area(bbox: Dict[str, float]) -> float:
        """Calculate area of bounding box."""
        return (bbox.get("x1", 0.0) - bbox.get("x0", 0.0)) * \
               (bbox.get("y1", 0.0) - bbox.get("y0", 0.0))

    # Track blocks to remove
    blocks_to_remove = set()

    # Check all pairs of blocks for nesting
    for i, block1 in enumerate(blocks):
        if i in blocks_to_remove:
            continue

        bbox1 = block1.get("bbox", {})
        area1 = bbox_area(bbox1)
        conf1 = block1.get("confidence", 0.0)

        for j, block2 in enumerate(blocks):
            if i == j or j in blocks_to_remove:
                continue

            bbox2 = block2.get("bbox", {})
            area2 = bbox_area(bbox2)
            conf2 = block2.get("confidence", 0.0)

            # Calculate overlap
            overlap = calculate_bbox_overlap(bbox1, bbox2)
            iou = overlap.get("iou", 0.0)
            overlap_ratio_1 = overlap.get("overlap_ratio_1", 0.0)
            overlap_ratio_2 = overlap.get("overlap_ratio_2", 0.0)

            # Check if blocks are nested (high overlap ratio for smaller block)
            # Use overlap ratio instead of IoU for nesting detection
            max_overlap_ratio = max(overlap_ratio_1, overlap_ratio_2)
            if max_overlap_ratio < iou_threshold:
                continue

            # Determine which is smaller
            if area1 < area2:
                smaller_idx = i
                larger_idx = j
                smaller_area = area1
                larger_area = area2
                smaller_conf = conf1
                larger_conf = conf2
            else:
                smaller_idx = j
                larger_idx = i
                smaller_area = area2
                larger_area = area1
                smaller_conf = conf2
                larger_conf = conf1

            # Check area ratio
            area_ratio = smaller_area / larger_area if larger_area > 0 else 0.0

            if area_ratio >= area_ratio_threshold:
                # Not nested - blocks are similar size
                continue

            # Check confidence difference
            confidence_diff = larger_conf - smaller_conf

            if confidence_diff >= confidence_difference_threshold:
                # Remove smaller block (lower confidence)
                blocks_to_remove.add(smaller_idx)
            # else: Keep both (confidence difference too small, ambiguous)

    # Filter out removed blocks
    filtered_blocks = [block for i, block in enumerate(blocks) if i not in blocks_to_remove]

    return filtered_blocks


def filter_weak_evidence_blocks(
    blocks: List[Dict[str, Any]],
    evidence_strength_threshold: float = 0.05,
    overlap_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Filter blocks with weak evidence (UC-005 / Phase 3 Increment 4).

    Removes blocks with low evidence strength (few cells over large vertical span)
    that overlap with stronger blocks.

    Parameters
    ----------
    blocks : List[Dict[str, Any]]
        List of text blocks with bbox, cells
    evidence_strength_threshold : float
        Minimum evidence strength (cells/vertical_span) to keep (default: 0.05)
    overlap_threshold : float
        IoU threshold for considering overlap with stronger blocks (default: 0.3)

    Returns
    -------
    List[Dict[str, Any]]
        Filtered list with weak evidence blocks removed

    Algorithm
    ---------
    1. Calculate evidence strength for each block:
       evidence_strength = cell_count / vertical_span
    2. Identify weak blocks (strength < threshold)
    3. For each weak block, check IoU with all other blocks
    4. If weak block overlaps (IoU > 0.3) with a stronger block:
       - Remove weak block
    5. Keep weak blocks that don't overlap with stronger blocks

    Evidence Strength Examples
    ---------------------------
    - Strong block: 20 cells in 100pt span = 0.20 cells/pt ✓
    - Medium block: 10 cells in 100pt span = 0.10 cells/pt ✓
    - Weak block: 5 cells in 100pt span = 0.05 cells/pt (borderline)
    - Very weak block: 3 cells in 200pt span = 0.015 cells/pt (likely noise)

    Use Case
    --------
    Long-distance spurious blocks that span large vertical distances but
    contain very few cells. These are often artifacts of over-aggressive
    clustering. Filter them if they overlap with stronger blocks.
    """
    if not blocks:
        return []

    # Helper: Calculate vertical span
    def vertical_span(bbox: Dict[str, float]) -> float:
        """Calculate vertical span of bbox."""
        return bbox.get("y1", 0.0) - bbox.get("y0", 0.0)

    # Helper: Calculate evidence strength
    def evidence_strength(block: Dict[str, Any]) -> float:
        """Calculate evidence strength (cells / vertical_span)."""
        bbox = block.get("bbox", {})
        cells = block.get("cells", [])
        v_span = vertical_span(bbox)

        if v_span <= 0:
            return 0.0

        return len(cells) / v_span

    # Calculate evidence strength for all blocks
    for block in blocks:
        if "evidence_strength" not in block:
            block["evidence_strength"] = evidence_strength(block)

    # Identify weak and strong blocks
    weak_blocks_indices = []
    strong_blocks_indices = []

    for i, block in enumerate(blocks):
        strength = block.get("evidence_strength", 0.0)
        if strength < evidence_strength_threshold:
            weak_blocks_indices.append(i)
        else:
            strong_blocks_indices.append(i)

    # Track blocks to remove
    blocks_to_remove = set()

    # Check each weak block for overlap with stronger blocks
    for weak_idx in weak_blocks_indices:
        weak_block = blocks[weak_idx]
        weak_bbox = weak_block.get("bbox", {})

        for strong_idx in strong_blocks_indices:
            strong_block = blocks[strong_idx]
            strong_bbox = strong_block.get("bbox", {})

            # Calculate overlap
            overlap = calculate_bbox_overlap(weak_bbox, strong_bbox)
            iou = overlap.get("iou", 0.0)

            if iou > overlap_threshold:
                # Weak block overlaps with stronger block - remove it
                blocks_to_remove.add(weak_idx)
                break

    # Filter out removed blocks
    filtered_blocks = [block for i, block in enumerate(blocks) if i not in blocks_to_remove]

    return filtered_blocks


def refine_text_blocks(
    blocks: List[Dict[str, Any]],
    tables: Optional[List[Dict[str, Any]]] = None,
    page_height: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply all block refinement and conflict resolution steps (UC-005 / Phase 3 Integration).

    Integrates all refinement functions in optimal order:
    1. Calculate confidence scores for all blocks
    2. Merge adjacent blocks with same alignment
    3. Detect and remove nested/engulfed blocks
    4. Filter weak evidence blocks
    5. Return refined blocks with statistics

    Parameters
    ----------
    blocks : List[Dict[str, Any]]
        Raw text blocks from initial detection
    tables : List[Dict[str, Any]], optional
        Docling table bboxes for table-aware processing
    page_height : float, optional
        Page height for normalization
    config : Dict[str, Any], optional
        Configuration overrides for refinement thresholds

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'blocks' and 'stats' keys
    """
    # Initialize config with defaults
    default_config = {
        "vertical_gap_threshold": 15.0,
        "horizontal_center_tolerance": 20.0,
        "iou_threshold": 0.8,
        "area_ratio_threshold": 0.5,
        "confidence_difference_threshold": 0.20,
        "evidence_strength_threshold": 0.05,
        "weak_overlap_threshold": 0.3,
    }

    if config:
        default_config.update(config)

    config = default_config

    # Initialize statistics
    stats = {
        "initial_count": len(blocks),
        "after_confidence": 0,
        "after_merging": 0,
        "after_nesting": 0,
        "after_weak_filter": 0,
        "final_count": 0,
        "removed_count": 0,
        "merged_count": 0,
    }

    # Step 1: Calculate confidence for all blocks
    for block in blocks:
        if "confidence" not in block:
            block["confidence"] = calculate_block_confidence(block, tables, page_height)

    stats["after_confidence"] = len(blocks)

    # Step 2: Merge adjacent blocks
    blocks = merge_adjacent_blocks(
        blocks,
        tables=tables,
        vertical_gap_threshold=config["vertical_gap_threshold"],
        horizontal_center_tolerance=config["horizontal_center_tolerance"]
    )

    stats["after_merging"] = len(blocks)
    stats["merged_count"] = stats["initial_count"] - stats["after_merging"]

    # Step 3: Detect and remove nested blocks
    blocks = detect_nested_blocks(
        blocks,
        tables=tables,
        iou_threshold=config["iou_threshold"],
        area_ratio_threshold=config["area_ratio_threshold"],
        confidence_difference_threshold=config["confidence_difference_threshold"]
    )

    stats["after_nesting"] = len(blocks)

    # Step 4: Filter weak evidence blocks
    blocks = filter_weak_evidence_blocks(
        blocks,
        evidence_strength_threshold=config["evidence_strength_threshold"],
        overlap_threshold=config["weak_overlap_threshold"]
    )

    stats["after_weak_filter"] = len(blocks)
    stats["final_count"] = len(blocks)
    stats["removed_count"] = stats["initial_count"] - stats["final_count"]

    return {
        "blocks": blocks,
        "stats": stats
    }
