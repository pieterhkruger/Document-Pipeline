"""
Alignment Baseline Detection Module
====================================
Detects alignment baselines (dominant left/right/top/bottom edges) and horizontal
alignment anomalies in PDF documents for forgery detection.

This module is part of the document forgery detection system.
See ALIGNMENT_AND_COLON_LOGIC_SUMMARY.md for detailed documentation.
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from statistics import median

# =============================================================================
# CONSTANTS
# =============================================================================

# Configuration flags
USE_BASELINE_CONFIDENCE_FILTERING = True  # Enable UC-002 confidence-based baseline filtering

# Baseline detection constants
BLOCK_WEIGHT_CAP = 6                   # cap line_count influence when seeding baselines with blocks
RAW_BASELINE_EXTEND_TOLERANCE = 2.0    # max distance (pt) for raw cells to piggyback on block baselines
MIN_VERTICAL_CLUSTER_WEIGHT = 3.0      # minimum weighted support for vertical baselines
VERTICAL_BASELINE_MERGE_TOLERANCE = 3.0  # pt - merge vertical baselines closer than this if coverage overlaps
MIN_VERTICAL_OVERLAP_RATIO = 0.30        # require 30% shared coverage when merging
MIN_BASELINE_COVERAGE = 0.20             # drop block-seeded baselines covering <20% of page height (non-edge)


# =============================================================================
# HELPER FUNCTIONS (Dependencies)
# =============================================================================

def extract_bbox_coords(bbox: Dict) -> Tuple[float, float, float, float]:
    """
    Extract axis-aligned bounding box coordinates from various bbox formats.

    Supports:
    - Rotated rectangle format: {r_x0, r_y0, r_x1, r_y1, r_x2, r_y2, r_x3, r_y3}
    - Simple format: {x0, y0, x1, y1}

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    # Check if it's the rotated rectangle format (4 corners)
    if 'r_x0' in bbox:
        # Extract all 4 corners
        x_coords = [
            bbox.get('r_x0', 0),
            bbox.get('r_x1', 0),
            bbox.get('r_x2', 0),
            bbox.get('r_x3', 0)
        ]
        y_coords = [
            bbox.get('r_y0', 0),
            bbox.get('r_y1', 0),
            bbox.get('r_y2', 0),
            bbox.get('r_y3', 0)
        ]

        # Get axis-aligned bounding box (min/max of all corners)
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        return x_min, y_min, x_max, y_max

    # Check if it's the simple format (x0, y0, x1, y1)
    elif 'x0' in bbox:
        return bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']

    # Check if it's Docling table format (l, t, r, b)
    elif all(key in bbox for key in ('l', 't', 'r', 'b')):
        x0 = float(bbox['l'])
        y0 = float(min(bbox['t'], bbox['b']))
        x1 = float(bbox['r'])
        y1 = float(max(bbox['t'], bbox['b']))
        return x0, y0, x1, y1

    else:
        raise ValueError(f"Unsupported bbox format: {bbox}")


def calculate_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) for two bboxes.

    Bbox formats supported:
    - Rotated rectangle: {r_x0, r_y0, r_x1, r_y1, r_x2, r_y2, r_x3, r_y3, coord_origin}
    - Simple format: {x0, y0, x1, y1}
    """
    try:
        # Extract coordinates using format-aware helper
        x1_min, y1_min, x1_max, y1_max = extract_bbox_coords(bbox1)
        x2_min, y2_min, x2_max, y2_max = extract_bbox_coords(bbox2)

        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    except Exception:
        return 0.0


def _cluster_weight(cluster_cells: List[Dict[str, Any]]) -> float:
    return sum(cell.get("_weight", 1.0) for cell in cluster_cells)


def _collect_axis_centers(cluster_cells: List[Dict[str, Any]], axis: str) -> List[float]:
    values: List[float] = []
    for cell in cluster_cells:
        bbox = cell.get("bbox", {})
        if not bbox:
            continue
        if axis == "y":
            center = (bbox.get("y0", 0) + bbox.get("y1", 0)) / 2
        else:
            center = (bbox.get("x0", 0) + bbox.get("x1", 0)) / 2
        repeats = max(1, min(int(round(cell.get("_weight", 1.0))), BLOCK_WEIGHT_CAP))
        values.extend([center] * repeats)
    return values


def _is_within_block_support(edge: float, block_values: List[float]) -> bool:
    if not block_values:
        return False
    return min(abs(edge - value) for value in block_values) <= RAW_BASELINE_EXTEND_TOLERANCE


def _merge_spans(spans: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    normalized = [
        (min(start, end), max(start, end))
        for start, end in spans
        if start is not None and end is not None
    ]
    normalized = [span for span in normalized if span[1] > span[0]]
    if not normalized:
        return []
    normalized.sort(key=lambda v: v[0])
    merged: List[Tuple[float, float]] = []
    current_start, current_end = normalized[0]
    for start, end in normalized[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _span_total_length(spans: List[Tuple[float, float]]) -> float:
    merged = _merge_spans(spans)
    return sum(end - start for start, end in merged)


def _span_overlap_ratio(spans_a: List[Tuple[float, float]], spans_b: List[Tuple[float, float]]) -> float:
    if not spans_a or not spans_b:
        return 0.0
    merged_a = _merge_spans(spans_a)
    merged_b = _merge_spans(spans_b)
    i = j = 0
    overlap = 0.0
    while i < len(merged_a) and j < len(merged_b):
        start = max(merged_a[i][0], merged_b[j][0])
        end = min(merged_a[i][1], merged_b[j][1])
        if end > start:
            overlap += end - start
        if merged_a[i][1] < merged_b[j][1]:
            i += 1
        else:
            j += 1
    min_total = min(_span_total_length(spans_a), _span_total_length(spans_b))
    return overlap / min_total if min_total > 0 else 0.0


def _compute_span_coverage(spans: List[Tuple[float, float]], page_height: Optional[float]) -> float:
    if not page_height or page_height <= 0:
        return 0.0
    total = _span_total_length(spans)
    return min(1.0, max(0.0, total / page_height))


def _merge_baseline_dict(primary: Dict[str, Any], secondary: Dict[str, Any], page_height: Optional[float]) -> None:
    total_weight = primary.get('count', 0) + secondary.get('count', 0)
    if total_weight > 0:
        primary['value'] = (
            primary.get('value', 0) * primary.get('count', 0) +
            secondary.get('value', 0) * secondary.get('count', 0)
        ) / total_weight
    primary['count'] = total_weight
    primary.setdefault('y_values', [])
    primary['y_values'].extend(secondary.get('y_values', []))
    primary['block_support'] = primary.get('block_support', 0) + secondary.get('block_support', 0)
    primary['raw_support'] = primary.get('raw_support', 0) + secondary.get('raw_support', 0)
    primary.setdefault('_block_spans', [])
    primary['_block_spans'].extend(secondary.get('_block_spans', []))
    primary['coverage'] = round(_compute_span_coverage(primary['_block_spans'], page_height), 3) if primary['_block_spans'] else primary.get('coverage', 0.0)


def _merge_vertical_baselines_list(baselines: List[Dict[str, Any]], page_height: Optional[float]) -> List[Dict[str, Any]]:
    if not baselines:
        return []
    sorted_baselines = sorted(baselines, key=lambda b: b.get('value', 0.0))
    merged: List[Dict[str, Any]] = []
    for baseline in sorted_baselines:
        if not merged:
            merged.append(baseline)
            continue
        prev = merged[-1]
        if abs(prev.get('value', 0.0) - baseline.get('value', 0.0)) <= VERTICAL_BASELINE_MERGE_TOLERANCE:
            overlap_ratio = _span_overlap_ratio(prev.get('_block_spans', []), baseline.get('_block_spans', []))
            if overlap_ratio >= MIN_VERTICAL_OVERLAP_RATIO:
                _merge_baseline_dict(prev, baseline, page_height)
                continue
        merged.append(baseline)
    return merged


def build_alignment_metadata(docling_payload: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Build lookup of Docling structural metadata (table cells, text labels) by page.
    Used to suppress alignment checks for headers/section markers.
    """
    metadata_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    if not isinstance(docling_payload, dict):
        return metadata_by_page

    doc_struct = docling_payload.get("docling_document")
    if not isinstance(doc_struct, dict):
        return metadata_by_page

    tables = doc_struct.get("tables", [])
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, dict):
                continue

            prov_entries = table.get("prov", [])
            page_index: Optional[int] = None
            if isinstance(prov_entries, list):
                for prov in prov_entries:
                    if not isinstance(prov, dict):
                        continue
                    page_no = prov.get("page_no")
                    if page_no is None:
                        continue
                    try:
                        page_no = int(page_no)
                    except (TypeError, ValueError):
                        continue
                    page_index = page_no - 1 if page_no > 0 else page_no
                    break

            if page_index is None:
                continue

            data = table.get("data", {})
            cell_candidates: List[Dict[str, Any]] = []

            grid = data.get("grid")
            if isinstance(grid, list):
                for row in grid:
                    if isinstance(row, list):
                        cell_candidates.extend(c for c in row if isinstance(c, dict))
            elif isinstance(data.get("table_cells"), list):
                cell_candidates.extend(c for c in data["table_cells"] if isinstance(c, dict))

            for cell in cell_candidates:
                bbox = cell.get("bbox")
                if not isinstance(bbox, dict):
                    continue

                try:
                    x0, y0, x1, y1 = extract_bbox_coords(bbox)
                except Exception:
                    continue

                metadata_by_page[page_index].append({
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "column_header": bool(cell.get("column_header")),
                    "row_header": bool(cell.get("row_header")),
                    "row_section": bool(cell.get("row_section")),
                    "type": "table_cell"
                })

    texts = doc_struct.get("texts", [])
    if isinstance(texts, list):
        for text_item in texts:
            if not isinstance(text_item, dict):
                continue

            prov_entries = text_item.get("prov", [])
            if not isinstance(prov_entries, list):
                continue

            bbox = None
            page_index: Optional[int] = None
            for prov in prov_entries:
                if not isinstance(prov, dict):
                    continue
                page_no = prov.get("page_no")
                bbox_dict = prov.get("bbox")
                if bbox_dict is None:
                    continue
                try:
                    x0, y0, x1, y1 = extract_bbox_coords(bbox_dict)
                except Exception:
                    continue
                bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
                if page_no is not None:
                    try:
                        page_no = int(page_no)
                    except (TypeError, ValueError):
                        page_no = None
                if page_no is not None:
                    page_index = page_no - 1 if page_no > 0 else page_no
                break

            if bbox is None or page_index is None:
                continue

            metadata_by_page[page_index].append({
                "bbox": bbox,
                "label": text_item.get("label"),
                "type": "text"
            })

    return metadata_by_page


def match_alignment_metadata(
    bbox: Dict[str, float],
    metadata_items: List[Dict[str, Any]],
    min_iou: float = 0.6
) -> Optional[Dict[str, Any]]:
    """
    Find the best Docling metadata entry overlapping the provided bbox.
    """
    best_item: Optional[Dict[str, Any]] = None
    best_iou: float = 0.0

    for item in metadata_items:
        item_bbox = item.get("bbox")
        if not item_bbox:
            continue
        try:
            overlap = calculate_iou(bbox, item_bbox)
        except Exception:
            continue
        if overlap > best_iou:
            best_iou = overlap
            best_item = item

    if best_iou >= min_iou:
        return best_item

    return None


def consolidate_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Consolidate cells by removing words that are contained within larger phrases/cells.

    Example: If "HOURS" is spatially contained in "HOURS /UNITS", keep only "HOURS /UNITS".

    Args:
        cells: List of cell dictionaries with 'text' and 'bbox' keys

    Returns:
        Filtered list of cells with contained words removed
    """
    if not cells:
        return []

    # Sort cells by area (descending) to process larger cells first
    sorted_cells = sorted(
        cells,
        key=lambda c: (c["bbox"]["x1"] - c["bbox"]["x0"]) * (c["bbox"]["y1"] - c["bbox"]["y0"]),
        reverse=True
    )

    consolidated = []
    removed_indices = set()

    for i, cell in enumerate(sorted_cells):
        if i in removed_indices:
            continue

        cell_bbox = cell["bbox"]
        cell_text = cell["text"].strip()

        # Check if this cell contains any smaller cells
        for j, other_cell in enumerate(sorted_cells[i+1:], start=i+1):
            if j in removed_indices:
                continue

            other_bbox = other_cell["bbox"]
            other_text = other_cell["text"].strip()

            # Check if other_cell is spatially contained within cell
            # Allow small tolerance for rounding errors (0.5pt)
            tolerance = 0.5
            x_contained = (other_bbox["x0"] >= cell_bbox["x0"] - tolerance and
                          other_bbox["x1"] <= cell_bbox["x1"] + tolerance)
            y_contained = (other_bbox["y0"] >= cell_bbox["y0"] - tolerance and
                          other_bbox["y1"] <= cell_bbox["y1"] + tolerance)

            # Check if the text of the smaller cell appears in the larger cell
            text_contained = other_text.lower() in cell_text.lower()

            if x_contained and y_contained and text_contained:
                # Mark the smaller cell for removal
                removed_indices.add(j)

        consolidated.append(cell)

    return consolidated


# =============================================================================
# BASELINE DETECTION FUNCTIONS
# =============================================================================

def detect_alignment_baselines_with_debug(
    cells: List[Dict[str, Any]],
    page_width: float,
    page_metadata: Optional[List[Dict[str, Any]]] = None,
    page_height: Optional[float] = None,
    blocks: Optional[List[Dict[str, Any]]] = None
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Detect alignment baselines with debugging information.

    Returns:
        Tuple of (baselines, debug_info)
    """
    debug_info = {
        'total_cells': len(cells),
        'filtered_cells': 0,
        'left_clusters': {},
        'right_clusters': {},
        'top_clusters': {},
        'bottom_clusters': {},
        'baselines_found': 0,
        'baselines_rejected': [],
        'sample_cells': [],  # Sample of cells with their coordinates
        'block_count': len(blocks or [])
    }

    baselines = detect_alignment_baselines(cells, page_width, page_metadata, page_height, blocks=blocks)

    # Re-run clustering to get debug info
    if cells:
        filtered_cells = []
        for cell in cells:
            if page_metadata:
                match = match_alignment_metadata(cell["bbox"], page_metadata, min_iou=0.6)
                if match:
                    if match.get("column_header") or match.get("row_header"):
                        continue
                    if match.get("label") == "section_header":
                        continue
            filtered_cells.append(cell)

        debug_info['filtered_cells'] = len(filtered_cells)

        # Store sample cells for debugging (first 30)
        for cell in filtered_cells[:30]:
            bbox = cell.get('bbox', {})
            debug_info['sample_cells'].append({
                'text': cell.get('text', '')[:50],  # Truncate long text
                'x0': round(bbox.get('x0', 0), 1),
                'y0': round(bbox.get('y0', 0), 1),
                'x1': round(bbox.get('x1', 0), 1),
                'y1': round(bbox.get('y1', 0), 1)
            })

        if filtered_cells:
            # Cluster edges (using same 1pt precision as main detection)
            left_clusters = defaultdict(list)
            right_clusters = defaultdict(list)
            top_clusters = defaultdict(list)
            bottom_clusters = defaultdict(list)

            for cell in filtered_cells:
                bbox = cell["bbox"]
                left_clusters[round(bbox["x0"])].append(cell)  # Changed from 0.1pt to 1pt precision
                right_clusters[round(bbox["x1"])].append(cell)
                top_clusters[round(bbox["y0"])].append(cell)
                bottom_clusters[round(bbox["y1"])].append(cell)

            # Store cluster counts
            debug_info['left_clusters'] = {k: len(v) for k, v in sorted(left_clusters.items(), key=lambda x: -len(x[1]))[:20]}
            debug_info['right_clusters'] = {k: len(v) for k, v in sorted(right_clusters.items(), key=lambda x: -len(x[1]))[:20]}
            debug_info['top_clusters'] = {k: len(v) for k, v in sorted(top_clusters.items(), key=lambda x: -len(x[1]))[:20]}
            debug_info['bottom_clusters'] = {k: len(v) for k, v in sorted(bottom_clusters.items(), key=lambda x: -len(x[1]))[:20]}

            # Track rejected clusters
            # For vertical baselines (left/right): need 3
            # For horizontal baselines (top/bottom): need 2
            for orientation, clusters in [
                ('left', left_clusters),
                ('right', right_clusters),
                ('top', top_clusters),
                ('bottom', bottom_clusters)
            ]:
                min_threshold = 2 if orientation in ['top', 'bottom'] else 3
                for key, cells_list in clusters.items():
                    count = len(cells_list)
                    if count < min_threshold:
                        debug_info['baselines_rejected'].append({
                            'orientation': orientation,
                            'value': key,
                            'count': count,
                            'reason': f'Below minimum threshold (need {min_threshold}, have {count})'
                        })

    debug_info['baselines_found'] = len(baselines)
    debug_info['vertical_baseline_stats'] = [
        {
            'value': round(b.get('value', 0.0), 2),
            'coverage': b.get('coverage', 0.0),
            'block_support': b.get('block_support', 0),
            'raw_support': b.get('raw_support', b.get('count', 0))
        }
        for b in baselines
        if b.get('orientation') in ('left', 'right')
    ]

    return baselines, debug_info


def detect_alignment_baselines(
    cells: List[Dict[str, Any]],
    page_width: float,
    page_metadata: Optional[List[Dict[str, Any]]] = None,
    page_height: Optional[float] = None,
    blocks: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Detect alignment baselines (dominant left/right/top/bottom edges) in a page.

    Returns list of baselines with:
    - orientation: "left", "right", "top", or "bottom"
    - value: x-coordinate (for left/right) or y-coordinate (for top/bottom) of the baseline
    - page_position: positional classification (e.g., "page_left", "page_top", etc.)
    - y_values: list of y-coordinates of blocks aligned to this baseline (for vertical baselines)
    - x_values: list of x-coordinates of blocks aligned to this baseline (for horizontal baselines)
    - count: number of items aligned to this baseline
    """
    if not cells:
        return []

    # Filter out structural elements using metadata
    filtered_cells = []
    for cell in cells:
        if page_metadata:
            match = match_alignment_metadata(cell["bbox"], page_metadata, min_iou=0.6)
            if match:
                # Skip table headers and section headers
                if match.get("column_header") or match.get("row_header"):
                    continue
                if match.get("label") == "section_header":
                    continue

        filtered_cells.append(cell)

    if not filtered_cells:
        return []

    # Cluster edges with 1pt precision for more forgiving alignment detection
    # This helps when values have slight variations (e.g., 220.0, 220.3, 220.7 all → 220)
    left_clusters: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    right_clusters: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    top_clusters: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    bottom_clusters: Dict[float, List[Dict[str, Any]]] = defaultdict(list)

    block_left_values: List[float] = []
    block_right_values: List[float] = []
    if blocks:
        for block in blocks:
            bbox = (block or {}).get("bbox")
            if not bbox:
                continue
            weight = max(1.0, min(float(block.get("line_count", 1)), float(BLOCK_WEIGHT_CAP)))
            base_entry = {
                "bbox": bbox,
                "_weight": weight,
                "_source": "block",
                "_block_id": block.get("block_id"),
            }
            left_edge = bbox.get("x0", 0.0)
            right_edge = bbox.get("x1", 0.0)
            block_left_values.append(left_edge)
            block_right_values.append(right_edge)
            left_clusters[round(left_edge)].append(dict(base_entry))
            right_clusters[round(right_edge)].append(dict(base_entry))

    for cell in filtered_cells:
        bbox = cell["bbox"]
        left_edge = bbox["x0"]
        right_edge = bbox["x1"]
        top_edge = bbox["y0"]
        bottom_edge = bbox["y1"]

        # Round to nearest 1pt for clustering (increased from 0.1pt for better tolerance)
        # This groups items within 0.5pt of each other into the same cluster
        left_key = round(left_edge)
        right_key = round(right_edge)
        top_key = round(top_edge)
        bottom_key = round(bottom_edge)

        allow_left = not blocks or not block_left_values or _is_within_block_support(left_edge, block_left_values)
        allow_right = not blocks or not block_right_values or _is_within_block_support(right_edge, block_right_values)

        left_entry = {"bbox": bbox, "_weight": 1.0, "_source": "cell"}
        right_entry = {"bbox": bbox, "_weight": 1.0, "_source": "cell"}

        if allow_left:
            left_clusters[left_key].append(left_entry)
        if allow_right:
            right_clusters[right_key].append(right_entry)
        top_clusters[top_key].append(cell)
        bottom_clusters[bottom_key].append(cell)

    baselines = []

    # Process left-aligned baselines (vertical)
    for cluster_key, cluster_cells in left_clusters.items():
        cluster_weight = _cluster_weight(cluster_cells)
        if cluster_weight < MIN_VERTICAL_CLUSTER_WEIGHT:
            continue
        if blocks and block_left_values:
            has_block_support = any(cell.get("_source") == "block" for cell in cluster_cells)
            if not has_block_support:
                continue

        # Calculate mean x-coordinate
        weighted_sum = sum(c["bbox"]["x0"] * c.get("_weight", 1.0) for c in cluster_cells)
        mean_x = weighted_sum / max(cluster_weight, 1e-6)

        # Collect y-values (use center y for each block)
        y_values = _collect_axis_centers(cluster_cells, axis="y")
        block_spans: List[Tuple[float, float]] = []
        span_candidates: List[Tuple[float, float]] = []
        block_support = 0.0
        for entry in cluster_cells:
            bbox_entry = entry.get("bbox", {})
            span = (bbox_entry.get("y0", 0.0), bbox_entry.get("y1", 0.0))
            span_candidates.append(span)
            if entry.get("_source") == "block":
                block_support += entry.get("_weight", 1.0)
                block_spans.append(span)
        coverage_spans = block_spans if block_spans else span_candidates
        coverage = _compute_span_coverage(coverage_spans, page_height) if coverage_spans else 0.0

        # Determine page position
        if mean_x < page_width * 0.25:
            page_position = "page_left"
        elif mean_x > page_width * 0.75:
            page_position = "page_right"
        else:
            page_position = "mid_page"

        baselines.append({
            "orientation": "left",
            "value": mean_x,
            "page_position": page_position,
            "y_values": y_values,
            "count": int(round(cluster_weight)),
            "block_support": int(round(block_support)),
            "raw_support": max(0, int(round(cluster_weight - block_support))),
            "coverage": round(coverage, 3) if coverage else 0.0,
            "_block_spans": coverage_spans
        })

    # Process right-aligned baselines (vertical)
    for cluster_key, cluster_cells in right_clusters.items():
        cluster_weight = _cluster_weight(cluster_cells)
        if cluster_weight < MIN_VERTICAL_CLUSTER_WEIGHT:
            continue
        if blocks and block_right_values:
            has_block_support = any(cell.get("_source") == "block" for cell in cluster_cells)
            if not has_block_support:
                continue

        # Calculate mean x-coordinate
        weighted_sum = sum(c["bbox"]["x1"] * c.get("_weight", 1.0) for c in cluster_cells)
        mean_x = weighted_sum / max(cluster_weight, 1e-6)

        # Collect y-values (use center y for each block)
        y_values = _collect_axis_centers(cluster_cells, axis="y")
        block_spans: List[Tuple[float, float]] = []
        span_candidates: List[Tuple[float, float]] = []
        block_support = 0.0
        for entry in cluster_cells:
            bbox_entry = entry.get("bbox", {})
            span = (bbox_entry.get("y0", 0.0), bbox_entry.get("y1", 0.0))
            span_candidates.append(span)
            if entry.get("_source") == "block":
                block_support += entry.get("_weight", 1.0)
                block_spans.append(span)
        coverage_spans = block_spans if block_spans else span_candidates
        coverage = _compute_span_coverage(coverage_spans, page_height) if coverage_spans else 0.0

        # Determine page position
        if mean_x < page_width * 0.25:
            page_position = "page_left"
        elif mean_x > page_width * 0.75:
            page_position = "page_right"
        else:
            page_position = "mid_page"

        baselines.append({
            "orientation": "right",
            "value": mean_x,
            "page_position": page_position,
            "y_values": y_values,
            "count": int(round(cluster_weight))
        })

    # Process top-aligned baselines (horizontal)
    # Lower threshold to 2 for horizontal baselines (table rows may have fewer items)
    for cluster_key, cluster_cells in top_clusters.items():
        if len(cluster_cells) < 2:  # Require at least 2 items (lowered from 3)
            continue

        cluster_weight = _cluster_weight(cluster_cells)
        block_support = 0.0  # Horizontal clusters currently derive from raw cells only
        block_spans = []
        for entry in cluster_cells:
            bbox_entry = entry.get("bbox", {})
            block_spans.append((bbox_entry.get("x0", 0.0), bbox_entry.get("x1", 0.0)))
        coverage = _compute_span_coverage(block_spans, page_width) if block_spans else 0.0

        # Calculate mean y-coordinate
        mean_y = sum(c["bbox"]["y0"] for c in cluster_cells) / len(cluster_cells)

        # Collect x-values (use center x for each block)
        x_values = [(c["bbox"]["x0"] + c["bbox"]["x1"]) / 2 for c in cluster_cells]

        # Determine page position (if page_height is available)
        if page_height:
            if mean_y < page_height * 0.25:
                page_position = "page_top"
            elif mean_y > page_height * 0.75:
                page_position = "page_bottom"
            else:
                page_position = "mid_page"
        else:
            page_position = "unknown"

        baselines.append({
            "orientation": "top",
            "value": mean_y,
            "page_position": page_position,
            "x_values": x_values,
            "count": int(round(cluster_weight)),
            "block_support": int(round(block_support)),
            "raw_support": max(0, int(round(cluster_weight - block_support))),
            "coverage": round(coverage, 3) if coverage else 0.0,
            "_block_spans": coverage_spans
        })

    # Process bottom-aligned baselines (horizontal)
    # Lower threshold to 2 for horizontal baselines (table rows may have fewer items)
    for cluster_key, cluster_cells in bottom_clusters.items():
        if len(cluster_cells) < 2:  # Require at least 2 items (lowered from 3)
            continue

        cluster_weight = _cluster_weight(cluster_cells)
        block_support = 0.0
        block_spans = []
        for entry in cluster_cells:
            bbox_entry = entry.get("bbox", {})
            block_spans.append((bbox_entry.get("x0", 0.0), bbox_entry.get("x1", 0.0)))
        coverage = _compute_span_coverage(block_spans, page_width) if block_spans else 0.0

        # Calculate mean y-coordinate
        mean_y = sum(c["bbox"]["y1"] for c in cluster_cells) / len(cluster_cells)

        # Collect x-values (use center x for each block)
        x_values = [(c["bbox"]["x0"] + c["bbox"]["x1"]) / 2 for c in cluster_cells]

        # Determine page position (if page_height is available)
        if page_height:
            if mean_y < page_height * 0.25:
                page_position = "page_top"
            elif mean_y > page_height * 0.75:
                page_position = "page_bottom"
            else:
                page_position = "mid_page"
        else:
            page_position = "unknown"

        baselines.append({
            "orientation": "bottom",
            "value": mean_y,
            "page_position": page_position,
            "x_values": x_values,
            "count": int(round(cluster_weight)),
            "block_support": int(round(block_support)),
            "raw_support": max(0, int(round(cluster_weight - block_support))),
            "coverage": round(coverage, 3) if coverage else 0.0,
            "_block_spans": block_spans
        })

    # Sort baselines by count (descending) then by value
    baselines.sort(key=lambda b: (-b["count"], b["value"]))

    # Filter horizontal baselines in close proximity - keep only the one with highest count
    # This prevents multiple overlapping horizontal lines for the same row
    filtered_baselines = []
    proximity_threshold = 5.0  # pt - if horizontal baselines are within 5pt, filter them

    # Group baselines by orientation
    vertical_baselines = [b for b in baselines if b['orientation'] in ['left', 'right']]
    horizontal_baselines = [b for b in baselines if b['orientation'] in ['top', 'bottom']]

    vertical_baselines = _merge_vertical_baselines_list(vertical_baselines, page_height)
    if blocks:
        filtered_vertical = []
        for baseline in vertical_baselines:
            coverage = baseline.get('coverage', 0.0)
            page_pos = baseline.get('page_position')
            if coverage < MIN_BASELINE_COVERAGE and page_pos not in ("page_left", "page_right"):
                continue
            filtered_vertical.append(baseline)
        vertical_baselines = filtered_vertical

    filtered_baselines.extend(vertical_baselines)

    # Filter horizontal baselines by proximity
    if horizontal_baselines:
        # Sort horizontal by count (already sorted from above, but ensure descending order)
        horizontal_baselines.sort(key=lambda b: -b['count'])

        kept_horizontal = []
        for baseline in horizontal_baselines:
            y_value = baseline['value']

            # Check if this baseline is too close to any already-kept baseline
            is_too_close = False
            for kept in kept_horizontal:
                if abs(y_value - kept['value']) < proximity_threshold:
                    # Too close - skip this baseline (keep the one with higher count)
                    is_too_close = True
                    break

            if not is_too_close:
                kept_horizontal.append(baseline)

        filtered_baselines.extend(kept_horizontal)

    # Re-sort after filtering
    filtered_baselines.sort(key=lambda b: (-b["count"], b["value"]))

    for baseline in filtered_baselines:
        baseline.pop("_block_spans", None)
        if 'coverage' not in baseline:
            baseline['coverage'] = 0.0

    return filtered_baselines


# =============================================================================
# BASELINE CONFIDENCE SCORING AND FILTERING (UC-002)
# =============================================================================

# Baseline confidence scoring constants
# Phase 1: Very lenient thresholds to avoid over-filtering during initial implementation
MIN_CONFIDENCE_SCORE = 0.03         # Minimum confidence to retain baseline (very lenient for Phase 1)
MIN_SUPPORT_COUNT = 1.0             # Minimum weighted support count (allow singletons for Phase 1)
MIN_COVERAGE_FOR_MID_PAGE = 0.0     # Disabled for Phase 1 (baselines without blocks have 0 coverage)
MIN_CONSISTENCY_SCORE = 0.0         # Minimum consistency (0-1, where 1 is perfect)
PAGE_EDGE_MARGIN = 50.0             # pt - consider baseline near edge if within this margin


def calculate_baseline_consistency(baseline: Dict[str, Any]) -> float:
    """
    Calculate consistency score for a baseline (how tightly cells cluster around it).

    Uses y-values (for vertical baselines) or x-values (for horizontal baselines)
    to measure standard deviation. Lower deviation = higher consistency.

    Returns:
        Consistency score from 0.0 (poor) to 1.0 (perfect)
    """
    orientation = baseline.get("orientation", "")

    # Get coordinate values
    if orientation in ("left", "right"):
        values = baseline.get("y_values", [])
    elif orientation in ("top", "bottom"):
        values = baseline.get("x_values", [])
    else:
        return 0.0

    if not values or len(values) < 2:
        return 1.0  # Single point is perfectly consistent

    # Calculate standard deviation
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std_dev = variance ** 0.5

    # Normalize to 0-1 scale
    # Use 50pt as maximum expected deviation for score=0
    # Anything below 5pt gets score close to 1.0
    max_deviation = 50.0
    normalized_score = max(0.0, 1.0 - (std_dev / max_deviation))

    return round(normalized_score, 3)


def calculate_page_edge_proximity(baseline: Dict[str, Any], page_width: float, page_height: Optional[float] = None) -> float:
    """
    Calculate distance to nearest page edge.

    Returns:
        Distance in points to nearest edge (0.0 if on edge)
    """
    orientation = baseline.get("orientation", "")
    value = baseline.get("value", 0.0)

    if orientation in ("left", "right"):
        # Vertical baseline - measure distance to left/right edges
        dist_to_left = abs(value - 0.0)
        dist_to_right = abs(value - page_width) if page_width else float('inf')
        return min(dist_to_left, dist_to_right)

    elif orientation in ("top", "bottom"):
        # Horizontal baseline - measure distance to top/bottom edges
        if not page_height:
            return float('inf')
        dist_to_top = abs(value - 0.0)
        dist_to_bottom = abs(value - page_height)
        return min(dist_to_top, dist_to_bottom)

    return float('inf')


def calculate_baseline_confidence(
    baseline: Dict[str, Any],
    page_width: float,
    page_height: Optional[float] = None
) -> float:
    """
    Calculate overall confidence score for a baseline.

    Combines multiple factors:
    - Support count (weighted by blocks)
    - Coverage (percentage of page covered)
    - Consistency (how tightly aligned)
    - Page edge proximity (baselines near edges are more reliable)

    Returns:
        Confidence score from 0.0 to 1.0
    """
    # Extract metrics
    count = baseline.get("count", 0)
    block_support = baseline.get("block_support", 0)
    coverage = baseline.get("coverage", 0.0)
    consistency = calculate_baseline_consistency(baseline)
    edge_proximity = calculate_page_edge_proximity(baseline, page_width, page_height)

    # Support score (0-1): normalize by typical maximum of 10 weighted items
    # (changed from 20 to make scoring more lenient for low-count baselines)
    support_score = min(1.0, (count + block_support * 0.5) / 10.0)

    # Coverage score (already 0-1)
    coverage_score = coverage

    # Edge proximity bonus (0-0.2): baselines within PAGE_EDGE_MARGIN get bonus
    edge_bonus = 0.2 if edge_proximity <= PAGE_EDGE_MARGIN else 0.0

    # Weighted combination
    # - Support: 40% weight (most important)
    # - Coverage: 30% weight (important for mid-page baselines)
    # - Consistency: 20% weight (helpful but not critical)
    # - Edge bonus: 10% weight (tie-breaker)
    confidence = (
        support_score * 0.40 +
        coverage_score * 0.30 +
        consistency * 0.20 +
        edge_bonus
    )

    return round(confidence, 3)


def filter_baselines_by_confidence(
    baselines: List[Dict[str, Any]],
    page_width: float,
    page_height: Optional[float] = None,
    debug: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Filter baselines by confidence score, removing low-quality baselines.

    Applies minimum thresholds:
    - Confidence score
    - Support count
    - Coverage (for mid-page baselines)
    - Consistency

    Parameters
    ----------
    baselines : List[Dict]
        List of baseline dictionaries
    page_width : float
        Page width in points
    page_height : Optional[float]
        Page height in points
    debug : Optional[Dict]
        Debug dictionary to populate with filtering info

    Returns
    -------
    List[Dict]
        Filtered baselines that meet minimum quality thresholds
    """
    if debug is not None:
        debug["baseline_filtering"] = {
            "total_baselines_input": len(baselines),
            "filtered_baselines": [],
            "retained_baselines": []
        }

    retained = []

    for baseline in baselines:
        # Calculate all scores
        confidence = calculate_baseline_confidence(baseline, page_width, page_height)
        consistency = calculate_baseline_consistency(baseline)
        edge_proximity = calculate_page_edge_proximity(baseline, page_width, page_height)

        # Add scores to baseline
        baseline["confidence"] = confidence
        baseline["consistency"] = consistency
        baseline["edge_proximity"] = round(edge_proximity, 2)

        # Extract metrics for filtering
        count = baseline.get("count", 0)
        coverage = baseline.get("coverage", 0.0)
        page_position = baseline.get("page_position", "unknown")
        orientation = baseline.get("orientation", "unknown")

        # Apply filters
        reasons_for_rejection = []

        # Filter 1: Minimum confidence
        if confidence < MIN_CONFIDENCE_SCORE:
            reasons_for_rejection.append(f"low_confidence ({confidence:.3f} < {MIN_CONFIDENCE_SCORE})")

        # Filter 2: Minimum support
        if count < MIN_SUPPORT_COUNT:
            reasons_for_rejection.append(f"low_support ({count} < {MIN_SUPPORT_COUNT})")

        # Filter 3: Mid-page VERTICAL baselines need higher coverage
        # (Coverage only applies to vertical baselines - left/right)
        if (orientation in ("left", "right") and
            page_position == "mid_page" and
            coverage < MIN_COVERAGE_FOR_MID_PAGE):
            reasons_for_rejection.append(f"low_coverage_mid_page ({coverage:.3f} < {MIN_COVERAGE_FOR_MID_PAGE})")

        # Filter 4: Minimum consistency
        if consistency < MIN_CONSISTENCY_SCORE:
            reasons_for_rejection.append(f"low_consistency ({consistency:.3f} < {MIN_CONSISTENCY_SCORE})")

        # Decision
        if reasons_for_rejection:
            if debug is not None:
                debug["baseline_filtering"]["filtered_baselines"].append({
                    "orientation": orientation,
                    "value": round(baseline.get("value", 0.0), 2),
                    "count": count,
                    "confidence": confidence,
                    "coverage": coverage,
                    "consistency": consistency,
                    "reasons": reasons_for_rejection
                })
        else:
            retained.append(baseline)
            if debug is not None:
                debug["baseline_filtering"]["retained_baselines"].append({
                    "orientation": orientation,
                    "value": round(baseline.get("value", 0.0), 2),
                    "count": count,
                    "confidence": confidence,
                    "coverage": coverage,
                    "consistency": consistency,
                    "page_position": page_position
                })

    if debug is not None:
        debug["baseline_filtering"]["total_baselines_retained"] = len(retained)
        debug["baseline_filtering"]["total_baselines_filtered"] = len(baselines) - len(retained)

    return retained


def refine_baselines_with_confidence_filtering(
    baselines: List[Dict[str, Any]],
    page_width: float,
    page_height: Optional[float] = None,
    debug: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Apply confidence-based filtering to refine baseline list.

    This is the main entry point for UC-002 baseline refinement.

    Steps:
    1. Calculate confidence scores for all baselines
    2. Filter baselines below minimum thresholds
    3. Re-sort by confidence (descending)

    Parameters
    ----------
    baselines : List[Dict]
        Input baselines from detect_alignment_baselines
    page_width : float
        Page width in points
    page_height : Optional[float]
        Page height in points
    debug : Optional[Dict]
        Debug dictionary to populate

    Returns
    -------
    List[Dict]
        Refined baselines with confidence scores
    """
    if not baselines:
        return []

    # Filter by confidence
    refined_baselines = filter_baselines_by_confidence(
        baselines,
        page_width,
        page_height,
        debug=debug
    )

    # Re-sort by confidence (descending), then by count
    refined_baselines.sort(key=lambda b: (-b.get("confidence", 0.0), -b.get("count", 0)))

    return refined_baselines


# =============================================================================
# HORIZONTAL ALIGNMENT ANOMALY DETECTION
# =============================================================================

def detect_horizontal_alignment_anomalies(
    cells: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]],
    page_num: int,
    page_metadata: Optional[List[Dict[str, Any]]] = None,
    analyze_value_content_type_func=None,
    is_common_english_phrase_func=None
) -> List[Dict[str, Any]]:
    """
    Detect text items that overlap an existing horizontal baseline without
    receiving support from a lower baseline (strong indicator of pasted text
    that was not vertically aligned back onto the row grid).

    Args:
        cells: List of cell dictionaries
        baselines: List of detected baselines
        page_num: Page number
        page_metadata: Optional metadata for filtering
        analyze_value_content_type_func: Function to analyze content type (imported from colon_spacing_detection)
        is_common_english_phrase_func: Function to check if phrase is common (imported from colon_spacing_detection)
    """
    if not cells or not baselines:
        return []

    # Filter out structural elements
    filtered_cells = []
    for cell in cells:
        if page_metadata:
            match = match_alignment_metadata(cell["bbox"], page_metadata, min_iou=0.6)
            if match:
                if match.get("column_header") or match.get("row_header"):
                    continue
                if match.get("label") == "section_header":
                    continue
        filtered_cells.append(cell)

    if not filtered_cells:
        return []

    # Focus on bottom-oriented baselines (represent row bottoms)
    bottom_baselines = sorted(
        (b for b in baselines if b.get("orientation") == "bottom"),
        key=lambda b: b["value"]
    )
    if not bottom_baselines:
        return []

    from bisect import bisect_left, bisect_right

    bottom_values = [b["value"] for b in bottom_baselines]

    # Tuning constants (conservative to minimise false positives)
    INTERIOR_MARGIN = 0.6  # points - avoid catching baselines that merely touch edges
    MIN_BASELINE_COUNT = 4  # baseline must be supported by enough items to be trusted
    MIN_TOP_OFFSET = 2.5  # baseline must sit at least this far below the top of the text box
    MIN_BOTTOM_OFFSET = 2.5  # and at least this far above the bottom of the text box
    MIN_OFFSET_RATIO = 0.3  # baseline must sit away from edges (>=30% of height)
    MAX_OFFSET_RATIO = 0.7  # but still within the interior (<=70% from the top/bottom)
    MAX_OFFSET_DIFF = 1.75  # top/bottom offsets should be similar (baseline through middle)
    MIN_GAP_BELOW = 4.0  # require a decent empty gap before the next baseline below
    MIN_GROUP_CHAR_COUNT = 5  # aggregated text must have substance
    MAX_GROUP_GAP = 14.0  # max space (pt) between adjacent segments in a group

    candidate_cells: List[Dict[str, Any]] = []

    for cell in filtered_cells:
        text = (cell.get("text") or "").strip()
        bbox = cell["bbox"]

        if not text or not any(ch.isalnum() for ch in text):
            continue

        y0 = bbox["y0"]
        y1 = bbox["y1"]
        height = y1 - y0

        if height <= 2.0:  # extremely thin boxes typically noise
            continue

        search_low = y0 + INTERIOR_MARGIN
        search_high = y1 - INTERIOR_MARGIN

        start_idx = bisect_right(bottom_values, search_low)
        end_idx = bisect_left(bottom_values, search_high)

        if start_idx >= end_idx:
            continue

        for idx in range(start_idx, end_idx):
            baseline = bottom_baselines[idx]
            baseline_value = baseline["value"]
            baseline_count = baseline.get("count", 0)

            if baseline_count < MIN_BASELINE_COUNT:
                continue

            top_offset = baseline_value - y0
            bottom_offset = y1 - baseline_value

            if top_offset < MIN_TOP_OFFSET or bottom_offset < MIN_BOTTOM_OFFSET:
                continue

            ratio_top = top_offset / height
            ratio_bottom = bottom_offset / height

            if ratio_top < MIN_OFFSET_RATIO or ratio_bottom < MIN_OFFSET_RATIO:
                continue
            if ratio_top > MAX_OFFSET_RATIO or ratio_bottom > MAX_OFFSET_RATIO:
                continue
            if abs(top_offset - bottom_offset) > MAX_OFFSET_DIFF:
                continue

            next_idx = bisect_left(bottom_values, y1 - 1e-6)
            gap_below = bottom_values[next_idx] - y1 if next_idx < len(bottom_values) else None
            if gap_below is not None and gap_below < MIN_GAP_BELOW:
                continue

            candidate_cells.append({
                "cell": cell,
                "baseline_value": baseline_value,
                "baseline_count": baseline_count,
                "top_offset": top_offset,
                "bottom_offset": bottom_offset,
                "height": height,
                "gap_below": gap_below
            })
            break  # stop after first qualifying baseline inside this cell

    if not candidate_cells:
        return []

    grouped_candidates: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_cells:
        key = round(candidate["baseline_value"], 2)  # stabilise float precision
        grouped_candidates[key].append(candidate)

    anomalies: List[Dict[str, Any]] = []

    for candidate_group in grouped_candidates.values():
        candidate_group.sort(key=lambda c: c["cell"]["bbox"]["x0"])

        # Split into contiguous segments along the baseline
        segments: List[List[Dict[str, Any]]] = []
        current_segment: List[Dict[str, Any]] = [candidate_group[0]]

        for candidate in candidate_group[1:]:
            prev_bbox = current_segment[-1]["cell"]["bbox"]
            candidate_bbox = candidate["cell"]["bbox"]
            gap = candidate_bbox["x0"] - prev_bbox["x1"]

            if gap > MAX_GROUP_GAP:
                segments.append(current_segment)
                current_segment = [candidate]
            else:
                current_segment.append(candidate)

        segments.append(current_segment)

        for segment in segments:
            segment_text = " ".join(item["cell"]["text"] for item in segment).strip()
            char_count = len(segment_text.replace(" ", ""))

            if char_count < MIN_GROUP_CHAR_COUNT:
                continue

            x0 = min(item["cell"]["bbox"]["x0"] for item in segment)
            y0 = min(item["cell"]["bbox"]["y0"] for item in segment)
            x1 = max(item["cell"]["bbox"]["x1"] for item in segment)
            y1 = max(item["cell"]["bbox"]["y1"] for item in segment)

            top_offsets = [item["top_offset"] for item in segment]
            bottom_offsets = [item["bottom_offset"] for item in segment]
            gap_values = [item["gap_below"] for item in segment if item["gap_below"] is not None]

            avg_baseline = sum(item["baseline_value"] for item in segment) / len(segment)
            baseline_count = max(item["baseline_count"] for item in segment)

            details_lines = [
                f"Baseline intersects text block at y={avg_baseline:.2f}pt (count={baseline_count}).",
                f"Top offset range: {min(top_offsets):.2f}pt to {max(top_offsets):.2f}pt.",
                f"Bottom offset range: {min(bottom_offsets):.2f}pt to {max(bottom_offsets):.2f}pt."
            ]

            if gap_values:
                details_lines.append(f"Next baseline below is {min(gap_values):.2f}pt away.")
            else:
                details_lines.append("No supporting baseline detected below this text.")

            component_texts = [item["cell"]["text"] for item in segment]
            details_lines.append(f"Component segments: {', '.join(component_texts)}.")

            # Use content analysis if function is provided
            classification = "horizontal_misalignment"
            severity = "high"
            counter_evidence: Optional[Dict[str, str]] = None

            if analyze_value_content_type_func and is_common_english_phrase_func:
                content_info = analyze_value_content_type_func(segment_text)
                if content_info.get("content_type") == "text" and is_common_english_phrase_func(segment_text):
                    classification = "horizontal_misalignment_low_risk"
                    severity = "low"
                    counter_evidence = {
                        "title": "Counter Evidence: Text unlikely to have been forged.",
                        "details": "Not a name/date/currency/number"
                    }
            else:
                content_info = None

            anomaly = {
                "page": page_num,
                "text": segment_text,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "classification": classification,
                "reason": "Text overlaps prior baseline without nearby support below.",
                "details": "\n".join(details_lines),
                "font_info": next((item["cell"].get("font_info") for item in segment if item["cell"].get("font_info")), None),
                "detection_pattern": "baseline_intrusion_no_support",
                "components": component_texts,
                "baseline_value": avg_baseline,
                "baseline_count": baseline_count,
                "content_analysis": content_info,
                "severity": severity
            }

            if counter_evidence:
                anomaly["counter_evidence"] = counter_evidence

            anomalies.append(anomaly)

    return anomalies
