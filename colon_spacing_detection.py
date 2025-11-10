"""
Colon Spacing Detection Module
===============================
Detects spacing inconsistencies in label:value pairs separated by colons.

This module is part of the document forgery detection system.
See ALIGNMENT_AND_COLON_LOGIC_SUMMARY.md for detailed documentation.
"""

import re
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict


# =============================================================================
# CONSTANTS
# =============================================================================

# Colon pattern detection thresholds (for label:value spacing analysis)
COLON_SPACING_TOLERANCE = 2.0          # pt - deviation threshold for flagging anomalies
COLON_MAX_DISTANCE = 75.0              # pt - maximum label-to-value distance (increased from 50.0)
COLON_VALUE_CHAIN_MAX_DISTANCE = 20.0  # pt - maximum gap between chained value cells
COLON_BASELINE_TOLERANCE = 3.0         # pt - vertical alignment tolerance (increased from 2.0)
COLON_MIN_CLUSTER_SIZE = 3             # minimum items to establish a pattern
COLON_RIGHT_ALIGN_TOLERANCE = 1.5      # pt - right edge alignment tolerance


# =============================================================================
# CONTENT TYPE ANALYSIS (NLP-BASED)
# =============================================================================

# Set of common English terms used to downgrade misalignment suspicion when
# the detected text does not resemble a proper noun, number, date, or currency.
# These include generic document labels that frequently appear in statements,
# payslips, invoices, etc. Words should remain lowercase for normalization.
COMMON_ENGLISH_WORDS: Set[str] = {
    'a', 'an', 'and', 'at', 'by', 'for', 'from', 'in', 'of', 'on', 'or', 'the', 'to', 'with',
    'account', 'accounts', 'accrual', 'adjustment', 'allocation', 'amount', 'amounts', 'analysis',
    'balance', 'balances', 'bank', 'category', 'charge', 'charges', 'closing', 'column', 'columns',
    'code', 'contact', 'credit', 'current', 'debit', 'deduction', 'deductions', 'department',
    'description', 'detail', 'details', 'document', 'due', 'earnings', 'email', 'employee',
    'employees', 'expenses', 'fee', 'fees', 'floor', 'footer', 'grand', 'header', 'hours',
    'information', 'invoice', 'item', 'items', 'job', 'known', 'leave', 'line', 'lines', 'memo',
    'message', 'method', 'month', 'note', 'notes', 'opening', 'outstanding', 'overview',
    'page', 'pay', 'payment', 'period', 'permanent', 'previous', 'rate', 'reason', 'reference',
    'remarks', 'report', 'row', 'rows', 'salary', 'schedule', 'section', 'service', 'slip',
    'statement', 'status', 'subtotal', 'summary', 'support', 'switch', 'table', 'tax',
    'title', 'total', 'totals', 'type', 'unit', 'units', 'value', 'year'
}


def is_common_english_phrase(text: str) -> bool:
    """
    Determine whether the supplied text is composed entirely of common English
    words (ignoring punctuation). Used to add counter evidence for low-risk
    misalignment detections.
    """
    if not text:
        return False

    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return False

    # Require that every token be recognised as common; this keeps addresses
    return all(word in COMMON_ENGLISH_WORDS for word in words)


def analyze_value_content_type(text: str) -> Dict[str, Any]:
    """
    Analyze the content type of a value to determine suspiciousness level.

    Uses NLP-like heuristics to detect:
    - Names (proper nouns with title case)
    - Currency amounts
    - Numbers
    - Dates

    Returns:
        Dict with:
        - content_type: 'name', 'currency', 'number', 'date', 'text'
        - suspicion_multiplier: float (1.0 = normal, >1.0 = more suspicious)
        - details: str describing the content type
    """
    text = text.strip()

    # Currency detection (with symbols or common patterns)
    currency_patterns = [
        r'^\$?\s*\d{1,3}(,\d{3})*(\.\d{2})?$',  # $1,234.56 or 1,234.56
        r'^\d{1,3}(,\d{3})*(\.\d{2})?\s*$',     # 1,234.56
        r'^R\s*\d+(\.\d{2})?$',                   # R1234.56 (South African Rand)
        r'^€\s*\d+(\.\d{2})?$',                   # €1234.56
        r'^£\s*\d+(\.\d{2})?$',                   # £1234.56
        r'^\d+\.\d{2}$',                          # 123.45 (likely currency)
    ]
    for pattern in currency_patterns:
        if re.match(pattern, text):
            return {
                'content_type': 'currency',
                'suspicion_multiplier': 2.5,  # Very suspicious if currency is misaligned
                'details': 'Currency amount'
            }

    # Number detection (including IDs, phone numbers, etc.)
    number_patterns = [
        r'^\d+$',                                # Pure digits
        r'^\d{2,}[/-]\d{2,}[/-]\d{2,}$',        # ID numbers like 12-345-67
        r'^\d{10,}$',                            # Long numbers (likely IDs)
        r'^[A-Z]{1,3}\d+$',                      # Codes like AB123
    ]
    for pattern in number_patterns:
        if re.match(pattern, text):
            # Check if it's a date first
            if re.match(r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$', text) or \
               re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}$', text):
                return {
                    'content_type': 'date',
                    'suspicion_multiplier': 1.0,  # Dates are less suspicious
                    'details': 'Date'
                }
            return {
                'content_type': 'number',
                'suspicion_multiplier': 2.0,  # Suspicious if number is misaligned
                'details': 'Numeric value or ID'
            }

    # Name detection (title case with multiple words)
    # Names typically have:
    # - Title case (first letter capitalized)
    # - 2-4 words
    # - Mostly alphabetic characters
    words = text.split()
    if len(words) >= 2 and len(words) <= 5:
        # Check if all words are title case and mostly alphabetic
        is_title_case = all(
            word[0].isupper() and word[1:].islower() if len(word) > 1 else word.isupper()
            for word in words if word.isalpha()
        )
        is_mostly_alpha = sum(1 for w in words if w.isalpha()) >= len(words) * 0.8

        if is_title_case and is_mostly_alpha:
            return {
                'content_type': 'name',
                'suspicion_multiplier': 3.0,  # VERY suspicious if name is misaligned
                'details': 'Likely a person name'
            }

    # Default to text
    return {
        'content_type': 'text',
        'suspicion_multiplier': 1.0,
        'details': 'Text value'
    }


# =============================================================================
# COLON PAIR EXTRACTION
# =============================================================================

def is_excluded_colon(text: str) -> bool:
    """
    Check if text contains a colon that should be excluded (time, date, URL, ratio).
    """
    # Time patterns: HH:MM or HH:MM:SS
    if re.search(r'\d{1,2}:\d{2}(:\d{2})?', text):
        return True

    # URL patterns: http://, https://, mailto:
    if re.search(r'(https?|mailto|ftp):', text, re.IGNORECASE):
        return True

    # Ratio patterns: N:M (e.g., 16:9, 4:3)
    if re.search(r'\b\d+:\d+\b', text):
        return True

    return False


def is_excluded_value(text: str) -> bool:
    """
    Check if a value text should be excluded from colon pattern analysis.
    Excludes dates, times, URLs, etc.
    """
    # Date patterns: YYYY/MM/DD, DD/MM/YYYY, YYYY-MM-DD
    if re.search(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', text):
        return True
    if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', text):
        return True

    # Time patterns: HH:MM or HH:MM:SS
    if re.search(r'\d{1,2}:\d{2}(:\d{2})?', text):
        return True

    # URL patterns
    if re.search(r'(https?|mailto|ftp)://', text, re.IGNORECASE):
        return True

    return False


def extract_colon_pairs(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract label:value pairs from cells containing colons.

    Handles two cases:
    1. Label and value in same cell: "LABEL: VALUE"
    2. Label and value in separate cells: "LABEL:" in one cell, "VALUE" in next cell

    Returns list of pairs with spacing information.
    """
    pairs = []

    for i, cell in enumerate(cells):
        text = cell['text']
        bbox = cell['bbox']
        font_info = cell.get('font_info')

        # Skip if colon is part of time/date/URL
        if is_excluded_colon(text):
            continue

        # Case 1: Label ends with colon (separate cells)
        if text.endswith(':'):
            label_text = text
            label_bbox = bbox

            # Find value cell (next cell on same baseline, to the right)
            value_cell = None
            label_center_y = (bbox['y0'] + bbox['y1']) / 2

            for j in range(i + 1, len(cells)):
                candidate = cells[j]
                candidate_bbox = candidate['bbox']

                # Check if on same baseline (within 2pt)
                candidate_center_y = (candidate_bbox['y0'] + candidate_bbox['y1']) / 2
                if abs(candidate_center_y - label_center_y) > COLON_BASELINE_TOLERANCE:
                    continue

                # Check if to the right
                if candidate_bbox['x0'] <= bbox['x1']:
                    continue

                # Check if reasonably close
                horizontal_gap = candidate_bbox['x0'] - bbox['x1']
                if horizontal_gap > COLON_MAX_DISTANCE:
                    continue

                # Found potential value cell
                value_cell = candidate
                break

            if not value_cell:
                continue

            # Collect all contiguous value cells on the same baseline (captures "CAPITEC BANK", etc.)
            value_cells = [value_cell]
            last_bbox = value_cell['bbox']

            for k in range(j + 1, len(cells)):
                next_candidate = cells[k]
                next_bbox = next_candidate['bbox']
                next_text = next_candidate['text']

                # Stop if we reach another label containing a colon
                if ':' in next_text:
                    break

                # Ensure same baseline within tolerance
                next_center_y = (next_bbox['y0'] + next_bbox['y1']) / 2
                if abs(next_center_y - label_center_y) > COLON_BASELINE_TOLERANCE:
                    break

                # Ensure it is to the right and reasonably close to previous segment
                if next_bbox['x0'] <= last_bbox['x1']:
                    continue
                gap_to_previous = next_bbox['x0'] - last_bbox['x1']
                if gap_to_previous > COLON_VALUE_CHAIN_MAX_DISTANCE:
                    break

                value_cells.append(next_candidate)
                last_bbox = next_bbox

            # Combine value text and bbox
            value_text = " ".join(cell['text'] for cell in value_cells).strip()
            value_font_info = value_cells[0].get('font_info')

            combined_value_bbox = {
                'x0': min(cell['bbox']['x0'] for cell in value_cells),
                'y0': min(cell['bbox']['y0'] for cell in value_cells),
                'x1': max(cell['bbox']['x1'] for cell in value_cells),
                'y1': max(cell['bbox']['y1'] for cell in value_cells)
            }

            # Calculate spacing (from label right edge to value left edge)
            spacing = combined_value_bbox['x0'] - label_bbox['x1']

            # Compute highlight bbox starting at the colon (right edge of label)
            average_char_width = (label_bbox['x1'] - label_bbox['x0']) / max(len(label_text), 1)
            colon_estimate = label_bbox['x1'] - average_char_width
            colon_highlight_start = max(label_bbox['x0'], min(colon_estimate, value_cells[0]['bbox']['x0']))
            highlight_bbox = {
                'x0': colon_highlight_start,
                'y0': min(label_bbox['y0'], combined_value_bbox['y0']),
                'x1': combined_value_bbox['x1'],
                'y1': max(label_bbox['y1'], combined_value_bbox['y1'])
            }

            pairs.append({
                'label': label_text,
                'value': value_text,
                'label_bbox': label_bbox,
                'value_bbox': combined_value_bbox,
                'spacing': spacing,
                'full_text': f"{label_text} {value_text}",
                'full_bbox': bbox,
                'is_same_cell': False,  # Separate cells
                'font_info': value_font_info,  # Font info from value cell
                'highlight_bbox': highlight_bbox
            })

        # Case 2: Label and value in same cell: "LABEL: VALUE"
        # Analyze these too, but mark them as same-cell so we can check alignment differently
        elif ':' in text:
            colon_idx = text.find(':')
            label_text = text[:colon_idx + 1].strip()
            value_text = text[colon_idx + 1:].strip()

            # Skip if no value after colon
            if not value_text:
                continue

            # Estimate positions (simplified - assumes monospace-ish layout)
            text_width = bbox['x1'] - bbox['x0']
            colon_ratio = (colon_idx + 1) / len(text)

            # Estimate colon position
            colon_x = bbox['x0'] + (text_width * colon_ratio)

            # Estimate value start position
            value_start_ratio = (colon_idx + 1 + (len(text) - len(text[colon_idx + 1:].lstrip()))) / len(text)
            value_x0 = bbox['x0'] + (text_width * value_start_ratio)

            # Calculate spacing (distance from colon to value start)
            spacing = value_x0 - colon_x

            # Create label and value bboxes (estimates)
            label_bbox = {
                'x0': bbox['x0'],
                'y0': bbox['y0'],
                'x1': colon_x,
                'y1': bbox['y1']
            }

            value_bbox = {
                'x0': value_x0,
                'y0': bbox['y0'],
                'x1': bbox['x1'],
                'y1': bbox['y1']
            }

            # Highlight from the colon position through the value
            colon_highlight_start = max(bbox['x0'], colon_x - 1.0)
            average_char_width = max((label_bbox['x1'] - label_bbox['x0']) / max(len(label_text), 1), 0)
            colon_highlight_start = max(label_bbox['x0'], label_bbox['x1'] - average_char_width)
            highlight_bbox = {
                'x0': colon_highlight_start,
                'y0': bbox['y0'],
                'x1': bbox['x1'],
                'y1': bbox['y1']
            }

            pairs.append({
                'label': label_text,
                'value': value_text,
                'label_bbox': label_bbox,
                'value_bbox': value_bbox,
                'spacing': spacing,
                'full_text': text,
                'full_bbox': bbox,
                'is_same_cell': True,  # Same cell - will check alignment instead of spacing
                'font_info': font_info,  # Font info from cell
                'highlight_bbox': highlight_bbox
            })

    return pairs


# =============================================================================
# SPACING ANALYSIS
# =============================================================================

def analyze_colon_spacing(pairs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Analyze spacing patterns in colon pairs to find dominant spacing.

    Returns:
        Dict with dominant_spacing and cluster_info, or None if insufficient data
    """
    if len(pairs) < COLON_MIN_CLUSTER_SIZE:
        return None

    # Collect all spacing values
    spacings = [p['spacing'] for p in pairs]

    # Bucket spacings at 0.5pt precision
    spacing_buckets: Dict[float, List[float]] = defaultdict(list)
    for spacing in spacings:
        bucket_key = round(spacing * 2) / 2  # Round to nearest 0.5pt
        spacing_buckets[bucket_key].append(spacing)

    # Find dominant cluster (most frequent)
    if not spacing_buckets:
        return None

    dominant_bucket, dominant_items = max(
        spacing_buckets.items(),
        key=lambda kv: len(kv[1])
    )

    # Require minimum cluster size
    if len(dominant_items) < COLON_MIN_CLUSTER_SIZE:
        return None

    # Calculate mean spacing for dominant cluster
    dominant_spacing = sum(dominant_items) / len(dominant_items)

    return {
        'dominant_spacing': dominant_spacing,
        'dominant_count': len(dominant_items),
        'total_pairs': len(pairs)
    }


def detect_aligned_values_using_baselines(
    pairs: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]]
) -> Dict[str, set]:
    """
    Detect which value bboxes are aligned to detected baselines (exception to spacing rules).

    Uses the detected alignment baselines instead of custom clustering.

    Args:
        pairs: List of colon pairs
        baselines: List of alignment baselines from detect_alignment_baselines

    Returns:
        Dict with 'left_aligned' and 'right_aligned' sets of pair indices
    """
    if not pairs or not baselines:
        return {'left_aligned': set(), 'right_aligned': set()}

    left_aligned_indices = set()
    right_aligned_indices = set()

    # Extract left and right baselines
    left_baselines = [b['value'] for b in baselines if b.get('orientation') == 'left' and b.get('count', 0) >= 3]
    right_baselines = [b['value'] for b in baselines if b.get('orientation') == 'right' and b.get('count', 0) >= 3]

    # Check each pair against baselines
    for idx, pair in enumerate(pairs):
        value_bbox = pair['value_bbox']
        left_edge = value_bbox['x0']
        right_edge = value_bbox['x1']

        # Check left alignment (within tolerance)
        for baseline_x in left_baselines:
            if abs(left_edge - baseline_x) <= COLON_RIGHT_ALIGN_TOLERANCE:
                left_aligned_indices.add(idx)
                break

        # Check right alignment (within tolerance)
        for baseline_x in right_baselines:
            if abs(right_edge - baseline_x) <= COLON_RIGHT_ALIGN_TOLERANCE:
                right_aligned_indices.add(idx)
                break

    return {
        'left_aligned': left_aligned_indices,
        'right_aligned': right_aligned_indices
    }


def detect_right_aligned_values(pairs: List[Dict[str, Any]]) -> set:
    """
    Detect which value bboxes are right-aligned (exception to spacing rules).

    DEPRECATED: This function uses custom clustering. Use detect_aligned_values_using_baselines instead.

    Returns:
        Set of indices of right-aligned pairs
    """
    if len(pairs) < 2:  # Reduced from 3 to 2 for better detection
        return set()

    # Cluster right edges at 1.5pt precision (increased from 0.5pt for better tolerance)
    right_edge_buckets: Dict[float, List[int]] = defaultdict(list)

    for idx, pair in enumerate(pairs):
        right_edge = pair['value_bbox']['x1']
        bucket_key = round(right_edge * 0.67) / 0.67  # Round to nearest 1.5pt (1/0.67 ≈ 1.5)
        right_edge_buckets[bucket_key].append(idx)

    # Find clusters with ≥2 items (reduced from ≥3 for better detection)
    right_aligned_indices = set()
    for bucket_key, indices in right_edge_buckets.items():
        if len(indices) >= 2:  # Reduced from 3 to 2 for better detection
            right_aligned_indices.update(indices)

    return right_aligned_indices


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def detect_colon_spacing_anomalies(
    docling_payload: Dict[str, Any],
    extract_bbox_coords_func,
    reporter: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Detect spacing inconsistencies in label:value pairs separated by colons.

    This detects forgery by finding deviations from the dominant spacing pattern
    in colon-separated label-value pairs (e.g., "EMP.CODE: HM20927").

    Now includes NLP-based content analysis to flag names, currency, and numbers
    as more suspicious when misaligned.

    Args:
        docling_payload: The Docling payload dictionary
        extract_bbox_coords_func: Function to extract bbox coords (imported from alignment_detection)

    Returns:
        List of anomalies with classification:
        - 'consistent': Green - matches dominant pattern
        - 'deviation': Red - spacing deviation (potential forgery)
        - 'right_aligned': Orange - right-aligned exception
    """
    if not isinstance(docling_payload, dict):
        return []

    parsed_pages = docling_payload.get("parsed_pages")
    if not isinstance(parsed_pages, dict):
        return []

    all_anomalies: List[Dict[str, Any]] = []

    for page_key, page_data in parsed_pages.items():
        if not isinstance(page_data, dict):
            continue

        cells = page_data.get("cells", [])
        if not isinstance(cells, list) or not cells:
            if reporter:
                reporter.record_colon_analysis(page_no, {
                    "status": "no_cells_on_page",
                    "cell_count": len(cells) if isinstance(cells, list) else 0
                })
            continue

        page_no = page_data.get("page_no")
        if page_no is None:
            try:
                page_no = int(page_key)
            except (TypeError, ValueError):
                continue

        # Normalize ALL cells (not just those with colons - we need values too!)
        normalized_cells = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue

            text = (cell.get("text") or "").strip()
            if not text:
                continue

            rect = cell.get("rect") or cell.get("bbox")
            if not isinstance(rect, dict):
                continue

            try:
                x0, y0, x1, y1 = extract_bbox_coords_func(rect)
            except Exception:
                continue

            # Extract font information if available
            font_info = {}
            if 'font' in cell:
                font_info['font'] = cell['font']
            if 'font_size' in cell:
                font_info['font_size'] = cell['font_size']
            if 'font_name' in cell:
                font_info['font_name'] = cell['font_name']
            if 'font_family' in cell:
                font_info['font_family'] = cell['font_family']

            normalized_cells.append({
                "text": text,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "font_info": font_info if font_info else None
            })

        if not normalized_cells:
            if reporter:
                reporter.record_colon_analysis(page_no, {
                    "status": "no_text_cells_after_normalization"
                })
            continue

        # Extract colon pairs
        colon_pairs = extract_colon_pairs(normalized_cells)

        if len(colon_pairs) < 2:  # Need at least 2 pairs to establish a pattern
            if reporter:
                reporter.record_colon_analysis(page_no, {
                    "status": "insufficient_colon_pairs",
                    "pair_count": len(colon_pairs)
                })
            continue

        # Analyze spacing patterns
        spacing_analysis = analyze_colon_spacing(colon_pairs)

        if not spacing_analysis:
            if reporter:
                reporter.record_colon_analysis(page_no, {
                    "status": "no_spacing_pattern_detected",
                    "pair_count": len(colon_pairs)
                })
            continue

        dominant_spacing = spacing_analysis['dominant_spacing']
        page_report = {
            "status": "analyzed",
            "pair_count": len(colon_pairs),
            "dominant_spacing": dominant_spacing,
            "pairs": []
        }

        # Get baselines for this page (if available)
        baselines_by_page = docling_payload.get("_alignment_baselines", {})
        page_baselines = baselines_by_page.get(page_no, [])

        # Detect aligned values using baselines (preferred) or fallback to old method
        if page_baselines:
            aligned_values = detect_aligned_values_using_baselines(colon_pairs, page_baselines)
            left_aligned_indices = aligned_values['left_aligned']
            right_aligned_indices = aligned_values['right_aligned']
        else:
            # Fallback to old clustering method
            left_aligned_indices = set()
            right_aligned_indices = detect_right_aligned_values(colon_pairs)

        # Classify each pair
        for idx, pair in enumerate(colon_pairs):
            spacing = pair['spacing']
            deviation = abs(spacing - dominant_spacing)
            value_text = pair['value']
            is_same_cell = pair.get('is_same_cell', False)

            # Analyze content type for suspiciousness
            content_analysis = analyze_value_content_type(value_text)
            content_type = content_analysis['content_type']
            suspicion_multiplier = content_analysis['suspicion_multiplier']
            content_details = content_analysis['details']

            # Determine classification
            # For same-cell patterns, alignment is more important than spacing
            if is_same_cell:
                # Check if the entire cell (full_bbox) aligns to a left baseline
                full_bbox = pair.get('full_bbox', pair['value_bbox'])
                cell_x0 = full_bbox['x0']

                # Check alignment to left baselines
                is_aligned_to_baseline = False
                if page_baselines:
                    left_baselines = [b['value'] for b in page_baselines if b.get('orientation') == 'left' and b.get('count', 0) >= 3]
                    for baseline_x in left_baselines:
                        if abs(cell_x0 - baseline_x) <= COLON_RIGHT_ALIGN_TOLERANCE:
                            is_aligned_to_baseline = True
                            break

                if is_aligned_to_baseline:
                    # Aligned to baseline - consistent (green)
                    classification = 'consistent'
                    reason = f"Single-cell text aligned to left baseline (x0: {cell_x0:.1f}pt)"
                else:
                    # Not aligned to any baseline - suspicious (red)
                    classification = 'deviation'
                    if suspicion_multiplier > 1.0:
                        suspicion_level = "⚠️ HIGHLY SUSPICIOUS" if suspicion_multiplier >= 2.5 else "⚠️ SUSPICIOUS"
                        reason = f"{suspicion_level}: {content_details} not aligned to any baseline! Cell x0: {cell_x0:.1f}pt"
                    else:
                        reason = f"Single-cell text not aligned to baseline (x0: {cell_x0:.1f}pt)"
            else:
                # Multi-cell patterns - use spacing analysis
                if idx in left_aligned_indices:
                    # Left-aligned to baseline (green with different reason)
                    classification = 'consistent'
                    reason = f"Left-aligned to baseline (spacing: {spacing:.1f}pt, pattern: {dominant_spacing:.1f}pt)"
                elif idx in right_aligned_indices:
                    # Right-aligned to baseline (orange)
                    classification = 'right_aligned'
                    reason = f"Right-aligned to baseline (spacing: {spacing:.1f}pt, pattern: {dominant_spacing:.1f}pt)"
                elif deviation <= COLON_SPACING_TOLERANCE:
                    # Consistent with pattern (green)
                    classification = 'consistent'
                    reason = f"Consistent spacing (spacing: {spacing:.1f}pt, pattern: {dominant_spacing:.1f}pt)"
                else:
                    # Deviation from pattern (red - potential forgery)
                    classification = 'deviation'

                    # Add content-specific warning if suspicious content type
                    if suspicion_multiplier > 1.0:
                        suspicion_level = "⚠️ HIGHLY SUSPICIOUS" if suspicion_multiplier >= 2.5 else "⚠️ SUSPICIOUS"
                        reason = f"{suspicion_level}: {content_details} misaligned! Spacing deviation: {deviation:.1f}pt (spacing: {spacing:.1f}pt, pattern: {dominant_spacing:.1f}pt)"
                    else:
                        reason = f"Spacing deviation (spacing: {spacing:.1f}pt, pattern: {dominant_spacing:.1f}pt, deviation: {deviation:.1f}pt)"

            highlight_bbox = pair.get('highlight_bbox')
            if highlight_bbox and classification not in ('consistent', 'right_aligned'):
                bbox_for_display = highlight_bbox
            else:
                bbox_for_display = pair['value_bbox']

            anomaly_record = {
                'page': page_no,
                'label': pair['label'],
                'value': pair['value'],
                'bbox': bbox_for_display,
                'label_bbox': pair['label_bbox'],
                'classification': classification,
                'spacing': spacing,
                'dominant_spacing': dominant_spacing,
                'deviation': deviation,
                'reason': reason,
                'content_type': content_type,
                'suspicion_multiplier': suspicion_multiplier,
                'is_same_cell': is_same_cell,
                'font_info': pair.get('font_info')
            }
            all_anomalies.append(anomaly_record)

            # Record highlighted items (deviations and right-aligned) in the report
            if reporter and classification in ('deviation', 'right_aligned'):
                reporter.record_highlighted_item(
                    page_no=page_no,
                    item_type="colon_spacing",
                    item_data={
                        "label": pair['label'],
                        "value": pair['value'],
                        "bbox": bbox_for_display,
                        "classification": classification,
                        "spacing": spacing,
                        "dominant_spacing": dominant_spacing,
                        "deviation": deviation,
                        "reason": reason,
                        "content_type": content_type,
                        "suspicion_multiplier": suspicion_multiplier,
                        "is_same_cell": is_same_cell,
                        "font_info": pair.get('font_info')
                    }
                )

            page_report["pairs"].append({
                "label": pair['label'],
                "value": pair['value'],
                "classification": classification,
                "spacing": spacing,
                "deviation": deviation,
                "reason": reason,
                "content_type": content_type,
                "is_same_cell": is_same_cell
            })

        if reporter:
            reporter.record_colon_analysis(page_no, page_report)

    return all_anomalies
