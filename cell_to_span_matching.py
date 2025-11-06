"""
Cell-to-Span Matching Pipeline
===============================

Directly matches Docling cells (lines from parsed_pages) to PyMuPDF spans.

Key Insight:
- Docling parsed_pages.cells = line-level elements (e.g., "Customer Care: 0860 123 000")
- PyMuPDF spans = line/phrase-level elements (same granularity!)
- Both extracted from PyMuPDF backend = same source!

Expected: 99%+ match rate (same granularity, same source)

Approach:
1. Extract cells from parsed_pages (line-level)
2. Extract spans from PyMuPDF (line-level)
3. Match cells → spans directly (text similarity + bbox overlap)
4. No word grouping needed (same granularity!)

Advantages over word-level matching:
- ✅ Same granularity (line/span level)
- ✅ Exact bboxes from PDF (not estimated)
- ✅ Font metadata available in both
- ✅ Color metadata available in both
- ✅ Simpler matching logic (no tokenization)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import Levenshtein
from pathlib import Path
import fitz  # PyMuPDF

# Import classes from cell_data_classes to avoid duplicates
from cell_data_classes import DoclingCell, PyMuPDFSpan, CellToSpanMatch


def calculate_iou(bbox1: Tuple[float, float, float, float],
                  bbox2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        bbox1: (x0, y0, x1, y1)
        bbox2: (x0, y0, x1, y1)

    Returns:
        IoU score (0.0 - 1.0)
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    # Calculate intersection
    x0_i = max(x0_1, x0_2)
    y0_i = max(y0_1, y0_2)
    x1_i = min(x1_1, x1_2)
    y1_i = min(y1_1, y1_2)

    if x1_i <= x0_i or y1_i <= y0_i:
        return 0.0  # No intersection

    intersection_area = (x1_i - x0_i) * (y1_i - y0_i)

    # Calculate union
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def levenshtein_ratio(text1: str, text2: str) -> float:
    """
    Calculate normalized Levenshtein similarity ratio.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity ratio (0.0 - 1.0)
    """
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    # Normalize whitespace
    text1_norm = " ".join(text1.split())
    text2_norm = " ".join(text2.split())

    distance = Levenshtein.distance(text1_norm, text2_norm)
    max_len = max(len(text1_norm), len(text2_norm))

    if max_len == 0:
        return 1.0

    return 1.0 - (distance / max_len)


def convert_rect_to_bbox(rect: Dict[str, Any], page_height: float = None) -> Tuple[float, float, float, float]:
    """
    Convert Docling rect to bbox tuple in BOTTOMLEFT coords.

    Docling rect has 4 corners:
    - (r_x0, r_y0): top-left in TOPLEFT coords
    - (r_x1, r_y1): top-right in TOPLEFT coords
    - (r_x2, r_y2): bottom-right in TOPLEFT coords
    - (r_x3, r_y3): bottom-left in TOPLEFT coords

    Args:
        rect: Docling rect with r_x0, r_y0, etc.
        page_height: Page height for coordinate conversion (if None, stays in TOPLEFT)

    Returns:
        (x0, y0, x1, y1) tuple in BOTTOMLEFT coords
    """
    # Get all corners
    x0 = min(rect.get('r_x0', 0.0), rect.get('r_x3', 0.0))
    x1 = max(rect.get('r_x1', 0.0), rect.get('r_x2', 0.0))

    # Get min and max y values from all 4 corners
    all_y_values = [
        rect.get('r_y0', 0.0),
        rect.get('r_y1', 0.0),
        rect.get('r_y2', 0.0),
        rect.get('r_y3', 0.0)
    ]
    y_min = min(all_y_values)  # Visual top (higher on page in TOPLEFT)
    y_max = max(all_y_values)  # Visual bottom (lower on page in TOPLEFT)

    # If page_height provided, convert TOPLEFT to BOTTOMLEFT
    if page_height is not None:
        # In BOTTOMLEFT coords: origin is at bottom-left of page
        # Visual top → larger y value in BOTTOMLEFT
        # Visual bottom → smaller y value in BOTTOMLEFT
        y0_bottomleft = page_height - y_max  # Visual bottom → smaller y
        y1_bottomleft = page_height - y_min  # Visual top → larger y
        return (x0, y0_bottomleft, x1, y1_bottomleft)

    # Otherwise return in TOPLEFT coords (for IoU calculation, coords don't matter as long as consistent)
    return (x0, y_min, x1, y_max)


def extract_cells_from_enhanced_payload(
    payload: Dict[str, Any],
    page_num: int
) -> List[DoclingCell]:
    """
    Extract Docling cells from enhanced payload.

    Args:
        payload: Enhanced Docling payload with parsed_pages
        page_num: Page number (0-indexed)

    Returns:
        List of DoclingCell objects
    """
    cells = []

    if 'parsed_pages' not in payload:
        raise ValueError("Payload missing 'parsed_pages' (not an enhanced payload?)")

    page_key = str(page_num)
    if page_key not in payload['parsed_pages']:
        raise ValueError(f"Page {page_num} not found in parsed_pages")

    page_data = payload['parsed_pages'][page_key]
    cells_data = page_data.get('cells', [])

    # Get page height for coordinate conversion
    page_size = page_data.get('size', {})
    page_height = page_size.get('height', None)

    for cell_data in cells_data:
        cell = DoclingCell(
            text=cell_data.get('text', ''),
            bbox=convert_rect_to_bbox(cell_data.get('rect', {}), page_height),
            font_name=cell_data.get('font_name'),
            font_key=cell_data.get('font_key'),
            rgba=cell_data.get('rgba'),
            confidence=cell_data.get('confidence'),
            from_ocr=cell_data.get('from_ocr'),
            index=cell_data.get('index')
        )
        cells.append(cell)

    return cells


def extract_pymupdf_spans(pdf_path: Path, page_num: int) -> List[PyMuPDFSpan]:
    """
    Extract PyMuPDF spans from PDF.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)

    Returns:
        List of PyMuPDFSpan objects
    """
    doc = fitz.open(pdf_path)

    if page_num >= len(doc):
        doc.close()
        raise ValueError(f"Page {page_num} out of range (document has {len(doc)} pages)")

    page = doc[page_num]
    page_dict = page.get_text("dict")

    spans = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # Skip non-text blocks
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                bbox_list = span.get("bbox", [0, 0, 0, 0])

                # Convert bbox from TOPLEFT to BOTTOMLEFT coords
                x0, y0_top, x1, y1_top = bbox_list
                page_height = page.rect.height
                y0_bottom = page_height - y1_top
                y1_bottom = page_height - y0_top

                bbox = (x0, y0_bottom, x1, y1_bottom)

                # Extract font and color
                font_name = span.get("font", None)
                font_size = span.get("size", None)
                color = span.get("color", None)

                # Convert color from int to RGB tuple
                color_rgb = None
                if color is not None:
                    r = (color >> 16) & 0xFF
                    g = (color >> 8) & 0xFF
                    b = color & 0xFF
                    color_rgb = (r, g, b)

                flags = span.get("flags", None)

                span_obj = PyMuPDFSpan(
                    text=text,
                    bbox=bbox,
                    font_name=font_name,
                    font_size=font_size,
                    color_rgb=color_rgb,
                    flags=flags
                )
                spans.append(span_obj)

    doc.close()
    return spans


def match_cell_to_span(
    cell: DoclingCell,
    span: PyMuPDFSpan,
    min_text_similarity: float = 0.85,
    min_bbox_iou: float = 0.3
) -> Optional[CellToSpanMatch]:
    """
    Try to match a Docling cell to a PyMuPDF span.

    Args:
        cell: Docling cell
        span: PyMuPDF span
        min_text_similarity: Minimum Levenshtein ratio (0.85)
        min_bbox_iou: Minimum bbox IoU (0.3)

    Returns:
        CellToSpanMatch if match found, None otherwise
    """
    # Calculate text similarity
    text_sim = levenshtein_ratio(cell.text, span.text)

    # Calculate bbox IoU
    bbox_iou = calculate_iou(cell.bbox, span.bbox)

    # Determine match method
    match_method = None

    # Method 1: High text similarity + reasonable spatial overlap
    if text_sim >= min_text_similarity and bbox_iou >= min_bbox_iou:
        match_method = "text_based"

    # Method 2: High spatial overlap + relaxed text (for OCR differences)
    elif bbox_iou >= 0.7 and text_sim >= 0.75:
        match_method = "spatial_based"

    # Method 3: Very high spatial overlap (complete containment)
    elif bbox_iou >= 0.9:
        match_method = "combined"

    if match_method:
        return CellToSpanMatch(
            docling_cell=cell,
            pymupdf_span=span,
            text_similarity=text_sim,
            bbox_iou=bbox_iou,
            match_method=match_method
        )

    return None


def match_cells_to_spans(
    cells: List[DoclingCell],
    spans: List[PyMuPDFSpan],
    min_text_similarity: float = 0.85,
    min_bbox_iou: float = 0.3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Match Docling cells to PyMuPDF spans.

    Args:
        cells: List of Docling cells
        spans: List of PyMuPDF spans
        min_text_similarity: Minimum text similarity (0.85)
        min_bbox_iou: Minimum bbox IoU (0.3)
        verbose: Print detailed matching info

    Returns:
        Dictionary with matches, unmatched cells, unmatched spans, and statistics
    """
    matches: List[CellToSpanMatch] = []
    matched_cell_indices = set()
    matched_span_indices = set()

    # Try to match each cell to best span
    for cell_idx, cell in enumerate(cells):
        best_match = None
        best_score = 0.0
        best_span_idx = None

        for span_idx, span in enumerate(spans):
            if span_idx in matched_span_indices:
                continue  # Skip already matched spans

            match = match_cell_to_span(cell, span, min_text_similarity, min_bbox_iou)

            if match:
                # Combined score: text similarity + bbox IoU
                score = match.text_similarity * 0.6 + match.bbox_iou * 0.4

                if score > best_score:
                    best_match = match
                    best_score = score
                    best_span_idx = span_idx

        if best_match:
            matches.append(best_match)
            matched_cell_indices.add(cell_idx)
            matched_span_indices.add(best_span_idx)

            if verbose:
                print(f"  Match: '{best_match.docling_cell.text[:50]}' -> '{best_match.pymupdf_span.text[:50]}'")
                print(f"         Text: {best_match.text_similarity:.2f}, IoU: {best_match.bbox_iou:.2f}, Method: {best_match.match_method}")

    # Collect unmatched
    unmatched_cells = [cell for idx, cell in enumerate(cells) if idx not in matched_cell_indices]
    unmatched_spans = [span for idx, span in enumerate(spans) if idx not in matched_span_indices]

    # Statistics
    stats = {
        "total_cells": len(cells),
        "total_spans": len(spans),
        "matched_cells": len(matched_cell_indices),
        "matched_spans": len(matched_span_indices),
        "unmatched_cells": len(unmatched_cells),
        "unmatched_spans": len(unmatched_spans),
        "cell_match_rate": len(matched_cell_indices) / len(cells) * 100 if cells else 0,
        "span_match_rate": len(matched_span_indices) / len(spans) * 100 if spans else 0,
        "match_methods": {}
    }

    # Count match methods
    for match in matches:
        method = match.match_method
        stats["match_methods"][method] = stats["match_methods"].get(method, 0) + 1

    return {
        "matches": matches,
        "unmatched_cells": unmatched_cells,
        "unmatched_spans": unmatched_spans,
        "statistics": stats
    }


def cell_to_span_matching_pipeline(
    payload: Dict[str, Any],
    pdf_path: Path,
    page_num: int,
    min_text_similarity: float = 0.85,
    min_bbox_iou: float = 0.3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Complete cell-to-span matching pipeline.

    Args:
        payload: Enhanced Docling payload with parsed_pages
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        min_text_similarity: Minimum text similarity (0.85)
        min_bbox_iou: Minimum bbox IoU (0.3)
        verbose: Print detailed info

    Returns:
        Dictionary with matches and statistics
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"CELL-TO-SPAN MATCHING PIPELINE")
        print(f"{'='*80}")
        print(f"Document: {pdf_path.name}")
        print(f"Page: {page_num}")

    # Extract cells from enhanced payload
    if verbose:
        print(f"\n[1/3] Extracting Docling cells from parsed_pages...")

    cells = extract_cells_from_enhanced_payload(payload, page_num)

    if verbose:
        print(f"      OK Extracted {len(cells)} cells")

    # Extract spans from PyMuPDF
    if verbose:
        print(f"\n[2/3] Extracting PyMuPDF spans...")

    spans = extract_pymupdf_spans(pdf_path, page_num)

    if verbose:
        print(f"      OK Extracted {len(spans)} spans")

    # Match cells to spans
    if verbose:
        print(f"\n[3/3] Matching cells to spans...")
        print(f"      Thresholds: text_sim={min_text_similarity}, bbox_iou={min_bbox_iou}")

    result = match_cells_to_spans(cells, spans, min_text_similarity, min_bbox_iou, verbose)

    if verbose:
        stats = result['statistics']
        print(f"\n{'='*80}")
        print(f"MATCHING RESULTS")
        print(f"{'='*80}")
        print(f"Cells:            {stats['total_cells']}")
        print(f"Spans:            {stats['total_spans']}")
        print(f"Matched Cells:    {stats['matched_cells']} ({stats['cell_match_rate']:.1f}%)")
        print(f"Matched Spans:    {stats['matched_spans']} ({stats['span_match_rate']:.1f}%)")
        print(f"Unmatched Cells:  {stats['unmatched_cells']}")
        print(f"Unmatched Spans:  {stats['unmatched_spans']}")
        print(f"\nMatch Methods:")
        for method, count in stats['match_methods'].items():
            print(f"  {method:20s} {count}")
        print(f"{'='*80}\n")

    return result
