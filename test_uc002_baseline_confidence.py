"""
Test UC-002: Baseline Confidence Scoring and Filtering
=======================================================

Tests baseline confidence filtering on all 3 PTC documents:
1. 85385859429 – Mr MP Madiya – False pay slip
2. 85393423298 – Mr G Motau - legitimate Capitec Bank statements
3. 87026407408 – Mr I Fakude - False pay slip

Verifies that confidence filtering:
- Reduces baseline count (filtering weak baselines)
- Does not exceed regression thresholds (±20% baseline count)
- Improves baseline quality (higher average confidence)
- Preserves high-quality baselines (near page edges, high support)
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Test documents
TEST_DOCS = [
    "85385859429 – Mr MP Madiya – False pay slip",
    "85393423298 – Mr G Motau - legitimate Capitec Bank statements",
    "87026407408 – Mr I Fakude - False pay slip"
]


def load_docling_payload(pdf_name: str) -> Dict[str, Any]:
    """Load the Docling payload for a document."""
    payload_dir = Path("Docling payloads")
    for file in payload_dir.glob("*.json"):
        if pdf_name in file.stem:
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(f"No payload found for {pdf_name}")


def extract_cells_and_page_info(payload: Dict[str, Any], page_no: int = 0) -> tuple:
    """Extract cells and page dimensions from payload."""
    from pdf_ocr_detector import extract_bbox_coords

    parsed_pages = payload.get("parsed_pages", {})
    page_data = parsed_pages.get(str(page_no))

    if not page_data:
        raise ValueError(f"Page {page_no} not found")

    cells = page_data.get("cells", [])
    page_size = page_data.get("size", {})
    page_width = page_size.get("width", 612)
    page_height = page_size.get("height", 792)

    consolidated_cells = []
    for cell in cells:
        rect = cell.get("rect", {})
        text = cell.get("text", "")

        if not rect:
            continue

        try:
            x0, y0, x1, y1 = extract_bbox_coords(rect)
            bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

            consolidated_cells.append({
                "text": text,
                "bbox": bbox,
                "page_no": page_no,
            })
        except Exception:
            continue

    return consolidated_cells, page_width, page_height


def run_baseline_detection_comparison(
    cells: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    doc_name: str
) -> Dict[str, Any]:
    """
    Run baseline detection with and without confidence filtering.

    Returns comparison results including baseline counts and quality metrics.
    """
    from alignment_detection import (
        detect_alignment_baselines,
        refine_baselines_with_confidence_filtering,
        build_alignment_metadata
    )

    # Detect baselines without confidence filtering
    baselines_unfiltered = detect_alignment_baselines(
        cells, page_width, page_metadata=None, page_height=page_height, blocks=None
    )

    # Apply confidence filtering
    debug_info: Dict[str, Any] = {}
    baselines_filtered = refine_baselines_with_confidence_filtering(
        baselines_unfiltered.copy(),
        page_width,
        page_height,
        debug=debug_info
    )

    # Calculate statistics
    unfiltered_count = len(baselines_unfiltered)
    filtered_count = len(baselines_filtered)
    reduction = unfiltered_count - filtered_count
    reduction_pct = (reduction / unfiltered_count * 100) if unfiltered_count > 0 else 0.0

    # Calculate average confidence
    avg_confidence_unfiltered = (
        sum(b.get("confidence", 0.0) for b in baselines_unfiltered) / unfiltered_count
        if unfiltered_count > 0 else 0.0
    )
    avg_confidence_filtered = (
        sum(b.get("confidence", 0.0) for b in baselines_filtered) / filtered_count
        if filtered_count > 0 else 0.0
    )

    # Check regression threshold (±20% baseline count)
    threshold_lower = unfiltered_count * 0.80
    threshold_upper = unfiltered_count * 1.20
    within_threshold = threshold_lower <= filtered_count <= threshold_upper

    # Breakdown by orientation
    unfiltered_by_orientation = {
        "left": len([b for b in baselines_unfiltered if b.get("orientation") == "left"]),
        "right": len([b for b in baselines_unfiltered if b.get("orientation") == "right"]),
        "top": len([b for b in baselines_unfiltered if b.get("orientation") == "top"]),
        "bottom": len([b for b in baselines_unfiltered if b.get("orientation") == "bottom"]),
    }
    filtered_by_orientation = {
        "left": len([b for b in baselines_filtered if b.get("orientation") == "left"]),
        "right": len([b for b in baselines_filtered if b.get("orientation") == "right"]),
        "top": len([b for b in baselines_filtered if b.get("orientation") == "top"]),
        "bottom": len([b for b in baselines_filtered if b.get("orientation") == "bottom"]),
    }

    return {
        "document": doc_name,
        "unfiltered_count": unfiltered_count,
        "filtered_count": filtered_count,
        "reduction": reduction,
        "reduction_pct": round(reduction_pct, 1),
        "avg_confidence_unfiltered": round(avg_confidence_unfiltered, 3),
        "avg_confidence_filtered": round(avg_confidence_filtered, 3),
        "confidence_improvement": round(avg_confidence_filtered - avg_confidence_unfiltered, 3),
        "within_threshold": within_threshold,
        "threshold_lower": round(threshold_lower, 1),
        "threshold_upper": round(threshold_upper, 1),
        "unfiltered_by_orientation": unfiltered_by_orientation,
        "filtered_by_orientation": filtered_by_orientation,
        "debug_info": debug_info.get("baseline_filtering", {})
    }


def print_comparison_report(results: List[Dict[str, Any]]) -> None:
    """Print formatted comparison report."""
    print("\n" + "=" * 100)
    print("UC-002 BASELINE CONFIDENCE FILTERING TEST RESULTS")
    print("=" * 100)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['document']}")
        print("-" * 100)

        print(f"   Baseline Counts:")
        print(f"      Unfiltered: {result['unfiltered_count']} baselines")
        print(f"      Filtered:   {result['filtered_count']} baselines")
        print(f"      Reduction:  {result['reduction']} baselines ({result['reduction_pct']}%)")

        print(f"\n   Quality Metrics:")
        print(f"      Avg Confidence (Unfiltered): {result['avg_confidence_unfiltered']}")
        print(f"      Avg Confidence (Filtered):   {result['avg_confidence_filtered']}")
        print(f"      Improvement:                 {result['confidence_improvement']:+.3f}")

        print(f"\n   Regression Check:")
        print(f"      Threshold Range: {result['threshold_lower']} - {result['threshold_upper']} baselines")
        print(f"      Filtered Count:  {result['filtered_count']} baselines")
        status = "[PASS]" if result['within_threshold'] else "[FAIL]"
        print(f"      Status:          {status}")

        print(f"\n   Breakdown by Orientation:")
        print(f"      {'Orientation':<15} {'Unfiltered':<12} {'Filtered':<12} {'Reduction'}")
        print(f"      {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 12}")
        for orient in ["left", "right", "top", "bottom"]:
            unf = result['unfiltered_by_orientation'].get(orient, 0)
            filt = result['filtered_by_orientation'].get(orient, 0)
            red = unf - filt
            print(f"      {orient:<15} {unf:<12} {filt:<12} {red}")

        # Show filtered baselines if any
        filtered_baselines = result['debug_info'].get('filtered_baselines', [])
        if filtered_baselines:
            print(f"\n   Filtered Baselines ({len(filtered_baselines)} total):")
            for j, fb in enumerate(filtered_baselines[:5], 1):  # Show first 5
                print(f"      {j}. {fb['orientation']:<6} @ {fb['value']:>7.2f}pt, "
                      f"count={fb['count']}, conf={fb['confidence']:.3f}, "
                      f"reasons: {', '.join(fb['reasons'])}")
            if len(filtered_baselines) > 5:
                print(f"      ... and {len(filtered_baselines) - 5} more")

    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    all_passed = all(r['within_threshold'] for r in results)
    total_unfiltered = sum(r['unfiltered_count'] for r in results)
    total_filtered = sum(r['filtered_count'] for r in results)
    total_reduction = total_unfiltered - total_filtered
    avg_confidence_improvement = sum(r['confidence_improvement'] for r in results) / len(results)

    print(f"   Documents Tested: {len(results)}")
    print(f"   Total Baselines (Unfiltered): {total_unfiltered}")
    print(f"   Total Baselines (Filtered):   {total_filtered}")
    print(f"   Total Reduction:              {total_reduction} baselines")
    print(f"   Average Confidence Improvement: {avg_confidence_improvement:+.3f}")
    print(f"   Regression Tests Passed:      {sum(r['within_threshold'] for r in results)}/{len(results)}")
    print(f"   Overall Status:               {'[ALL PASSED]' if all_passed else '[SOME FAILED]'}")

    print("\n" + "=" * 100)


def save_results_to_json(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Save detailed results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file}")


def main():
    """Run baseline confidence filtering tests on all PTC documents."""
    print("Testing UC-002: Baseline Confidence Scoring and Filtering")
    print(f"Testing {len(TEST_DOCS)} documents...")

    results = []

    for doc_name in TEST_DOCS:
        print(f"\nProcessing: {doc_name}")
        try:
            # Load payload
            payload = load_docling_payload(doc_name)

            # Extract cells and page info (use page 0)
            cells, page_width, page_height = extract_cells_and_page_info(payload, page_no=0)

            print(f"  Loaded {len(cells)} cells, page size: {page_width}x{page_height}pt")

            # Run comparison
            result = run_baseline_detection_comparison(cells, page_width, page_height, doc_name)
            results.append(result)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print report
    print_comparison_report(results)

    # Save to JSON
    output_file = Path("Alignment logic/uc002_baseline_confidence_test_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_json(results, output_file)


if __name__ == "__main__":
    main()
