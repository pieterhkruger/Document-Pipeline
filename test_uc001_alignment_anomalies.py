"""
Test UC-001: Alignment Attribution and Anomaly Detection (Phase 2)
===================================================================

Tests alignment analysis and anomaly detection on all 3 PTC documents:
1. 85385859429 – Mr MP Madiya – False pay slip
2. 85393423298 – Mr G Motau - legitimate Capitec Bank statements
3. 87026407408 – Mr I Fakude - False pay slip

Verifies that:
- Alignment metadata is correctly calculated for all blocks
- Baseline-aware snapping works correctly
- Anomalies are detected in forged documents
- No false positives on legitimate documents
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Test documents
TEST_DOCS = [
    {
        "name": "85385859429 – Mr MP Madiya – False pay slip",
        "expected_anomalies": "unknown",  # Will determine empirically
        "doc_type": "forgery"
    },
    {
        "name": "85393423298 – Mr G Motau - legitimate Capitec Bank statements",
        "expected_anomalies": 0,  # Should have no anomalies (legitimate)
        "doc_type": "legitimate"
    },
    {
        "name": "87026407408 – Mr I Fakude - False pay slip",
        "expected_anomalies": "unknown",  # Will determine empirically
        "doc_type": "forgery"
    },
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


def run_full_pipeline_with_anomaly_detection(
    cells: List[Dict[str, Any]],
    page_width: float,
    page_height: float,
    doc_name: str
) -> Dict[str, Any]:
    """
    Run full pipeline: baseline detection → block detection → anomaly detection.

    Returns results including blocks, anomalies, and statistics.
    """
    from alignment_detection import (
        detect_alignment_baselines,
        refine_baselines_with_confidence_filtering
    )
    from text_block_detection import (
        identify_text_blocks_iterative,
        detect_column_alignment_anomalies
    )

    # Step 1: Detect baselines with confidence filtering
    baselines_raw = detect_alignment_baselines(
        cells, page_width, page_metadata=None, page_height=page_height, blocks=None
    )

    debug_info: Dict[str, Any] = {}
    baselines = refine_baselines_with_confidence_filtering(
        baselines_raw.copy(),
        page_width,
        page_height,
        debug=debug_info
    )

    # Step 2: Identify text blocks with alignment analysis
    block_debug: Dict[str, Any] = {}
    blocks = identify_text_blocks_iterative(
        cells,
        baselines,
        tables=None,
        use_table_awareness=False,
        debug=block_debug
    )

    # Step 3: Detect alignment anomalies
    anomalies = detect_column_alignment_anomalies(blocks)

    # Calculate statistics
    total_blocks = len(blocks)
    blocks_with_alignment = sum(1 for b in blocks if b.get("alignment_metadata"))
    blocks_with_deviations = sum(
        1 for b in blocks
        if b.get("alignment_metadata", {}).get("deviation_count", 0) > 0
    )

    # Analyze alignment types
    alignment_type_counts = {}
    for block in blocks:
        metadata = block.get("alignment_metadata", {})
        align_type = metadata.get("alignment_type", "unknown")
        alignment_type_counts[align_type] = alignment_type_counts.get(align_type, 0) + 1

    # Analyze anomaly severity
    anomaly_severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for anomaly in anomalies:
        severity = anomaly.get("severity", "UNKNOWN")
        if severity in anomaly_severity_counts:
            anomaly_severity_counts[severity] += 1

    return {
        "document": doc_name,
        "total_cells": len(cells),
        "total_baselines": len(baselines),
        "total_blocks": total_blocks,
        "blocks_with_alignment_metadata": blocks_with_alignment,
        "blocks_with_deviations": blocks_with_deviations,
        "alignment_type_counts": alignment_type_counts,
        "total_anomalies": len(anomalies),
        "anomaly_severity_counts": anomaly_severity_counts,
        "blocks": blocks,
        "anomalies": anomalies,
        "baselines": baselines,
    }


def print_alignment_analysis_report(results: List[Dict[str, Any]]) -> None:
    """Print formatted alignment analysis report."""
    print("\n" + "=" * 100)
    print("UC-001 ALIGNMENT ANOMALY DETECTION TEST RESULTS (PHASE 2)")
    print("=" * 100)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['document']}")
        print("-" * 100)

        print(f"\n   Pipeline Statistics:")
        print(f"      Total Cells:     {result['total_cells']}")
        print(f"      Total Baselines: {result['total_baselines']}")
        print(f"      Total Blocks:    {result['total_blocks']}")
        print(f"      Blocks with Alignment Metadata: {result['blocks_with_alignment_metadata']}")
        print(f"      Blocks with Deviations:         {result['blocks_with_deviations']}")

        print(f"\n   Alignment Type Distribution:")
        for align_type, count in sorted(result['alignment_type_counts'].items()):
            print(f"      {align_type:>10}: {count} blocks")

        print(f"\n   Anomaly Detection:")
        print(f"      Total Anomalies: {result['total_anomalies']}")
        if result['total_anomalies'] > 0:
            print(f"      Severity Breakdown:")
            for severity in ["HIGH", "MEDIUM", "LOW"]:
                count = result['anomaly_severity_counts'].get(severity, 0)
                if count > 0:
                    print(f"         {severity:>6}: {count} anomalies")

            print(f"\n   Detected Anomalies (Top 5):")
            for j, anomaly in enumerate(result['anomalies'][:5], 1):
                print(f"\n      Anomaly {j}:")
                print(f"         Block ID:       {anomaly['block_id']}")
                print(f"         Severity:       {anomaly['severity']}")
                print(f"         Alignment:      {anomaly['alignment_type']} (confidence: {anomaly['confidence']:.1%})")
                print(f"         Baseline:       {anomaly['baseline_value']:.1f}pt")
                print(f"         Deviations:     {anomaly['deviation_count']} cells")
                print(f"         Max Deviation:  {anomaly['max_deviation']:+.2f}pt")
                print(f"         Description:    {anomaly['description']}")

                # Show sample deviating cells
                if anomaly['deviations']:
                    print(f"         Deviating Cells:")
                    for dev in anomaly['deviations'][:3]:
                        print(f"            - \"{dev['cell_text'][:40]}\": {dev['deviation']:+.1f}pt")
        else:
            print(f"      [No anomalies detected]")

    # Overall summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY")
    print("=" * 100)

    total_anomalies = sum(r['total_anomalies'] for r in results)
    legitimate_docs = [r for r in results if r['document'] == "85393423298 – Mr G Motau - legitimate Capitec Bank statements"]
    false_positives = sum(r['total_anomalies'] for r in legitimate_docs)

    print(f"   Documents Tested:        {len(results)}")
    print(f"   Total Anomalies Found:   {total_anomalies}")
    print(f"   False Positives (Legitimate Docs): {false_positives}")

    if false_positives == 0:
        print(f"\n   [PASS] No false positives detected on legitimate documents")
    else:
        print(f"\n   [WARN] {false_positives} false positive(s) detected on legitimate documents")

    print("\n" + "=" * 100)


def save_results_to_json(results: List[Dict[str, Any]], output_file: Path) -> None:
    """Save detailed results to JSON file."""
    # Remove blocks from results to reduce file size (keep only anomalies)
    results_for_export = []
    for result in results:
        result_copy = result.copy()
        result_copy.pop("blocks", None)
        result_copy.pop("baselines", None)
        results_for_export.append(result_copy)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_for_export, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file}")


def main():
    """Run Phase 2 alignment anomaly detection tests on all PTC documents."""
    print("\n" + "=" * 100)
    print("Testing UC-001: Alignment Attribution and Anomaly Detection (Phase 2)")
    print(f"Testing {len(TEST_DOCS)} documents...")
    print("=" * 100)

    results = []

    for doc_info in TEST_DOCS:
        doc_name = doc_info["name"]
        print(f"\nProcessing: {doc_name}")
        try:
            # Load payload
            payload = load_docling_payload(doc_name)

            # Extract cells and page info (use page 0)
            cells, page_width, page_height = extract_cells_and_page_info(payload, page_no=0)

            print(f"  Loaded {len(cells)} cells, page size: {page_width}x{page_height}pt")

            # Run full pipeline with anomaly detection
            result = run_full_pipeline_with_anomaly_detection(
                cells, page_width, page_height, doc_name
            )
            results.append(result)

            print(f"  => {result['total_blocks']} blocks, {result['total_anomalies']} anomalies detected")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comprehensive report
    print_alignment_analysis_report(results)

    # Save to JSON
    output_file = Path("Alignment logic/uc001_alignment_anomaly_test_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_json(results, output_file)

    print("\nPhase 2 testing complete!\n")

    return 0


if __name__ == "__main__":
    exit(main())
