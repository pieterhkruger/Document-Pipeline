"""
Phase 1 Smoke Test: DBSCAN+DBSCAN with Reduced h_eps
======================================================

Tests DBSCAN+DBSCAN with v_eps=12.0 and h_eps=[35.0, 40.0, 45.0]
to find optimal horizontal separation threshold.

Usage:
    python phase1_smoke_test.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List

# Test parameters
TEST_DOC = "87026407408 – Mr I Fakude - False pay slip"

# Test combinations - focus on h_eps tuning with DBSCAN+DBSCAN
TEST_COMBINATIONS = [
    # (vert_method, horiz_method, vert_eps, horiz_eps, min_samples)
    ("dbscan", "dbscan", 12.0, 35.0, 2),
    ("dbscan", "dbscan", 12.0, 40.0, 2),
    ("dbscan", "dbscan", 12.0, 45.0, 2),
]

def load_docling_payload(pdf_name: str) -> Dict[str, Any]:
    """Load the Docling payload for a document."""
    payload_dir = Path("Docling payloads")
    # Find matching payload file
    for file in payload_dir.glob("*.json"):
        if pdf_name in file.stem:
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(f"No payload found for {pdf_name}")


def extract_cells_and_baselines(payload: Dict[str, Any], page_no: int = 0):
    """Extract cells and baselines from payload."""
    from pdf_ocr_detector import extract_bbox_coords
    from alignment_detection import detect_alignment_baselines

    # Get page data
    parsed_pages = payload.get("parsed_pages", {})
    page_data = parsed_pages.get(str(page_no))

    if not page_data:
        raise ValueError(f"Page {page_no} not found")

    # Extract cells
    cells = page_data.get("cells", [])
    consolidated_cells = []

    for cell in cells:
        rect = cell.get("rect", {})
        text = cell.get("text", "")

        if not rect:
            continue

        # Convert to bbox
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

    # Detect baselines
    baselines_result = detect_alignment_baselines(consolidated_cells, page_no, {})
    if isinstance(baselines_result, tuple):
        baselines = baselines_result[0]  # Get first element if tuple
    else:
        baselines = baselines_result

    return consolidated_cells, baselines


def run_smoke_test():
    """Run smoke test with reduced h_eps values."""
    from text_block_detection import identify_text_blocks_two_stage

    print(f"Loading document: {TEST_DOC}")
    payload = load_docling_payload(TEST_DOC)
    cells, baselines = extract_cells_and_baselines(payload, page_no=0)

    print(f"Extracted {len(cells)} cells and {len(baselines)} baselines\n")

    output_dir = Path("Alignment logic/PHASE1_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for vert_method, horiz_method, vert_eps, horiz_eps, min_samples in TEST_COMBINATIONS:
        print(f"Testing {vert_method}+{horiz_method}: v_eps={vert_eps}, h_eps={horiz_eps}, min={min_samples}...")

        debug = {}
        try:
            blocks = identify_text_blocks_two_stage(
                cells, baselines,
                vertical_method=vert_method,
                horizontal_method=horiz_method,
                vertical_eps=vert_eps,
                horizontal_eps=horiz_eps,
                min_samples=min_samples,
                debug=debug
            )

            # Create output
            result = {
                "parameters": {
                    "vertical_method": vert_method,
                    "horizontal_method": horiz_method,
                    "vertical_eps": vert_eps,
                    "horizontal_eps": horiz_eps,
                    "min_samples": min_samples,
                },
                "debug": debug,
                "blocks": blocks  # Include full block details
            }

            # Save to file
            output_file = output_dir / f"dbscan_dbscan_v{vert_eps}_h{horiz_eps}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Print summary
            vert_clusters = debug.get("inputs", {}).get("vertical_clusters_found", 0)
            final_blocks = len(blocks)

            print(f"  Vert clusters: {vert_clusters}, Final blocks: {final_blocks}")

            # Show sample blocks with their texts
            if blocks:
                print("  Sample blocks:")
                for i, block in enumerate(blocks[:5], 1):
                    texts = block.get("texts", [])
                    line_count = block.get("line_count", 0)
                    alignment = block.get("dominant_alignment", "unknown")
                    print(f"    {i}. {line_count} lines ({alignment}): {', '.join(texts[:5])}")

            print()

            results.append({
                "vert_method": vert_method,
                "horiz_method": horiz_method,
                "vert_eps": vert_eps,
                "horiz_eps": horiz_eps,
                "vert_clusters": vert_clusters,
                "blocks": final_blocks,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            results.append({
                "vert_method": vert_method,
                "horiz_method": horiz_method,
                "vert_eps": vert_eps,
                "horiz_eps": horiz_eps,
                "vert_clusters": 0,
                "blocks": 0,
                "error": str(e)
            })

    # Print summary table
    print("\n" + "="*80)
    print("PHASE 1 SMOKE TEST RESULTS")
    print("="*80)
    print(f"{'h_eps':<10} {'Vert Clusters':<15} {'Final Blocks':<15} {'Status':<20}")
    print("-"*80)
    for r in results:
        if "error" not in r:
            status = "✓ OK"
            print(f"{r['horiz_eps']:<10} {r['vert_clusters']:<15} {r['blocks']:<15} {status:<20}")
        else:
            print(f"{r['horiz_eps']:<10} {'N/A':<15} {'N/A':<15} ERROR: {r['error']}")
    print("="*80)

    print(f"\nResults saved to: {output_dir}/")
    print("\nGoal: Find h_eps that splits bottom block into 3+ column blocks")
    print("Expected: 12-15 total blocks (current: 9 with h_eps=70.0)")
    print("\nCompare with baseline (h_eps=70.0): 9 blocks")


if __name__ == "__main__":
    run_smoke_test()
