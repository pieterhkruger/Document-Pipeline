"""
Two-Stage Clustering Smoke Test for Text Block Detection
==========================================================

Tests different combinations of vertical and horizontal clustering methods:
- DBSCAN + DBSCAN
- DBSCAN + Agglomerative
- Agglomerative + DBSCAN
- Agglomerative + Agglomerative

Generates JSON outputs for each combination to compare against expected results.

Usage:
    python two_stage_smoke_test.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List

# Test parameters
TEST_DOC = "87026407408 – Mr I Fakude - False pay slip"

# Test combinations
TEST_COMBINATIONS = [
    # (vert_method, horiz_method, vert_eps, horiz_eps, min_samples)
    ("dbscan", "dbscan", 10.0, 30.0, 2),
    ("dbscan", "dbscan", 10.0, 50.0, 2),
    ("dbscan", "dbscan", 10.0, 70.0, 2),
    ("dbscan", "dbscan", 12.0, 30.0, 2),
    ("dbscan", "dbscan", 12.0, 50.0, 2),
    ("dbscan", "dbscan", 12.0, 70.0, 2),
    ("dbscan", "agglomerative", 10.0, 30.0, 2),
    ("dbscan", "agglomerative", 10.0, 50.0, 2),
    ("dbscan", "agglomerative", 10.0, 70.0, 2),
    ("dbscan", "agglomerative", 12.0, 30.0, 2),
    ("dbscan", "agglomerative", 12.0, 50.0, 2),
    ("dbscan", "agglomerative", 12.0, 70.0, 2),
    ("agglomerative", "dbscan", 10.0, 30.0, 2),
    ("agglomerative", "dbscan", 10.0, 50.0, 2),
    ("agglomerative", "dbscan", 10.0, 70.0, 2),
    ("agglomerative", "dbscan", 12.0, 30.0, 2),
    ("agglomerative", "dbscan", 12.0, 50.0, 2),
    ("agglomerative", "dbscan", 12.0, 70.0, 2),
    ("agglomerative", "agglomerative", 10.0, 30.0, 2),
    ("agglomerative", "agglomerative", 10.0, 50.0, 2),
    ("agglomerative", "agglomerative", 10.0, 70.0, 2),
    ("agglomerative", "agglomerative", 12.0, 30.0, 2),
    ("agglomerative", "agglomerative", 12.0, 50.0, 2),
    ("agglomerative", "agglomerative", 12.0, 70.0, 2),
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
    """Run smoke test with different two-stage clustering parameters."""
    from text_block_detection import identify_text_blocks_two_stage

    print(f"Loading document: {TEST_DOC}")
    payload = load_docling_payload(TEST_DOC)
    cells, baselines = extract_cells_and_baselines(payload, page_no=0)

    print(f"Extracted {len(cells)} cells and {len(baselines)} baselines\n")

    output_dir = Path("Alignment logic/TWO_STAGE_tests")
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
                "blocks_summary": [
                    {
                        "block_id": block["block_id"],
                        "bbox": block["bbox"],
                        "line_count": block["line_count"],
                        "dominant_alignment": block["dominant_alignment"],
                    }
                    for block in blocks
                ]
            }

            # Save to file
            output_file = output_dir / f"{vert_method}_{horiz_method}_v{vert_eps}_h{horiz_eps}_m{min_samples}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Print summary
            vert_clusters = debug.get("inputs", {}).get("vertical_clusters_found", 0)
            final_blocks = len(blocks)

            print(f"  Vert clusters: {vert_clusters}, Final blocks: {final_blocks}")

            # Show some block texts
            if debug.get("vertical_clusters"):
                print("  Sample blocks:")
                for i, cluster in enumerate(debug["vertical_clusters"][:3], 1):
                    texts = cluster.get("texts", [])
                    print(f"    {i}. {cluster['line_count']} lines: {', '.join(texts[:5])}")

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
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"{'Vert Method':<15} {'Horiz Method':<15} {'v_eps':<8} {'h_eps':<8} {'Vert Clust':<12} {'Blocks':<8}")
    print("-"*100)
    for r in results:
        if "error" not in r:
            print(f"{r['vert_method']:<15} {r['horiz_method']:<15} {r['vert_eps']:<8} {r['horiz_eps']:<8} {r['vert_clusters']:<12} {r['blocks']:<8}")
        else:
            print(f"{r['vert_method']:<15} {r['horiz_method']:<15} {r['vert_eps']:<8} {r['horiz_eps']:<8} ERROR: {r['error']}")
    print("="*100)
    print(f"\nResults saved to: {output_dir}/")
    print("\nRecommended: Look for combinations that produce 5-12 blocks")
    print("with natural separation between left/middle/right columns")


if __name__ == "__main__":
    run_smoke_test()
