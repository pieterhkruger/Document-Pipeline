"""
DBSCAN/HDBSCAN Smoke Test for Text Block Detection
===========================================

Tests different clustering parameters to find optimal values for text block clustering.
Generates JSON outputs for each parameter combination to compare against expected results.

Compares:
- DBSCAN (with different eps/min_samples)
- HDBSCAN (hierarchical, auto-tunes epsilon)

Usage:
    python dbscan_smoke_test.py
"""

import json
from pathlib import Path
from typing import Dict, Any, List

# Test parameters
TEST_EPS_VALUES = [10.0, 12.0, 15.0]
TEST_MIN_SAMPLES = [3]
TEST_HORIZONTAL_SCALES = [0.2, 0.3, 0.4, 0.5]  # NEW: Test different horizontal scaling
TEST_HDBSCAN_MIN_CLUSTER_SIZE = [2, 3, 4, 5]
TEST_DOC = "87026407408 – Mr I Fakude - False pay slip"

# Check HDBSCAN availability
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

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
    """Run smoke test with different DBSCAN parameters."""
    from text_block_detection import identify_text_blocks_dbscan

    print(f"Loading document: {TEST_DOC}")
    payload = load_docling_payload(TEST_DOC)
    cells, baselines = extract_cells_and_baselines(payload, page_no=0)

    print(f"Extracted {len(cells)} cells and {len(baselines)} baselines\n")

    output_dir = Path("Alignment logic/DBSCAN_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for eps in TEST_EPS_VALUES:
        for min_samples in TEST_MIN_SAMPLES:
            for h_scale in TEST_HORIZONTAL_SCALES:
                print(f"Testing eps={eps}, min_samples={min_samples}, h_scale={h_scale}...")

                debug = {}
                blocks = identify_text_blocks_dbscan(
                    cells, baselines, eps=eps, min_samples=min_samples,
                    horizontal_scale=h_scale, debug=debug
                )

                # Create output
                result = {
                    "parameters": {
                        "eps": eps,
                        "min_samples": min_samples,
                        "horizontal_scale": h_scale
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
                output_file = output_dir / f"eps_{eps}_min_{min_samples}_hscale_{h_scale}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                # Print summary
                clusters_found = debug.get("inputs", {}).get("clusters_found", 0)
                noise_points = debug.get("inputs", {}).get("noise_points", 0)
                final_blocks = len(blocks)

                print(f"  Clusters: {clusters_found}, Noise: {noise_points}, Blocks: {final_blocks}")

                # Extract cluster details
                cluster_details = []
                for cluster in debug.get("clusters", []):
                    if cluster.get("accepted"):
                        texts = cluster.get("texts", [])
                        cluster_details.append({
                            "line_count": cluster.get("line_count"),
                            "texts": texts[:5] if len(texts) > 5 else texts  # Show first 5
                        })

                if cluster_details:
                    print("  Accepted clusters:")
                    for i, details in enumerate(cluster_details[:3], 1):  # Show first 3
                        print(f"    {i}. {details['line_count']} lines: {', '.join(details['texts'])}")

                print()

                results.append({
                    "eps": eps,
                    "min_samples": min_samples,
                    "h_scale": h_scale,
                    "clusters": clusters_found,
                    "noise": noise_points,
                    "blocks": final_blocks,
                })

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'eps':<8} {'min':<6} {'h_scale':<10} {'Clusters':<10} {'Noise':<8} {'Blocks':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['eps']:<8} {r['min_samples']:<6} {r['h_scale']:<10} {r['clusters']:<10} {r['noise']:<8} {r['blocks']:<8}")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nRecommended: Look for parameter combinations that produce 3-8 blocks")
    print("with wide horizontal extent (matching the expected blocks in the reference image)")


if __name__ == "__main__":
    run_smoke_test()
