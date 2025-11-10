"""
Quick test of iterative clustering implementation
"""

import json
from pathlib import Path

# Test parameters
TEST_DOC = "87026407408 – Mr I Fakude - False pay slip"

def load_docling_payload(pdf_name: str):
    """Load the Docling payload for a document."""
    payload_dir = Path("Docling payloads")
    for file in payload_dir.glob("*.json"):
        if pdf_name in file.stem:
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(f"No payload found for {pdf_name}")


def extract_cells_and_baselines(payload, page_no=0):
    """Extract cells and baselines from payload."""
    from pdf_ocr_detector import extract_bbox_coords
    from alignment_detection import detect_alignment_baselines

    parsed_pages = payload.get("parsed_pages", {})
    page_data = parsed_pages.get(str(page_no))

    if not page_data:
        raise ValueError(f"Page {page_no} not found")

    cells = page_data.get("cells", [])
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

    baselines_result = detect_alignment_baselines(consolidated_cells, page_no, {})
    if isinstance(baselines_result, tuple):
        baselines = baselines_result[0]
    else:
        baselines = baselines_result

    return consolidated_cells, baselines


def main():
    from text_block_detection import identify_text_blocks_iterative

    print(f"Testing iterative clustering on: {TEST_DOC}")
    payload = load_docling_payload(TEST_DOC)
    cells, baselines = extract_cells_and_baselines(payload, page_no=0)

    print(f"Extracted {len(cells)} cells and {len(baselines)} baselines\n")

    debug = {}
    blocks = identify_text_blocks_iterative(cells, baselines, debug=debug)

    print(f"Result: {len(blocks)} blocks created")
    print(f"\nConvergence Info:")
    conv = debug.get("convergence", {})
    print(f"  Converged: {conv.get('converged')}")
    print(f"  Iterations used: {conv.get('iterations_used')}")
    print(f"  Reason: {conv.get('convergence_reason')}")

    print(f"\nIteration History:")
    for it in debug.get("iterations", []):
        print(f"  Iteration {it['iteration']} ({it['dimension']}, eps={it['eps']}): "
              f"{it['blocks_before']} → {it['blocks_after']} blocks, "
              f"splits={it['splits_occurred']}")

    print(f"\nSample blocks:")
    for i, block in enumerate(blocks[:5], 1):
        texts = block.get("texts", [])
        print(f"  {i}. {block['line_count']} lines ({block['dominant_alignment']}): "
              f"{', '.join(texts[:5])}")

    # Save result
    output_file = Path("Alignment logic/iterative_test_result.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "blocks": blocks,
            "debug": debug
        }, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
