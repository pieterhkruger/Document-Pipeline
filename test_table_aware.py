"""
Test table-aware clustering implementation
"""

import json
from pathlib import Path

# Test parameters
TEST_DOC = "85393423298 – Mr G Motau - False pay slip"

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
    from text_block_detection import (
        identify_text_blocks_iterative,
        extract_table_metadata,
    )

    print(f"Testing table-aware clustering on: {TEST_DOC}")
    payload = load_docling_payload(TEST_DOC)
    cells, baselines = extract_cells_and_baselines(payload, page_no=0)

    print(f"Extracted {len(cells)} cells and {len(baselines)} baselines\n")

    # Extract table metadata
    tables = extract_table_metadata(payload, page_no=0)
    print(f"Found {len(tables)} tables on page 0")
    for i, table in enumerate(tables):
        print(f"  Table {i}: {table['num_rows']} rows × {table['num_cols']} cols, {len(table['cells'])} cells")

    # Test 1: Without table awareness
    print("\n" + "="*80)
    print("TEST 1: Without table awareness")
    print("="*80)
    debug1 = {}
    blocks1 = identify_text_blocks_iterative(
        cells, baselines,
        tables=None,  # No tables
        use_table_awareness=False,
        debug=debug1
    )

    print(f"Result: {len(blocks1)} blocks created")
    print(f"Convergence: {debug1.get('convergence', {})}")
    print(f"\nSample blocks:")
    for i, block in enumerate(blocks1[:5], 1):
        texts = block.get("texts", [])
        print(f"  {i}. {block['line_count']} lines ({block['dominant_alignment']}): "
              f"{', '.join(texts[:5])}")

    # Test 2: With table awareness
    print("\n" + "="*80)
    print("TEST 2: With table awareness")
    print("="*80)
    debug2 = {}
    blocks2 = identify_text_blocks_iterative(
        cells, baselines,
        tables=tables,  # Provide tables
        use_table_awareness=True,
        debug=debug2
    )

    print(f"Result: {len(blocks2)} blocks created")
    print(f"Convergence: {debug2.get('convergence', {})}")
    print(f"Table enrichment: {debug2.get('inputs', {}).get('cells_in_tables', 0)} cells in tables, "
          f"{debug2.get('inputs', {}).get('cells_not_in_tables', 0)} cells not in tables")

    print(f"\nSample blocks:")
    for i, block in enumerate(blocks2[:5], 1):
        texts = block.get("texts", [])
        print(f"  {i}. {block['line_count']} lines ({block['dominant_alignment']}): "
              f"{', '.join(texts[:5])}")

    # Save results
    output_dir = Path("Alignment logic")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file1 = output_dir / "table_aware_test_WITHOUT_tables.json"
    with open(output_file1, 'w', encoding='utf-8') as f:
        json.dump({
            "blocks": blocks1,
            "debug": debug1
        }, f, indent=2, ensure_ascii=False)

    output_file2 = output_dir / "table_aware_test_WITH_tables.json"
    with open(output_file2, 'w', encoding='utf-8') as f:
        json.dump({
            "blocks": blocks2,
            "debug": debug2,
            "tables": tables  # Include table metadata
        }, f, indent=2, ensure_ascii=False)

    print(f"\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Without table awareness: {len(blocks1)} blocks")
    print(f"With table awareness:    {len(blocks2)} blocks")
    print(f"Difference:              {len(blocks2) - len(blocks1):+d} blocks")

    print(f"\nResults saved to:")
    print(f"  {output_file1}")
    print(f"  {output_file2}")


if __name__ == "__main__":
    main()
