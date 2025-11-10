"""
Diagnostic script to investigate Docling visualization mapping issues.

This script investigates both Blue/Cyan and Orange/Brown boxes:
1. Loads Docling payload
2. Extracts table regions (Blue/Cyan boxes)
3. Extracts picture regions (potential Orange/Brown boxes)
4. Extracts rare properties from Adobe/pikepdf payloads (Orange/Brown boxes)
5. Extracts right-aligned baselines (Orange vertical lines)
6. Logs coordinate transformations and mapping issues
7. Saves comprehensive debug JSON

Usage:
    python diagnose_docling_tables.py "Document Name" [page_no]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_docling_payload(doc_name: str) -> Optional[Dict[str, Any]]:
    """Load Docling payload for a document."""
    payload_dir = Path("Docling payloads")

    # Try exact match first
    for file in payload_dir.glob("*.json"):
        if doc_name in file.stem:
            print(f"Loading payload: {file.name}")
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)

    return None


def extract_table_info(payload: Dict[str, Any], page_no: int = 0) -> Dict[str, Any]:
    """
    Extract comprehensive table information from Docling payload.

    Returns detailed information about:
    - Raw table bboxes from Docling
    - Table types/classifications
    - Coordinate systems
    - Cell counts
    """
    debug_info = {
        "page_number": page_no,
        "docling_tables": [],
        "coordinate_system": {},
        "transformation_issues": []
    }

    # Get page data
    parsed_pages = payload.get("parsed_pages", {})
    page_data = parsed_pages.get(str(page_no))

    if not page_data:
        debug_info["error"] = f"Page {page_no} not found in payload"
        return debug_info

    # Page dimensions
    page_size = page_data.get("size", {})
    debug_info["coordinate_system"] = {
        "page_width": page_size.get("width", 0),
        "page_height": page_size.get("height", 0),
        "origin": "Docling uses top-left origin (y increases downward)"
    }

    # Extract tables
    tables = page_data.get("tables", [])
    print(f"\nFound {len(tables)} tables in Docling payload for page {page_no}")

    for i, table in enumerate(tables):
        table_info = {
            "table_index": i,
            "raw_table_data": table,
            "bbox_analysis": {}
        }

        # Get bounding box
        bbox = table.get("bbox", {})
        if bbox:
            table_info["bbox_analysis"] = {
                "raw_bbox": bbox,
                "bbox_type": type(bbox).__name__,
                "bbox_keys": list(bbox.keys()) if isinstance(bbox, dict) else None,
                "coordinates": {
                    "l": bbox.get("l", bbox.get("x0", None)),
                    "t": bbox.get("t", bbox.get("y0", None)),
                    "r": bbox.get("r", bbox.get("x1", None)),
                    "b": bbox.get("b", bbox.get("y1", None))
                }
            }

            # Calculate dimensions
            if all(k in bbox for k in ["l", "t", "r", "b"]):
                width = bbox["r"] - bbox["l"]
                height = bbox["b"] - bbox["t"]
                table_info["bbox_analysis"]["dimensions"] = {
                    "width_pt": round(width, 2),
                    "height_pt": round(height, 2),
                    "area_pt2": round(width * height, 2)
                }

            # Check for coordinate issues
            if bbox.get("t", 0) > bbox.get("b", 0):
                table_info["bbox_analysis"]["warning"] = "Top > Bottom (inverted Y-axis?)"
                debug_info["transformation_issues"].append(f"Table {i}: Inverted Y coordinates")

        # Get table type/classification
        table_info["classification"] = {
            "type": table.get("type", "unknown"),
            "label": table.get("label", "unknown")
        }

        # Get table structure
        grid = table.get("grid", [])
        table_info["structure"] = {
            "num_rows": len(grid),
            "num_cols": len(grid[0]) if grid else 0,
            "total_cells": sum(len(row) for row in grid)
        }

        # Get table text content (first few cells for reference)
        sample_text = []
        for row_idx, row in enumerate(grid[:3]):  # First 3 rows
            for col_idx, cell in enumerate(row[:3]):  # First 3 cols
                cell_text = cell.get("text", "").strip()
                if cell_text:
                    sample_text.append(f"[{row_idx},{col_idx}]: {cell_text[:50]}")

        table_info["sample_content"] = sample_text[:10]

        debug_info["docling_tables"].append(table_info)

    return debug_info


def extract_cells_from_payload(payload: Dict[str, Any], page_no: int = 0) -> List[Dict[str, Any]]:
    """Extract cell information from payload."""
    from pdf_ocr_detector import extract_bbox_coords

    parsed_pages = payload.get("parsed_pages", {})
    page_data = parsed_pages.get(str(page_no))

    if not page_data:
        return []

    cells = page_data.get("cells", [])

    cell_info = []
    for cell in cells:
        rect = cell.get("rect", {})
        text = cell.get("text", "")

        if not rect:
            continue

        try:
            x0, y0, x1, y1 = extract_bbox_coords(rect)
            cell_info.append({
                "text": text,
                "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            })
        except Exception:
            continue

    return cell_info


def extract_picture_info(payload: Dict[str, Any], page_no: int = 0) -> Dict[str, Any]:
    """
    Extract Docling picture/figure information from payload.

    Pictures might be visualized as Orange/Brown boxes.
    """
    debug_info = {
        "page_number": page_no,
        "docling_pictures": [],
        "transformation_issues": []
    }

    # Get docling_document structure
    doc_struct = payload.get("docling_document", {})
    pictures = doc_struct.get("pictures", [])

    print(f"\nFound {len(pictures)} pictures in Docling payload")

    for i, picture in enumerate(pictures):
        # Get provenance (location info)
        prov_entries = picture.get("prov", [])

        for prov in prov_entries:
            if not isinstance(prov, dict):
                continue

            # Check if this picture is on the target page
            pic_page = prov.get("page_no", 0)
            if pic_page != (page_no + 1):  # Docling uses 1-based page numbers
                continue

            bbox = prov.get("bbox", {})
            if not bbox:
                continue

            picture_info = {
                "picture_index": i,
                "label": picture.get("label", "unknown"),
                "text": picture.get("text", "")[:100],  # First 100 chars
                "bbox_analysis": {
                    "raw_bbox": bbox,
                    "coord_origin": bbox.get("coord_origin", "unknown"),
                    "coordinates": {
                        "l": bbox.get("l"),
                        "t": bbox.get("t"),
                        "r": bbox.get("r"),
                        "b": bbox.get("b")
                    }
                }
            }

            # Calculate dimensions
            if all(k in bbox for k in ["l", "t", "r", "b"]):
                width = bbox["r"] - bbox["l"]
                height = abs(bbox["b"] - bbox["t"])  # Use abs in case of coord system differences
                picture_info["bbox_analysis"]["dimensions"] = {
                    "width_pt": round(width, 2),
                    "height_pt": round(height, 2),
                    "area_pt2": round(width * height, 2)
                }

            # Note coordinate system
            coord_origin = bbox.get("coord_origin", "UNKNOWN")
            if coord_origin == "BOTTOMLEFT":
                picture_info["bbox_analysis"]["note"] = "BOTTOMLEFT origin (y increases upward)"
            elif coord_origin == "TOPLEFT":
                picture_info["bbox_analysis"]["note"] = "TOPLEFT origin (y increases downward)"

            debug_info["docling_pictures"].append(picture_info)

    return debug_info


def load_adobe_payload(doc_name: str) -> Optional[Dict[str, Any]]:
    """Load Adobe payload for a document."""
    payload_dir = Path("Adobe payloads")

    if not payload_dir.exists():
        return None

    for file in payload_dir.glob("*.json"):
        if doc_name in file.stem:
            print(f"Loading Adobe payload: {file.name}")
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)

    return None


def extract_rare_properties_info(adobe_payload: Optional[Dict[str, Any]], page_no: int = 0) -> Dict[str, Any]:
    """
    Extract rare property information from Adobe payload.

    These are visualized as Orange/Brown boxes:
    - Orange (255, 165, 0): embedded fonts
    - Dark Orange (255, 140, 0): rare text sizes
    - Brown (139, 69, 19): rare line heights
    """
    debug_info = {
        "page_number": page_no,
        "has_adobe_payload": adobe_payload is not None,
        "rare_properties": {
            "embedded": [],
            "text_size": [],
            "line_height": []
        }
    }

    if not adobe_payload:
        return debug_info

    # This would require implementing the full rare property analysis
    # For now, just note that we'd extract bounding boxes for:
    # - Embedded fonts (orange)
    # - Rare text sizes (dark orange)
    # - Rare line heights (brown)

    debug_info["note"] = "Adobe rare property extraction would require full payload analysis"

    return debug_info


def extract_baseline_info(doc_name: str, page_no: int = 0) -> Dict[str, Any]:
    """
    Extract alignment baseline information.

    Right-aligned baselines are visualized as Orange vertical lines:
    - Dark Orange (255, 140, 0): page_right position
    - Orange (255, 165, 0): mid-page position
    """
    debug_info = {
        "page_number": page_no,
        "found_alignment_logic": False,
        "baselines": {
            "left": [],
            "right": [],
            "center": []
        }
    }

    alignment_logic_dir = Path("Alignment logic")

    for file in alignment_logic_dir.glob("*.json"):
        if doc_name.replace(" ", "_").replace("–", "").replace("-", "_") in file.stem.replace("-", "_"):
            debug_info["found_alignment_logic"] = True
            debug_info["alignment_logic_file"] = str(file)

            with open(file, 'r', encoding='utf-8') as f:
                alignment_data = json.load(f)

            # Extract baselines
            pages = alignment_data.get("pages", {})
            page_data = pages.get(str(page_no), {})
            baselines = page_data.get("baselines", {})

            for orientation in ["left", "right", "center"]:
                baseline_list = baselines.get(orientation, [])
                for baseline in baseline_list:
                    baseline_info = {
                        "value": baseline.get("value"),
                        "count": baseline.get("count"),
                        "page_position": baseline.get("page_position"),
                        "cells": baseline.get("cells", [])[:5]  # First 5 cells
                    }
                    debug_info["baselines"][orientation].append(baseline_info)

            break

    return debug_info


def compare_with_alignment_logic(doc_name: str, page_no: int = 0) -> Dict[str, Any]:
    """
    Compare Docling tables with what's in the alignment logic file.
    """
    alignment_logic_dir = Path("Alignment logic")

    comparison = {
        "found_alignment_logic": False,
        "tables_in_alignment_logic": [],
        "discrepancies": []
    }

    # Find alignment logic file
    for file in alignment_logic_dir.glob("*.json"):
        if doc_name.replace(" ", "_").replace("–", "").replace("-", "_") in file.stem.replace("-", "_"):
            comparison["found_alignment_logic"] = True
            comparison["alignment_logic_file"] = str(file)

            with open(file, 'r', encoding='utf-8') as f:
                alignment_data = json.load(f)

            # Extract tables from alignment logic
            pages = alignment_data.get("pages", {})
            page_data = pages.get(str(page_no), {})
            blocks = page_data.get("blocks", {}).get("blocks", [])

            # Count table-related blocks
            table_blocks = [b for b in blocks if b.get("in_table", False)]
            comparison["tables_in_alignment_logic"] = {
                "total_blocks": len(blocks),
                "table_blocks": len(table_blocks),
                "sample_table_blocks": [
                    {
                        "bbox": b.get("bbox"),
                        "cells": len(b.get("cells", [])),
                        "alignment": b.get("alignment_metadata", {}).get("alignment_type")
                    }
                    for b in table_blocks[:5]
                ]
            }

            break

    return comparison


def diagnose_document(doc_name: str, page_no: int = 0) -> Dict[str, Any]:
    """
    Run comprehensive diagnostic on a document.
    Investigates both Blue/Cyan (tables) and Orange/Brown (pictures, baselines, rare properties) boxes.
    """
    print("=" * 80)
    print(f"DIAGNOSING: {doc_name}")
    print(f"Page: {page_no}")
    print("=" * 80)

    diagnostic_report = {
        "document_name": doc_name,
        "page_number": page_no,
        "timestamp": None,
        "payload_analysis": {},
        "table_analysis": {},
        "picture_analysis": {},
        "baseline_analysis": {},
        "rare_properties_analysis": {},
        "cell_analysis": {},
        "comparison": {},
        "recommendations": []
    }

    # Load Docling payload
    payload = load_docling_payload(doc_name)
    if not payload:
        diagnostic_report["error"] = "Could not find Docling payload"
        print(f"\nERROR: Could not find Docling payload for '{doc_name}'")
        return diagnostic_report

    # ===== BLUE/CYAN BOXES: DOCLING TABLES =====
    print("\n" + "=" * 80)
    print("BLUE/CYAN BOXES: DOCLING TABLES")
    print("=" * 80)
    table_info = extract_table_info(payload, page_no)
    diagnostic_report["table_analysis"] = table_info

    # Print table summary
    for table in table_info.get("docling_tables", []):
        print(f"\nTable {table['table_index']}:")
        print(f"  BBox: {table['bbox_analysis'].get('coordinates')}")
        print(f"  Dimensions: {table['bbox_analysis'].get('dimensions')}")
        print(f"  Structure: {table['structure']}")
        print(f"  Sample content: {table['sample_content'][:3]}")

    # ===== ORANGE/BROWN BOXES: DOCLING PICTURES =====
    print("\n" + "=" * 80)
    print("ORANGE/BROWN BOXES: DOCLING PICTURES")
    print("=" * 80)
    picture_info = extract_picture_info(payload, page_no)
    diagnostic_report["picture_analysis"] = picture_info

    # Print picture summary
    for picture in picture_info.get("docling_pictures", []):
        print(f"\nPicture {picture['picture_index']}:")
        print(f"  Label: {picture['label']}")
        print(f"  BBox: {picture['bbox_analysis'].get('coordinates')}")
        print(f"  Dimensions: {picture['bbox_analysis'].get('dimensions')}")
        print(f"  Coord origin: {picture['bbox_analysis'].get('coord_origin')}")

    # ===== ORANGE LINES: RIGHT ALIGNMENT BASELINES =====
    print("\n" + "=" * 80)
    print("ORANGE LINES: ALIGNMENT BASELINES")
    print("=" * 80)
    baseline_info = extract_baseline_info(doc_name, page_no)
    diagnostic_report["baseline_analysis"] = baseline_info

    if baseline_info["found_alignment_logic"]:
        print(f"\nFound alignment logic file: {baseline_info['alignment_logic_file']}")
        for orientation in ["left", "right", "center"]:
            baselines = baseline_info["baselines"].get(orientation, [])
            if baselines:
                print(f"\n{orientation.upper()} baselines: {len(baselines)}")
                for bl in baselines[:3]:  # First 3
                    print(f"  Value: {bl['value']:.2f}pt, Count: {bl['count']}, Position: {bl['page_position']}")
    else:
        print("No alignment logic file found")

    # ===== ORANGE/BROWN BOXES: RARE ADOBE PROPERTIES =====
    print("\n" + "=" * 80)
    print("ORANGE/BROWN BOXES: RARE ADOBE PROPERTIES")
    print("=" * 80)
    adobe_payload = load_adobe_payload(doc_name)
    rare_props_info = extract_rare_properties_info(adobe_payload, page_no)
    diagnostic_report["rare_properties_analysis"] = rare_props_info

    if rare_props_info["has_adobe_payload"]:
        print("Adobe payload loaded (rare properties would be extracted here)")
    else:
        print("No Adobe payload found")

    # ===== CELL EXTRACTION =====
    print("\n" + "=" * 80)
    print("CELL EXTRACTION")
    print("=" * 80)
    cells = extract_cells_from_payload(payload, page_no)
    diagnostic_report["cell_analysis"] = {
        "total_cells": len(cells),
        "sample_cells": [
            {"text": c["text"][:50], "bbox": c["bbox"]}
            for c in cells[:10]
        ]
    }
    print(f"Extracted {len(cells)} cells from payload")

    # ===== ALIGNMENT LOGIC COMPARISON =====
    print("\n" + "=" * 80)
    print("ALIGNMENT LOGIC COMPARISON")
    print("=" * 80)
    comparison = compare_with_alignment_logic(doc_name, page_no)
    diagnostic_report["comparison"] = comparison

    if comparison["found_alignment_logic"]:
        print(f"Found alignment logic file: {comparison['alignment_logic_file']}")
        print(f"Total blocks: {comparison['tables_in_alignment_logic']['total_blocks']}")
        print(f"Table blocks: {comparison['tables_in_alignment_logic']['table_blocks']}")
    else:
        print("No alignment logic file found")

    # ===== GENERATE RECOMMENDATIONS =====
    recommendations = []

    # Table coordinate issues
    if table_info.get("transformation_issues"):
        recommendations.append({
            "issue": "Table coordinate transformation problems detected",
            "details": table_info["transformation_issues"],
            "suggestion": "Check Y-axis inversion in coordinate mapping for tables"
        })

    # No tables detected
    if len(table_info.get("docling_tables", [])) == 0:
        recommendations.append({
            "issue": "No tables detected by Docling",
            "suggestion": "Check if document actually contains tables or if Docling detection failed"
        })

    # Picture coordinate system differences
    if len(picture_info.get("docling_pictures", [])) > 0:
        coord_origins = set(p["bbox_analysis"].get("coord_origin") for p in picture_info["docling_pictures"])
        if "BOTTOMLEFT" in coord_origins:
            recommendations.append({
                "issue": "Pictures use BOTTOMLEFT coordinate system",
                "suggestion": "Verify coordinate transformation from BOTTOMLEFT to visualization coordinate system"
            })

    # Missing baselines
    if baseline_info["found_alignment_logic"]:
        total_baselines = sum(len(baseline_info["baselines"].get(o, [])) for o in ["left", "right", "center"])
        if total_baselines == 0:
            recommendations.append({
                "issue": "No alignment baselines found",
                "suggestion": "Check if baseline detection ran successfully"
            })

    diagnostic_report["recommendations"] = recommendations

    return diagnostic_report


def save_diagnostic_report(report: Dict[str, Any], doc_name: str):
    """Save diagnostic report to JSON file."""
    output_dir = Path("Diagnostic Reports")
    output_dir.mkdir(exist_ok=True)

    # Sanitize filename
    safe_name = doc_name.replace(" ", "_").replace("–", "-").replace(".", "")
    output_file = output_dir / f"{safe_name}_table_diagnostic.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"Diagnostic report saved to: {output_file}")
    print(f"{'=' * 80}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_docling_tables.py \"Document Name\"")
        print("\nAvailable documents:")
        payload_dir = Path("Docling payloads")
        for file in sorted(payload_dir.glob("*.json"))[:10]:
            print(f"  - {file.stem}")
        return 1

    doc_name = sys.argv[1]
    page_no = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # Run diagnostic
    report = diagnose_document(doc_name, page_no)

    # Save report
    save_diagnostic_report(report, doc_name)

    return 0


if __name__ == "__main__":
    exit(main())
