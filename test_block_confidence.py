"""
Test block confidence scoring functions (UC-005 / Phase 3 Increment 1)
"""

from text_block_detection import (
    calculate_bbox_overlap,
    calculate_block_confidence
)


def test_bbox_overlap_full_overlap():
    """Test Case 1: Identical bounding boxes (100% overlap)."""
    print("\n" + "=" * 80)
    print("TEST 1: Full overlap (identical bboxes)")
    print("=" * 80)

    bbox1 = {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 400.0}
    bbox2 = {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 400.0}

    result = calculate_bbox_overlap(bbox1, bbox2)

    print(f"\nBBox1: {bbox1}")
    print(f"BBox2: {bbox2}")
    print(f"\nIntersection Area: {result['intersection_area']}")
    print(f"Union Area: {result['union_area']}")
    print(f"IoU: {result['iou']}")
    print(f"Overlap Ratio 1: {result['overlap_ratio_1']}")
    print(f"Overlap Ratio 2: {result['overlap_ratio_2']}")

    # Expected: IoU = 1.0, both overlap ratios = 1.0
    assert result['iou'] == 1.0, f"Expected IoU=1.0, got {result['iou']}"
    assert result['overlap_ratio_1'] == 1.0, f"Expected overlap_ratio_1=1.0, got {result['overlap_ratio_1']}"
    assert result['overlap_ratio_2'] == 1.0, f"Expected overlap_ratio_2=1.0, got {result['overlap_ratio_2']}"

    print("\n[PASS] TEST 1 PASSED")


def test_bbox_overlap_partial_overlap():
    """Test Case 2: Partial overlap."""
    print("\n" + "=" * 80)
    print("TEST 2: Partial overlap")
    print("=" * 80)

    bbox1 = {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 400.0}
    bbox2 = {"x0": 200.0, "y0": 300.0, "x1": 400.0, "y1": 500.0}

    result = calculate_bbox_overlap(bbox1, bbox2)

    print(f"\nBBox1: {bbox1}")
    print(f"BBox2: {bbox2}")
    print(f"\nIntersection Area: {result['intersection_area']}")
    print(f"Union Area: {result['union_area']}")
    print(f"IoU: {result['iou']}")
    print(f"Overlap Ratio 1: {result['overlap_ratio_1']}")
    print(f"Overlap Ratio 2: {result['overlap_ratio_2']}")

    # Expected: IoU > 0 but < 1.0
    assert 0.0 < result['iou'] < 1.0, f"Expected 0 < IoU < 1, got {result['iou']}"
    assert result['intersection_area'] > 0, f"Expected intersection > 0"

    print("\n[PASS] TEST 2 PASSED")


def test_bbox_overlap_no_overlap():
    """Test Case 3: No overlap (disjoint boxes)."""
    print("\n" + "=" * 80)
    print("TEST 3: No overlap")
    print("=" * 80)

    bbox1 = {"x0": 100.0, "y0": 200.0, "x1": 200.0, "y1": 300.0}
    bbox2 = {"x0": 300.0, "y0": 400.0, "x1": 400.0, "y1": 500.0}

    result = calculate_bbox_overlap(bbox1, bbox2)

    print(f"\nBBox1: {bbox1}")
    print(f"BBox2: {bbox2}")
    print(f"\nIntersection Area: {result['intersection_area']}")
    print(f"IoU: {result['iou']}")

    # Expected: IoU = 0.0, all metrics = 0.0
    assert result['iou'] == 0.0, f"Expected IoU=0.0, got {result['iou']}"
    assert result['intersection_area'] == 0.0, f"Expected intersection=0.0"

    print("\n[PASS] TEST 3 PASSED")


def test_bbox_overlap_nested():
    """Test Case 4: Nested boxes (small inside large)."""
    print("\n" + "=" * 80)
    print("TEST 4: Nested boxes")
    print("=" * 80)

    bbox_large = {"x0": 100.0, "y0": 200.0, "x1": 500.0, "y1": 600.0}
    bbox_small = {"x0": 200.0, "y0": 300.0, "x1": 300.0, "y1": 400.0}

    result = calculate_bbox_overlap(bbox_large, bbox_small)

    print(f"\nLarge BBox: {bbox_large}")
    print(f"Small BBox: {bbox_small}")
    print(f"\nIntersection Area: {result['intersection_area']}")
    print(f"IoU: {result['iou']}")
    print(f"Overlap Ratio (large): {result['overlap_ratio_1']}")
    print(f"Overlap Ratio (small): {result['overlap_ratio_2']}")

    # Expected: overlap_ratio_2 = 1.0 (small is fully inside large)
    assert result['overlap_ratio_2'] == 1.0, f"Expected small fully inside, got {result['overlap_ratio_2']}"
    assert result['overlap_ratio_1'] < 1.0, f"Expected large partially overlapped"

    print("\n[PASS] TEST 4 PASSED")


def test_block_confidence_strong_block():
    """Test Case 5: Strong block (high confidence)."""
    print("\n" + "=" * 80)
    print("TEST 5: Strong block confidence")
    print("=" * 80)

    # Create a strong block: 30 cells, 95% alignment, in table
    block = {
        "bbox": {"x0": 100.0, "y0": 200.0, "x1": 400.0, "y1": 500.0},
        "cells": [{"text": f"Cell {i}"} for i in range(30)],
        "alignment_metadata": {
            "alignment_type": "right",
            "confidence": 0.95,
            "baseline_value": 400.0,
        }
    }

    # Table that overlaps block (IoU >= 0.7)
    tables = [
        {"bbox": {"x0": 90.0, "y0": 190.0, "x1": 410.0, "y1": 510.0}}
    ]

    confidence = calculate_block_confidence(block, tables, page_height=792.0)

    print(f"\nBlock: {len(block['cells'])} cells, alignment confidence: {block['alignment_metadata']['confidence']}")
    print(f"Block bbox: {block['bbox']}")
    print(f"Table bbox: {tables[0]['bbox']}")
    print(f"\nCalculated confidence: {confidence}")

    # Expected: High confidence (>0.8)
    assert confidence >= 0.8, f"Expected confidence >= 0.8 for strong block, got {confidence}"

    print("\n[PASS] TEST 5 PASSED")


def test_block_confidence_weak_block():
    """Test Case 6: Weak block (low confidence)."""
    print("\n" + "=" * 80)
    print("TEST 6: Weak block confidence")
    print("=" * 80)

    # Create a weak block: 3 cells, 50% alignment, not in table
    block = {
        "bbox": {"x0": 100.0, "y0": 200.0, "x1": 400.0, "y1": 500.0},
        "cells": [{"text": f"Cell {i}"} for i in range(3)],
        "alignment_metadata": {
            "alignment_type": "center",
            "confidence": 0.50,
            "baseline_value": None,
        }
    }

    # No tables
    tables = []

    confidence = calculate_block_confidence(block, tables, page_height=792.0)

    print(f"\nBlock: {len(block['cells'])} cells, alignment confidence: {block['alignment_metadata']['confidence']}")
    print(f"Block bbox: {block['bbox']}")
    print(f"No tables")
    print(f"\nCalculated confidence: {confidence}")

    # Expected: Low confidence (<0.5)
    assert confidence <= 0.5, f"Expected confidence <= 0.5 for weak block, got {confidence}"

    print("\n[PASS] TEST 6 PASSED")


def test_block_confidence_medium_block():
    """Test Case 7: Medium block (moderate confidence)."""
    print("\n" + "=" * 80)
    print("TEST 7: Medium block confidence")
    print("=" * 80)

    # Create a medium block: 12 cells, 75% alignment, not in table
    block = {
        "bbox": {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 400.0},
        "cells": [{"text": f"Cell {i}"} for i in range(12)],
        "alignment_metadata": {
            "alignment_type": "left",
            "confidence": 0.75,
            "baseline_value": 100.0,
        }
    }

    # No tables
    tables = []

    confidence = calculate_block_confidence(block, tables, page_height=792.0)

    print(f"\nBlock: {len(block['cells'])} cells, alignment confidence: {block['alignment_metadata']['confidence']}")
    print(f"Block bbox: {block['bbox']}")
    print(f"No tables")
    print(f"\nCalculated confidence: {confidence}")

    # Expected: Medium confidence (0.4 - 0.7)
    assert 0.4 <= confidence <= 0.7, f"Expected 0.4 <= confidence <= 0.7, got {confidence}"

    print("\n[PASS] TEST 7 PASSED")


def test_block_confidence_table_bonus():
    """Test Case 8: Table membership bonus effect."""
    print("\n" + "=" * 80)
    print("TEST 8: Table membership bonus")
    print("=" * 80)

    # Create identical blocks, one in table, one not
    block = {
        "bbox": {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 400.0},
        "cells": [{"text": f"Cell {i}"} for i in range(15)],
        "alignment_metadata": {
            "alignment_type": "left",
            "confidence": 0.80,
            "baseline_value": 100.0,
        }
    }

    # Table overlapping block
    tables_with_overlap = [
        {"bbox": {"x0": 90.0, "y0": 190.0, "x1": 310.0, "y1": 410.0}}
    ]

    confidence_with_table = calculate_block_confidence(block, tables_with_overlap, page_height=792.0)
    confidence_without_table = calculate_block_confidence(block, None, page_height=792.0)

    print(f"\nBlock: {len(block['cells'])} cells, alignment confidence: {block['alignment_metadata']['confidence']}")
    print(f"\nConfidence WITHOUT table: {confidence_without_table}")
    print(f"Confidence WITH table:    {confidence_with_table}")
    print(f"Difference (bonus):       {confidence_with_table - confidence_without_table}")

    # Expected: Difference should be ~0.20 (table bonus)
    bonus = confidence_with_table - confidence_without_table
    assert 0.15 <= bonus <= 0.25, f"Expected table bonus ~0.20, got {bonus}"

    print("\n[PASS] TEST 8 PASSED")


def main():
    print("\n" + "=" * 80)
    print("BLOCK CONFIDENCE SCORING TEST SUITE")
    print("Phase 3 / UC-005 / Increment 1")
    print("=" * 80)

    try:
        # Test bbox overlap functions
        test_bbox_overlap_full_overlap()
        test_bbox_overlap_partial_overlap()
        test_bbox_overlap_no_overlap()
        test_bbox_overlap_nested()

        # Test block confidence scoring
        test_block_confidence_strong_block()
        test_block_confidence_weak_block()
        test_block_confidence_medium_block()
        test_block_confidence_table_bonus()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print("\nBlock confidence scoring functions are working correctly.")
        print("Ready to proceed to Increment 2: Adjacent Block Merging\n")

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
