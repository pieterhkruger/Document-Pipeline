"""
Test nested block detection (UC-005 / Phase 3 Increment 3)
"""

from text_block_detection import detect_nested_blocks


def test_remove_nested_block_low_confidence():
    """Test Case 1: Remove nested block with significantly lower confidence."""
    print("\n" + "=" * 80)
    print("TEST 1: Remove nested block (low confidence)")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 500.0, "y1": 600.0},  # Large block: 400x400
            "cells": [{"text": f"Cell {i}"} for i in range(50)],
            "texts": ["Large table block"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "confidence": 0.85  # High confidence
        },
        {
            "bbox": {"x0": 150.0, "y0": 250.0, "x1": 250.0, "y1": 350.0},  # Small nested: 100x100 (fully inside)
            "cells": [{"text": f"Cell {i}"} for i in range(3)],
            "texts": ["Small nested block"],
            "alignment_metadata": {"alignment_type": "center", "confidence": 0.5},
            "confidence": 0.35  # Low confidence
        },
    ]

    filtered = detect_nested_blocks(blocks, tables=None, iou_threshold=0.8)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")
    print(f"Block 1 confidence: {blocks[0]['confidence']}")
    print(f"Block 2 confidence: {blocks[1]['confidence']}")
    print(f"Confidence difference: {blocks[0]['confidence'] - blocks[1]['confidence']}")

    # Expected: Small nested block removed (confidence diff = 0.50 > 0.20)
    assert len(filtered) == 1, f"Expected 1 block after filtering, got {len(filtered)}"
    assert filtered[0] == blocks[0], "Expected large block to remain"

    print("\n[PASS] TEST 1 PASSED")


def test_keep_nested_block_similar_confidence():
    """Test Case 2: Keep nested block when confidence is similar (ambiguous case)."""
    print("\n" + "=" * 80)
    print("TEST 2: Keep nested block (similar confidence)")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 500.0, "y1": 600.0},  # Large block: 400x400
            "cells": [{"text": f"Cell {i}"} for i in range(30)],
            "texts": ["Large block"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.8},
            "confidence": 0.70
        },
        {
            "bbox": {"x0": 150.0, "y0": 250.0, "x1": 250.0, "y1": 350.0},  # Small nested: 100x100 (fully inside)
            "cells": [{"text": f"Cell {i}"} for i in range(10)],
            "texts": ["Small block"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.75},
            "confidence": 0.65  # Similar confidence (diff = 0.05)
        },
    ]

    filtered = detect_nested_blocks(blocks, tables=None, iou_threshold=0.8)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")
    print(f"Block 1 confidence: {blocks[0]['confidence']}")
    print(f"Block 2 confidence: {blocks[1]['confidence']}")
    print(f"Confidence difference: {blocks[0]['confidence'] - blocks[1]['confidence']}")

    # Expected: Both blocks kept (confidence diff = 0.05 < 0.20)
    assert len(filtered) == 2, f"Expected 2 blocks kept, got {len(filtered)}"

    print("\n[PASS] TEST 2 PASSED")


def test_keep_similar_sized_blocks():
    """Test Case 3: Keep both blocks when they're similar size (not nested)."""
    print("\n" + "=" * 80)
    print("TEST 3: Keep similar-sized blocks (not nested)")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 400.0, "y1": 500.0},  # Block 1: 300x300
            "cells": [{"text": f"Cell {i}"} for i in range(30)],
            "texts": ["Block 1"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "confidence": 0.85
        },
        {
            "bbox": {"x0": 150.0, "y0": 250.0, "x1": 450.0, "y1": 550.0},  # Block 2: 300x300
            "cells": [{"text": f"Cell {i}"} for i in range(25)],
            "texts": ["Block 2"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.8},
            "confidence": 0.75
        },
    ]

    filtered = detect_nested_blocks(blocks, tables=None, iou_threshold=0.8, area_ratio_threshold=0.5)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")
    print(f"Block 1 area: 90000pt²")
    print(f"Block 2 area: 90000pt²")
    print(f"Area ratio: 1.0 (similar size)")

    # Expected: Both blocks kept (area ratio = 1.0 > 0.5)
    assert len(filtered) == 2, f"Expected 2 blocks kept, got {len(filtered)}"

    print("\n[PASS] TEST 3 PASSED")


def test_remove_multiple_nested_blocks():
    """Test Case 4: Remove multiple nested blocks."""
    print("\n" + "=" * 80)
    print("TEST 4: Remove multiple nested blocks")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 500.0, "y1": 600.0},  # Large block: 400x400
            "cells": [{"text": f"Cell {i}"} for i in range(50)],
            "texts": ["Large block"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "confidence": 0.85
        },
        {
            "bbox": {"x0": 150.0, "y0": 250.0, "x1": 250.0, "y1": 350.0},  # Nested 1: 100x100 (fully inside)
            "cells": [{"text": f"Cell {i}"} for i in range(3)],
            "texts": ["Nested 1"],
            "alignment_metadata": {"alignment_type": "center", "confidence": 0.5},
            "confidence": 0.30
        },
        {
            "bbox": {"x0": 300.0, "y0": 400.0, "x1": 400.0, "y1": 500.0},  # Nested 2: 100x100 (fully inside)
            "cells": [{"text": f"Cell {i}"} for i in range(4)],
            "texts": ["Nested 2"],
            "alignment_metadata": {"alignment_type": "center", "confidence": 0.4},
            "confidence": 0.35
        },
    ]

    filtered = detect_nested_blocks(blocks, tables=None, iou_threshold=0.8)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")

    # Expected: Only large block remains (2 nested blocks removed)
    assert len(filtered) == 1, f"Expected 1 block after filtering, got {len(filtered)}"
    assert filtered[0] == blocks[0], "Expected large block to remain"

    print("\n[PASS] TEST 4 PASSED")


def test_no_nested_blocks():
    """Test Case 5: No nested blocks (all separate)."""
    print("\n" + "=" * 80)
    print("TEST 5: No nested blocks")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 400.0},
            "cells": [{"text": f"Cell {i}"} for i in range(10)],
            "texts": ["Block 1"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.8},
            "confidence": 0.70
        },
        {
            "bbox": {"x0": 350.0, "y0": 200.0, "x1": 550.0, "y1": 400.0},
            "cells": [{"text": f"Cell {i}"} for i in range(12)],
            "texts": ["Block 2"],
            "alignment_metadata": {"alignment_type": "right", "confidence": 0.85},
            "confidence": 0.75
        },
        {
            "bbox": {"x0": 100.0, "y0": 450.0, "x1": 300.0, "y1": 650.0},
            "cells": [{"text": f"Cell {i}"} for i in range(15)],
            "texts": ["Block 3"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "confidence": 0.80
        },
    ]

    filtered = detect_nested_blocks(blocks, tables=None, iou_threshold=0.8)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")
    print(f"All blocks are separate (no nesting)")

    # Expected: All 3 blocks kept (no nesting)
    assert len(filtered) == 3, f"Expected 3 blocks kept, got {len(filtered)}"

    print("\n[PASS] TEST 5 PASSED")


def test_partial_overlap_not_nested():
    """Test Case 6: Partial overlap with low IoU (not nested)."""
    print("\n" + "=" * 80)
    print("TEST 6: Partial overlap (not nested)")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 400.0, "y1": 500.0},
            "cells": [{"text": f"Cell {i}"} for i in range(30)],
            "texts": ["Block 1"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "confidence": 0.85
        },
        {
            "bbox": {"x0": 300.0, "y0": 400.0, "x1": 600.0, "y1": 700.0},  # Partial overlap
            "cells": [{"text": f"Cell {i}"} for i in range(5)],
            "texts": ["Block 2"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.6},
            "confidence": 0.50
        },
    ]

    filtered = detect_nested_blocks(blocks, tables=None, iou_threshold=0.8)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")
    print(f"Blocks have partial overlap (IoU < 0.8)")

    # Expected: Both blocks kept (IoU < 0.8)
    assert len(filtered) == 2, f"Expected 2 blocks kept, got {len(filtered)}"

    print("\n[PASS] TEST 6 PASSED")


def main():
    print("\n" + "=" * 80)
    print("NESTED BLOCK DETECTION TEST SUITE")
    print("Phase 3 / UC-005 / Increment 3")
    print("=" * 80)

    try:
        test_remove_nested_block_low_confidence()
        test_keep_nested_block_similar_confidence()
        test_keep_similar_sized_blocks()
        test_remove_multiple_nested_blocks()
        test_no_nested_blocks()
        test_partial_overlap_not_nested()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print("\nNested block detection is working correctly.")
        print("Ready to proceed to Increment 4: Weak Evidence Filtering\n")

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
