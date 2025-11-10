"""
Test adjacent block merging (UC-005 / Phase 3 Increment 2)
"""

from text_block_detection import merge_adjacent_blocks


def test_merge_two_left_aligned_blocks():
    """Test Case 1: Two adjacent left-aligned blocks (should merge)."""
    print("\n" + "=" * 80)
    print("TEST 1: Merge two adjacent left-aligned blocks")
    print("=" * 80)

    blocks = [
        {
            "block_id": "block_1",
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 250.0, "y1": 215.0},  # center = 175
            "cells": [{"text": "123 Main Street"}],
            "texts": ["123 Main Street"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
        {
            "block_id": "block_2",
            "bbox": {"x0": 100.0, "y0": 220.0, "x1": 240.0, "y1": 235.0},  # center = 170
            "cells": [{"text": "Apartment 4B"}],
            "texts": ["Apartment 4B"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
    ]

    merged = merge_adjacent_blocks(blocks, tables=None, vertical_gap_threshold=15.0)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Merged blocks: {len(merged)}")
    print(f"\nMerged block bbox: {merged[0]['bbox']}")
    print(f"Merged cells: {len(merged[0]['cells'])}")
    print(f"Merged texts: {merged[0]['texts']}")

    # Expected: 2 blocks merged into 1
    assert len(merged) == 1, f"Expected 1 merged block, got {len(merged)}"
    assert len(merged[0]['cells']) == 2, f"Expected 2 cells, got {len(merged[0]['cells'])}"
    assert "merged_from" in merged[0], "Expected 'merged_from' field"

    print("\n[PASS] TEST 1 PASSED")


def test_merge_two_right_aligned_blocks():
    """Test Case 2: Two adjacent right-aligned blocks (should merge)."""
    print("\n" + "=" * 80)
    print("TEST 2: Merge two adjacent right-aligned blocks")
    print("=" * 80)

    blocks = [
        {
            "block_id": "block_1",
            "bbox": {"x0": 400.0, "y0": 200.0, "x1": 500.0, "y1": 215.0},
            "cells": [{"text": "19,631.85"}],
            "texts": ["19,631.85"],
            "alignment_metadata": {"alignment_type": "right", "confidence": 0.95},
            "dominant_alignment": "right"
        },
        {
            "block_id": "block_2",
            "bbox": {"x0": 420.0, "y0": 220.0, "x1": 500.0, "y1": 235.0},
            "cells": [{"text": "4,783.08"}],
            "texts": ["4,783.08"],
            "alignment_metadata": {"alignment_type": "right", "confidence": 0.95},
            "dominant_alignment": "right"
        },
    ]

    merged = merge_adjacent_blocks(blocks, tables=None, vertical_gap_threshold=15.0)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Merged blocks: {len(merged)}")

    # Expected: 2 blocks merged into 1
    assert len(merged) == 1, f"Expected 1 merged block, got {len(merged)}"
    assert len(merged[0]['cells']) == 2, f"Expected 2 cells, got {len(merged[0]['cells'])}"

    print("\n[PASS] TEST 2 PASSED")


def test_no_merge_gap_too_large():
    """Test Case 3: Two blocks with gap > threshold (should NOT merge)."""
    print("\n" + "=" * 80)
    print("TEST 3: No merge - gap too large")
    print("=" * 80)

    blocks = [
        {
            "block_id": "block_1",
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 250.0, "y1": 215.0},
            "cells": [{"text": "Header"}],
            "texts": ["Header"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
        {
            "block_id": "block_2",
            "bbox": {"x0": 100.0, "y0": 250.0, "x1": 200.0, "y1": 265.0},  # Gap = 35pt
            "cells": [{"text": "Body"}],
            "texts": ["Body"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
    ]

    merged = merge_adjacent_blocks(blocks, tables=None, vertical_gap_threshold=15.0)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Merged blocks: {len(merged)}")
    print(f"Gap between blocks: 35pt (exceeds threshold of 15pt)")

    # Expected: No merge, still 2 blocks
    assert len(merged) == 2, f"Expected 2 separate blocks, got {len(merged)}"

    print("\n[PASS] TEST 3 PASSED")


def test_no_merge_different_alignments():
    """Test Case 4: Two blocks with different alignments (should NOT merge)."""
    print("\n" + "=" * 80)
    print("TEST 4: No merge - different alignments")
    print("=" * 80)

    blocks = [
        {
            "block_id": "block_1",
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 250.0, "y1": 215.0},
            "cells": [{"text": "Label"}],
            "texts": ["Label"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
        {
            "block_id": "block_2",
            "bbox": {"x0": 400.0, "y0": 220.0, "x1": 500.0, "y1": 235.0},
            "cells": [{"text": "Value"}],
            "texts": ["Value"],
            "alignment_metadata": {"alignment_type": "right", "confidence": 0.9},
            "dominant_alignment": "right"
        },
    ]

    merged = merge_adjacent_blocks(blocks, tables=None, vertical_gap_threshold=15.0)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Merged blocks: {len(merged)}")
    print(f"Block 1 alignment: left, Block 2 alignment: right")

    # Expected: No merge, still 2 blocks
    assert len(merged) == 2, f"Expected 2 separate blocks, got {len(merged)}"

    print("\n[PASS] TEST 4 PASSED")


def test_merge_three_block_address():
    """Test Case 5: Multi-line address (3 blocks → 1 block)."""
    print("\n" + "=" * 80)
    print("TEST 5: Merge three-block address")
    print("=" * 80)

    blocks = [
        {
            "block_id": "block_1",
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 250.0, "y1": 215.0},  # center = 175
            "cells": [{"text": "123 Main Street"}],
            "texts": ["123 Main Street"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
        {
            "block_id": "block_2",
            "bbox": {"x0": 100.0, "y0": 220.0, "x1": 240.0, "y1": 235.0},  # center = 170
            "cells": [{"text": "Apartment 4B"}],
            "texts": ["Apartment 4B"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
        {
            "block_id": "block_3",
            "bbox": {"x0": 100.0, "y0": 240.0, "x1": 230.0, "y1": 255.0},  # center = 165
            "cells": [{"text": "Springfield"}],
            "texts": ["Springfield"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
    ]

    merged = merge_adjacent_blocks(blocks, tables=None, vertical_gap_threshold=15.0)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Merged blocks: {len(merged)}")
    print(f"\nMerged block bbox: {merged[0]['bbox']}")
    print(f"Merged cells: {len(merged[0]['cells'])}")
    print(f"Merged texts: {merged[0]['texts']}")

    # Expected: 3 blocks merged into 1
    assert len(merged) == 1, f"Expected 1 merged block, got {len(merged)}"
    assert len(merged[0]['cells']) == 3, f"Expected 3 cells, got {len(merged[0]['cells'])}"

    print("\n[PASS] TEST 5 PASSED")


def test_no_merge_different_tables():
    """Test Case 6: Blocks in different tables (should NOT merge)."""
    print("\n" + "=" * 80)
    print("TEST 6: No merge - different tables")
    print("=" * 80)

    blocks = [
        {
            "block_id": "block_1",
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 250.0, "y1": 215.0},
            "cells": [{"text": "Table 1 Data"}],
            "texts": ["Table 1 Data"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
        {
            "block_id": "block_2",
            "bbox": {"x0": 100.0, "y0": 220.0, "x1": 250.0, "y1": 235.0},
            "cells": [{"text": "Table 2 Data"}],
            "texts": ["Table 2 Data"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
    ]

    # Define two separate tables
    tables = [
        {"bbox": {"x0": 90.0, "y0": 190.0, "x1": 260.0, "y1": 218.0}},  # Contains block 1
        {"bbox": {"x0": 90.0, "y0": 219.0, "x1": 260.0, "y1": 240.0}},  # Contains block 2
    ]

    merged = merge_adjacent_blocks(blocks, tables=tables, vertical_gap_threshold=15.0)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Merged blocks: {len(merged)}")
    print(f"Block 1 in table 0, Block 2 in table 1")

    # Expected: No merge, still 2 blocks (different tables)
    assert len(merged) == 2, f"Expected 2 separate blocks, got {len(merged)}"

    print("\n[PASS] TEST 6 PASSED")


def test_no_merge_center_misaligned():
    """Test Case 7: Blocks with misaligned centers (should NOT merge)."""
    print("\n" + "=" * 80)
    print("TEST 7: No merge - centers misaligned")
    print("=" * 80)

    blocks = [
        {
            "block_id": "block_1",
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 250.0, "y1": 215.0},  # center = 175
            "cells": [{"text": "Left column"}],
            "texts": ["Left column"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
        {
            "block_id": "block_2",
            "bbox": {"x0": 300.0, "y0": 220.0, "x1": 450.0, "y1": 235.0},  # center = 375
            "cells": [{"text": "Right column"}],
            "texts": ["Right column"],
            "alignment_metadata": {"alignment_type": "left", "confidence": 0.9},
            "dominant_alignment": "left"
        },
    ]

    merged = merge_adjacent_blocks(blocks, tables=None,
                                   vertical_gap_threshold=15.0,
                                   horizontal_center_tolerance=20.0)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Merged blocks: {len(merged)}")
    print(f"Center offset: 200pt (exceeds tolerance of 20pt)")

    # Expected: No merge, still 2 blocks (centers too far apart)
    assert len(merged) == 2, f"Expected 2 separate blocks, got {len(merged)}"

    print("\n[PASS] TEST 7 PASSED")


def main():
    print("\n" + "=" * 80)
    print("ADJACENT BLOCK MERGING TEST SUITE")
    print("Phase 3 / UC-005 / Increment 2")
    print("=" * 80)

    try:
        test_merge_two_left_aligned_blocks()
        test_merge_two_right_aligned_blocks()
        test_no_merge_gap_too_large()
        test_no_merge_different_alignments()
        test_merge_three_block_address()
        test_no_merge_different_tables()
        test_no_merge_center_misaligned()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print("\nAdjacent block merging is working correctly.")
        print("Ready to proceed to Increment 3: Nested Block Detection\n")

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
