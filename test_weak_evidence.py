"""
Test weak evidence filtering (UC-005 / Phase 3 Increment 4)
"""

from text_block_detection import filter_weak_evidence_blocks


def test_remove_weak_block_with_overlap():
    """Test Case 1: Remove weak block that overlaps with strong block."""
    print("\n" + "=" * 80)
    print("TEST 1: Remove weak block (overlaps with strong)")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 300.0},  # 100pt span, 15 cells
            "cells": [{"text": f"Cell {i}"} for i in range(15)],
            "texts": ["Strong block"]
        },
        {
            "bbox": {"x0": 120.0, "y0": 150.0, "x1": 280.0, "y1": 450.0},  # 300pt span, 3 cells
            "cells": [{"text": f"Cell {i}"} for i in range(3)],
            "texts": ["Weak block"]
        },
    ]

    # Block 1 strength: 15 / 100 = 0.15 cells/pt (strong)
    # Block 2 strength: 3 / 300 = 0.01 cells/pt (weak)

    filtered = filter_weak_evidence_blocks(blocks, evidence_strength_threshold=0.05)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")
    print(f"Block 1 strength: 0.15 cells/pt (strong)")
    print(f"Block 2 strength: 0.01 cells/pt (weak, overlaps with block 1)")

    # Expected: Weak block removed
    assert len(filtered) == 1, f"Expected 1 block after filtering, got {len(filtered)}"

    print("\n[PASS] TEST 1 PASSED")


def test_keep_weak_block_no_overlap():
    """Test Case 2: Keep weak block that doesn't overlap with any strong block."""
    print("\n" + "=" * 80)
    print("TEST 2: Keep weak block (no overlap)")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 300.0},  # Strong block
            "cells": [{"text": f"Cell {i}"} for i in range(15)],
            "texts": ["Strong block"]
        },
        {
            "bbox": {"x0": 400.0, "y0": 200.0, "x1": 500.0, "y1": 500.0},  # Weak, no overlap
            "cells": [{"text": f"Cell {i}"} for i in range(3)],
            "texts": ["Weak block"]
        },
    ]

    filtered = filter_weak_evidence_blocks(blocks, evidence_strength_threshold=0.05)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")
    print(f"Weak block has no overlap with strong block")

    # Expected: Both blocks kept (no overlap)
    assert len(filtered) == 2, f"Expected 2 blocks kept, got {len(filtered)}"

    print("\n[PASS] TEST 2 PASSED")


def test_keep_all_strong_blocks():
    """Test Case 3: Keep all strong blocks."""
    print("\n" + "=" * 80)
    print("TEST 3: Keep all strong blocks")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 300.0},  # 0.15 cells/pt
            "cells": [{"text": f"Cell {i}"} for i in range(15)],
            "texts": ["Block 1"]
        },
        {
            "bbox": {"x0": 350.0, "y0": 200.0, "x1": 550.0, "y1": 300.0},  # 0.20 cells/pt
            "cells": [{"text": f"Cell {i}"} for i in range(20)],
            "texts": ["Block 2"]
        },
        {
            "bbox": {"x0": 100.0, "y0": 350.0, "x1": 300.0, "y1": 450.0},  # 0.10 cells/pt
            "cells": [{"text": f"Cell {i}"} for i in range(10)],
            "texts": ["Block 3"]
        },
    ]

    filtered = filter_weak_evidence_blocks(blocks, evidence_strength_threshold=0.05)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")
    print(f"All blocks are strong (strength > 0.05)")

    # Expected: All 3 blocks kept
    assert len(filtered) == 3, f"Expected 3 blocks kept, got {len(filtered)}"

    print("\n[PASS] TEST 3 PASSED")


def test_remove_multiple_weak_blocks():
    """Test Case 4: Remove multiple weak blocks with overlap."""
    print("\n" + "=" * 80)
    print("TEST 4: Remove multiple weak blocks")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 400.0},  # Strong
            "cells": [{"text": f"Cell {i}"} for i in range(25)],
            "texts": ["Strong block"]
        },
        {
            "bbox": {"x0": 120.0, "y0": 150.0, "x1": 280.0, "y1": 450.0},  # Weak 1
            "cells": [{"text": f"Cell {i}"} for i in range(3)],
            "texts": ["Weak 1"]
        },
        {
            "bbox": {"x0": 150.0, "y0": 180.0, "x1": 250.0, "y1": 480.0},  # Weak 2
            "cells": [{"text": f"Cell {i}"} for i in range(4)],
            "texts": ["Weak 2"]
        },
    ]

    filtered = filter_weak_evidence_blocks(blocks, evidence_strength_threshold=0.05)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")

    # Expected: Only strong block remains
    assert len(filtered) == 1, f"Expected 1 block after filtering, got {len(filtered)}"

    print("\n[PASS] TEST 4 PASSED")


def test_borderline_evidence_strength():
    """Test Case 5: Test borderline evidence strength (exactly at threshold)."""
    print("\n" + "=" * 80)
    print("TEST 5: Borderline evidence strength")
    print("=" * 80)

    blocks = [
        {
            "bbox": {"x0": 100.0, "y0": 200.0, "x1": 300.0, "y1": 300.0},  # 0.15 cells/pt
            "cells": [{"text": f"Cell {i}"} for i in range(15)],
            "texts": ["Strong block"]
        },
        {
            "bbox": {"x0": 120.0, "y0": 220.0, "x1": 280.0, "y1": 320.0},  # 5 cells / 100pt = 0.05
            "cells": [{"text": f"Cell {i}"} for i in range(5)],
            "texts": ["Borderline block"]
        },
    ]

    filtered = filter_weak_evidence_blocks(blocks, evidence_strength_threshold=0.05)

    print(f"\nOriginal blocks: {len(blocks)}")
    print(f"Filtered blocks: {len(filtered)}")
    print(f"Block 2 strength: 0.05 cells/pt (exactly at threshold)")

    # Expected: Borderline block kept (strength == threshold, not <)
    assert len(filtered) == 2, f"Expected 2 blocks kept, got {len(filtered)}"

    print("\n[PASS] TEST 5 PASSED")


def main():
    print("\n" + "=" * 80)
    print("WEAK EVIDENCE FILTERING TEST SUITE")
    print("Phase 3 / UC-005 / Increment 4")
    print("=" * 80)

    try:
        test_remove_weak_block_with_overlap()
        test_keep_weak_block_no_overlap()
        test_keep_all_strong_blocks()
        test_remove_multiple_weak_blocks()
        test_borderline_evidence_strength()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print("\nWeak evidence filtering is working correctly.")
        print("Ready to proceed to Increment 5: Conflict Resolution & Integration\n")

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
