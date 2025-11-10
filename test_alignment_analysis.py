"""
Test alignment analysis functions (UC-001 / Phase 2 Increment 1)
"""

from text_block_detection import (
    analyze_block_alignment,
    calculate_alignment_confidence,
    detect_baseline_deviations,
    BASELINE_SNAP_TOLERANCE
)


def test_right_aligned_column_with_deviation():
    """
    Test Case 1: Right-aligned column with one deviating cell.

    Simulates the FNB statement scenario where 25/26 cells align to x=500,
    and 1 cell extends to x=512.3 (the "Cr" suffix anomaly).
    """
    print("\n" + "=" * 80)
    print("TEST 1: Right-aligned column with deviation")
    print("=" * 80)

    # Create sample cells
    cluster_cells = []

    # 25 cells aligned to x=500
    for i in range(25):
        cluster_cells.append({
            "bbox": {"x0": 420.0, "y0": 100.0 + i * 15, "x1": 500.0, "y1": 112.0 + i * 15},
            "cell": {"text": f"19,631.85"}
        })

    # 1 deviating cell at x=512.3 (with "Cr" suffix)
    cluster_cells.append({
        "bbox": {"x0": 420.0, "y0": 100.0 + 25 * 15, "x1": 512.3, "y1": 112.0 + 25 * 15},
        "cell": {"text": "19,631.85 Cr"}
    })

    # Define baselines
    left_baselines = [420.0]  # Common left edge
    right_baselines = [500.0]  # Common right edge (most cells)

    # Analyze alignment
    result = analyze_block_alignment(cluster_cells, left_baselines, right_baselines)

    print(f"\nAlignment Type: {result['alignment_type']}")
    print(f"Baseline Value: {result['baseline_value']}")
    print(f"Confidence: {result['confidence']} ({result['aligned_count']}/{result['total_count']} cells)")
    print(f"Deviations: {len(result['deviations'])}")

    if result['deviations']:
        print(f"\nDeviating cells:")
        for dev in result['deviations']:
            print(f"  - \"{dev['cell_text']}\": expected={dev['expected_edge']}, "
                  f"actual={dev['actual_edge']}, deviation={dev['deviation']:+.2f}pt")

    # Expected results
    assert result['alignment_type'] == 'right', f"Expected 'right', got '{result['alignment_type']}'"
    assert result['baseline_value'] == 500.0, f"Expected baseline 500.0, got {result['baseline_value']}"
    assert result['confidence'] >= 0.96, f"Expected confidence ≥0.96 (25/26), got {result['confidence']}"
    assert len(result['deviations']) == 1, f"Expected 1 deviation, got {len(result['deviations'])}"
    assert result['deviations'][0]['deviation'] > 0, "Expected positive deviation"

    print("\n[PASS] TEST 1 PASSED")


def test_perfect_left_aligned_column():
    """
    Test Case 2: Perfectly left-aligned column with no deviations.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Perfect left-aligned column")
    print("=" * 80)

    # Create sample cells - all aligned to left edge at x=72.0
    cluster_cells = []
    for i in range(15):
        cluster_cells.append({
            "bbox": {"x0": 72.0, "y0": 100.0 + i * 15, "x1": 150.0 + i * 5, "y1": 112.0 + i * 15},
            "cell": {"text": f"Item {i+1}"}
        })

    # Define baselines
    left_baselines = [72.0]
    right_baselines = []

    # Analyze alignment
    result = analyze_block_alignment(cluster_cells, left_baselines, right_baselines)

    print(f"\nAlignment Type: {result['alignment_type']}")
    print(f"Baseline Value: {result['baseline_value']}")
    print(f"Confidence: {result['confidence']} ({result['aligned_count']}/{result['total_count']} cells)")
    print(f"Deviations: {len(result['deviations'])}")

    # Expected results
    assert result['alignment_type'] == 'left', f"Expected 'left', got '{result['alignment_type']}'"
    assert result['baseline_value'] == 72.0, f"Expected baseline 72.0, got {result['baseline_value']}"
    assert result['confidence'] == 1.0, f"Expected confidence 1.0, got {result['confidence']}"
    assert len(result['deviations']) == 0, f"Expected 0 deviations, got {len(result['deviations'])}"

    print("\n[PASS] TEST 2 PASSED")


def test_mixed_alignment():
    """
    Test Case 3: Mixed alignment (e.g., table with labels and values).
    """
    print("\n" + "=" * 80)
    print("TEST 3: Mixed alignment")
    print("=" * 80)

    # Create sample cells - mix of left and right aligned
    cluster_cells = []

    # 5 left-aligned cells
    for i in range(5):
        cluster_cells.append({
            "bbox": {"x0": 72.0, "y0": 100.0 + i * 20, "x1": 150.0, "y1": 112.0 + i * 20},
            "cell": {"text": f"Label {i+1}"}
        })

    # 5 right-aligned cells
    for i in range(5):
        cluster_cells.append({
            "bbox": {"x0": 400.0, "y0": 100.0 + i * 20, "x1": 500.0, "y1": 112.0 + i * 20},
            "cell": {"text": f"Value {i+1}"}
        })

    # Define baselines
    left_baselines = [72.0]
    right_baselines = [500.0]

    # Analyze alignment
    result = analyze_block_alignment(cluster_cells, left_baselines, right_baselines)

    print(f"\nAlignment Type: {result['alignment_type']}")
    print(f"Baseline Value: {result['baseline_value']}")
    print(f"Confidence: {result['confidence']} ({result['aligned_count']}/{result['total_count']} cells)")

    # Expected results - should be "mixed" (50% left, 50% right)
    assert result['alignment_type'] == 'mixed', f"Expected 'mixed', got '{result['alignment_type']}'"

    print("\n[PASS] TEST 3 PASSED")


def test_center_aligned():
    """
    Test Case 4: Center-aligned text (no baselines match).
    """
    print("\n" + "=" * 80)
    print("TEST 4: Center-aligned text")
    print("=" * 80)

    # Create sample cells - centered around x=300
    cluster_cells = []
    for i in range(10):
        width = 100.0 + i * 10  # Varying widths
        x0 = 300.0 - width / 2
        x1 = 300.0 + width / 2
        cluster_cells.append({
            "bbox": {"x0": x0, "y0": 100.0 + i * 15, "x1": x1, "y1": 112.0 + i * 15},
            "cell": {"text": f"Centered Text {i+1}"}
        })

    # Define baselines (far from actual edges)
    left_baselines = [72.0]
    right_baselines = [540.0]

    # Analyze alignment
    result = analyze_block_alignment(cluster_cells, left_baselines, right_baselines)

    print(f"\nAlignment Type: {result['alignment_type']}")
    print(f"Baseline Value: {result['baseline_value']}")
    print(f"Confidence: {result['confidence']}")

    # Expected results - should be "center" (no alignment to baselines)
    assert result['alignment_type'] == 'center', f"Expected 'center', got '{result['alignment_type']}'"
    assert result['baseline_value'] is None, f"Expected None baseline, got {result['baseline_value']}"
    assert result['confidence'] == 0.0, f"Expected confidence 0.0, got {result['confidence']}"

    print("\n[PASS] TEST 4 PASSED")


def main():
    print("\n" + "=" * 80)
    print("ALIGNMENT ANALYSIS FUNCTIONS TEST SUITE")
    print("Phase 2 / UC-001 / Increment 1")
    print("=" * 80)

    try:
        test_right_aligned_column_with_deviation()
        test_perfect_left_aligned_column()
        test_mixed_alignment()
        test_center_aligned()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        print("\nAlignment analysis functions are working correctly.")
        print("Ready to proceed to Increment 2: Enhanced Block Metadata\n")

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
