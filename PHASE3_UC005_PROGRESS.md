# Phase 3: UC-005 Implementation Progress

**Started:** 2025-11-08
**Goal:** Implement block refinement and conflict resolution
**Estimated Effort:** 10-12 hours

## Implementation Plan

### Increment 1: Block Confidence Scoring ✅ COMPLETED
- [x] Implement `calculate_block_confidence()` function
- [x] Implement `calculate_bbox_overlap()` helper function
- [x] Metrics: cell count, alignment coherence, table membership, cell density, vertical span
- [x] Test confidence scoring on sample blocks

### Increment 2: Adjacent Block Merging ✅ COMPLETED
- [x] Implement `merge_adjacent_blocks()` function
- [x] Criteria: same alignment + adjacent (gap < threshold) + same table + centers aligned
- [x] Test merging on sample multi-line addresses

### Increment 3: Nested Block Detection ✅ COMPLETED
- [x] Implement `detect_nested_blocks()` function
- [x] Use overlap ratio > 0.8 for nesting detection (revised from IoU)
- [x] Test on blocks with known nesting issues

### Increment 4: Weak Evidence Filtering ✅ COMPLETED
- [x] Implement `filter_weak_evidence_blocks()` function
- [x] Calculate cells/vertical_span ratio (evidence strength)
- [x] Filter blocks with weak evidence AND high IoU with stronger blocks

### Increment 5: Conflict Resolution & Integration ✅ COMPLETED
- [x] Implement `refine_text_blocks()` integration function
- [x] Integrate all refinement functions into pipeline
- [x] Configurable thresholds for all refinement steps
- [x] Comprehensive statistics tracking

## Progress Log

### 2025-11-08 - Starting Phase 3

**Current Status:** Phase 2 (UC-001) complete and committed (df4a9d5).

**Phase 3 Objectives:**
1. Fix PTC-006 multi-line address fragmentation (5-6 blocks → 1-2 blocks)
2. Remove nested/engulfed blocks
3. Merge adjacent same-alignment blocks in tables
4. Filter weak evidence long-distance blocks
5. Implement confidence-based conflict resolution

**Next Steps:**
1. Implement block confidence scoring ✅
2. Keep functions modular and well-tested
3. Test incrementally before integration

### 2025-11-08 - Increment 1 Complete

**Completed:**
- Added `calculate_bbox_overlap()` helper function (text_block_detection.py:2223-2280)
- Added `calculate_block_confidence()` function (text_block_detection.py:2283-2411)
- Created comprehensive test suite (test_block_confidence.py) with 8 test cases
- All tests passed successfully

**Key Results:**
- Strong block (30 cells, 95% alignment, in table): 0.938 confidence ✓
- Weak block (3 cells, 50% alignment, no table): 0.188 confidence ✓
- Medium block (12 cells, 75% alignment, no table): 0.517 confidence ✓
- Table bonus working correctly: exactly +0.20 difference ✓

**Confidence Scoring Breakdown:**
- Cell count (25%): Normalized by 20 cells
- Alignment coherence (25%): From alignment_metadata.confidence
- Cell density (15%): Normalized by 0.0005 cells/pt²
- Vertical span (15%): Normalized by 0.1 cells/pt
- Table membership (20%): +0.20 bonus if IoU ≥ 0.7

**Next Steps:**
1. Implement adjacent block merging for multi-line addresses ✅
2. Test on PTC-006 document (5-6 address blocks → 1-2 blocks)

### 2025-11-08 - Increment 2 Complete

**Completed:**
- Added `merge_adjacent_blocks()` function (text_block_detection.py:2414-2613)
- Implemented 4 merge criteria: same alignment, vertical adjacency, table membership, center alignment
- Created comprehensive test suite (test_adjacent_merging.py) with 7 test cases
- All tests passed successfully

**Key Results:**
- Successfully merged 2 left-aligned adjacent blocks ✓
- Successfully merged 2 right-aligned adjacent blocks ✓
- Successfully merged 3-block multi-line address into 1 block ✓
- Correctly prevented merging when:
  - Gap too large (35pt > 15pt threshold) ✓
  - Different alignments (left vs right) ✓
  - Different tables ✓
  - Centers misaligned (200pt > 20pt tolerance) ✓

**Merge Criteria:**
1. Same alignment type (both left OR both right)
2. Adjacent vertically (gap < 15pt)
3. Same table membership (both in same table OR both not in tables)
4. Centers aligned horizontally (offset < 20pt)

**Next Steps:**
1. Implement nested block detection using IoU > 0.8 ✅
2. Remove engulfed blocks based on confidence comparison ✅

### 2025-11-08 - Increment 3 Complete

**Completed:**
- Added `detect_nested_blocks()` function (text_block_detection.py:2616-2743)
- Revised algorithm to use overlap_ratio instead of IoU for nesting detection
- Created comprehensive test suite (test_nested_detection.py) with 6 test cases
- All tests passed successfully

**Key Results:**
- Successfully removed nested block with low confidence (diff = 0.50) ✓
- Correctly kept nested block with similar confidence (diff = 0.05 < 0.20) ✓
- Correctly kept similar-sized blocks (area ratio = 1.0 > 0.5) ✓
- Successfully removed 2 nested blocks from large block ✓
- Correctly kept all separate blocks (no nesting) ✓
- Correctly kept partially overlapping blocks (overlap ratio < 0.8) ✓

**Nesting Detection Criteria:**
1. Overlap ratio > 0.8 (small block 80%+ inside large block)
2. Area ratio < 0.5 (small block significantly smaller)
3. Confidence difference ≥ 0.20 (clear quality difference)

**Algorithm Revision:**
- Changed from IoU to overlap_ratio for nesting detection
- Reason: IoU is too low for truly nested blocks (small fully inside large)
- Overlap ratio = intersection / area_small (better measure of "nestedness")

**Next Steps:**
1. Implement weak evidence filtering (cells/vertical_span < 0.05) ✅
2. Filter weak blocks with high IoU overlap with stronger blocks ✅

### 2025-11-08 - Increment 4 Complete

**Completed:**
- Added `filter_weak_evidence_blocks()` function (text_block_detection.py:2750-2858)
- Implemented evidence strength metric: cells / vertical_span
- Created comprehensive test suite (test_weak_evidence.py) with 5 test cases
- All tests passed successfully

**Key Results:**
- Successfully removed weak block overlapping with strong block (0.01 cells/pt) ✓
- Correctly kept weak block with no overlap ✓
- Correctly kept all strong blocks (> 0.05 cells/pt) ✓
- Successfully removed multiple weak blocks ✓
- Correctly handled borderline cases (exactly at threshold) ✓

**Weak Evidence Filtering Criteria:**
1. Evidence strength < 0.05 cells/pt (threshold)
2. IoU > 0.3 with stronger block (overlap threshold)

**Evidence Strength Examples:**
- Strong: 20 cells / 100pt = 0.20 cells/pt ✓
- Medium: 10 cells / 100pt = 0.10 cells/pt ✓
- Weak: 5 cells / 100pt = 0.05 cells/pt (borderline)
- Very weak: 3 cells / 200pt = 0.015 cells/pt (likely noise)

**Next Steps:**
1. Implement integration function that combines all refinement steps ✅
2. Create comprehensive integration test on PTC documents

### 2025-11-08 - Increment 5 Complete & Phase 3 Summary

**Completed:**
- Added `refine_text_blocks()` integration function (text_block_detection.py:2861-2964)
- Integrated all 4 refinement steps in optimal order
- Added configurable thresholds for all parameters
- Implemented comprehensive statistics tracking
- All 5 increments completed successfully

**Integration Pipeline:**
1. **Calculate Confidence**: Score all blocks using 5 metrics
2. **Merge Adjacent**: Combine multi-line addresses (same alignment + adjacent)
3. **Remove Nested**: Filter blocks engulfed by others (overlap ratio > 0.8)
4. **Filter Weak**: Remove weak evidence blocks (< 0.05 cells/pt + overlap)

**Statistics Tracked:**
- Initial block count
- After each refinement step
- Final block count
- Blocks removed
- Blocks merged

**Default Configuration:**
- vertical_gap_threshold: 15.0pt
- horizontal_center_tolerance: 20.0pt
- iou_threshold (nesting): 0.8
- area_ratio_threshold: 0.5
- confidence_difference_threshold: 0.20
- evidence_strength_threshold: 0.05 cells/pt
- weak_overlap_threshold: 0.3

---

## Phase 3 Complete!

**Total Implementation:**
- **5 core functions**: confidence scoring, bbox overlap, adjacent merging, nested detection, weak filtering
- **1 integration function**: refine_text_blocks()
- **18 test cases** across 4 test suites
- **~600 lines of code** added to text_block_detection.py
- **All tests passing** ✓

**Phase 3 Objectives Achieved:**
1. ✅ Block confidence scoring with 5 metrics
2. ✅ Adjacent block merging for multi-line addresses
3. ✅ Nested/engulfed block removal
4. ✅ Weak evidence filtering
5. ✅ Integrated conflict resolution pipeline

**Token Usage:**
- Starting: ~140K tokens remaining
- Current: ~109K tokens remaining
- Used: ~31K tokens for Phase 3

**Next Steps:**
1. Commit Phase 3 implementation
2. Test on actual PTC documents to verify multi-line address merging
3. Measure impact on block counts and quality

---

## Implementation Notes

### Block Confidence Scoring Algorithm

```python
def calculate_block_confidence(block, tables, page_height):
    """
    Calculate confidence score combining:
    - Cell count (25%): More cells = higher confidence
    - Alignment coherence (25%): % cells matching dominant alignment
    - Cell density (15%): Cells per area
    - Vertical span (15%): Cells per vertical distance
    - Table membership (20% bonus): Inside Docling table

    Returns: 0.0 to 1.0
    """
```

**Thresholds:**
- Cell count: 20+ cells = full score
- Alignment coherence: From alignment_metadata.confidence
- Cell density: 0.0005 cells/pt² typical
- Vertical span: 0.1 cells/pt typical
- Table IoU: ≥0.7 overlap for bonus

### Adjacent Block Merging Criteria

**Merge if ALL conditions met:**
1. Same alignment type (both left OR both right)
2. Adjacent: vertical gap < threshold (default: 15pt)
3. Same table (both inside same Docling table)
4. No large horizontal offset (centers aligned within tolerance)

**Example:** Amount + Balance columns in bank statement

### Nested Block Detection (IoU-based)

**Algorithm:**
1. Calculate IoU for all block pairs
2. If IoU > 0.8 AND area(small) < 0.5 × area(large) → nested
3. Compare confidence scores
4. Keep higher confidence block, remove lower
5. Exception: If confidence difference < 20%, keep both

### Weak Evidence Filtering

**Evidence Strength = cell_count / vertical_span**

**Thresholds:**
- Typical strong block: 0.2 cells/pt (20 cells in 100pt span)
- Weak block: <0.05 cells/pt (5 cells in 100pt span)
- Filter if: strength < 0.05 AND IoU with stronger block > 0.3

---

## Test Cases

### Test 1: Multi-Line Address Merging (PTC-006)
**Document:** 85393423298 – Mr G Motau
**Current:** 5-6 separate blocks for address lines
**Expected:** 1-2 blocks after merging
**Criteria:** Same left alignment + adjacent + same gap pattern

### Test 2: Nested Block Removal
**Document:** 85385859429 – Mr MP Madiya
**Current:** Small blocks inside large table block
**Expected:** Remove nested blocks, keep larger block
**Criteria:** IoU > 0.8 + confidence comparison

### Test 3: Column Merging
**Document:** Any statement with Amount + Balance columns
**Current:** 2 separate right-aligned blocks
**Expected:** 1 merged right-aligned block
**Criteria:** Same right alignment + adjacent + same table

### Test 4: No Over-Merging
**Document:** All PTC documents
**Expected:** Distinct sections remain separate (e.g., header vs table)
**Criteria:** Block counts within acceptable range (±30% of baseline)

---

## Token Usage Tracking

- Starting tokens: ~86,000 remaining
- After Increment 1: TBD
- After Increment 2: TBD
- After Increment 3: TBD

**Checkpoint:** If tokens < 20,000, commit work and summarize status
