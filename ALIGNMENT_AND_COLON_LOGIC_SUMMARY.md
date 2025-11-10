# Alignment and Colon Spacing Detection System

**Last Updated:** 2025-11-07
**Status:** Implemented, undergoing refinement

This document describes the complete alignment baseline detection and colon spacing analysis system used for document forgery detection.

---

## Table of Contents

1. [Overview](#overview)
2. [Alignment Baseline Detection](#alignment-baseline-detection)
3. [Colon Spacing Analysis](#colon-spacing-analysis)
4. [Integration and Classification](#integration-and-classification)
5. [Font Information Extraction](#font-information-extraction)
6. [Visualization System](#visualization-system)
7. [Future Requirements](#future-requirements)

---

## Overview

The system detects document forgeries by analyzing two key aspects:

1. **Alignment Baselines**: Dominant edges (left/right/top/bottom) where text elements align
2. **Colon Spacing Patterns**: Consistency in label:value pair spacing and alignment

These are combined to detect anomalies that indicate potential tampering or forgery.

---

## Alignment Baseline Detection

### Purpose

Detect dominant alignment edges on a page to establish "expected" alignments for text elements. Deviations from these baselines indicate potential forgery.

### Implementation

**Function:** `detect_alignment_baselines(cells, page_width, page_metadata, page_height)`
**Location:** `pdf_ocr_detector.py`

### Algorithm

#### 2025-11-07 Update — Shared Spatial Grid Foundation

- Introduced `spatial_grid.py`, a reusable uniform grid that powers fast bbox overlap detection, nearest-neighbour discovery, and single-link clustering. Hidden-text checks now call `find_overlapping_pairs_optimized()` which simply wraps the grid’s `overlapping_pairs()` for full coverage without quadratic scans.
- `text_block_detection.identify_text_blocks()` was reworked to run on top of the same grid (vertical `cluster_text()` plus spacing estimation). This ensures that text blocks feeding baseline detection are built from direction-aware clusters rather than ad-hoc gap sorting.
- Block finalization still snaps edges to nearby baselines, but the new clustering inputs yield noticeably more stable purple block overlays and alignment statistics for downstream detectors.

1. **Cell Filtering**
   - Filter out structural elements (headers, section headers) using metadata matching
   - Prevents false baselines from non-content elements

2. **Edge Clustering**
   - **Clustering Precision:** 1pt (items within ±0.5pt grouped together)
   - **Rationale:** Forgiving tolerance accounts for slight OCR/rendering variations
   - Groups cells by their edge coordinates:
     - **Left edges:** `x0` coordinates
     - **Right edges:** `x1` coordinates
     - **Top edges:** `y0` coordinates
     - **Bottom edges:** `y1` coordinates

3. **Minimum Thresholds**
   - **Vertical baselines (left/right):** Minimum 3 items
   - **Horizontal baselines (top/bottom):** Minimum 2 items
   - **Rationale:** Lower threshold for horizontal (table rows may have fewer items)

4. **Proximity Filtering** ✨ NEW
   - **Horizontal baselines only:** If multiple baselines within 5pt, keep highest count
   - **Purpose:** Prevents overlapping lines (e.g., both top and bottom of same row)
   - **Vertical baselines:** No filtering (don't have this issue)

5. **Baseline Metadata**
   Each baseline includes:
   ```python
   {
       "orientation": "left" | "right" | "top" | "bottom",
       "value": float,  # x or y coordinate in PDF points
       "page_position": "page_left" | "page_right" | "page_top" | "page_bottom" | "mid_page",
       "count": int,  # Number of items aligned to this baseline
       "y_values": [float],  # For vertical baselines
       "x_values": [float]   # For horizontal baselines
   }
   ```

### Baseline Colors

**Vertical Baselines:**
- **Left-aligned:** Cyan/Blue
  - Page edge: Deep sky blue (solid)
  - Mid-page: Cyan (semi-transparent)
- **Right-aligned:** Orange/Yellow
  - Page edge: Dark orange (solid)
  - Mid-page: Orange (semi-transparent)

**Horizontal Baselines:**
- **Top-aligned:** Purple
  - Page edge: Dark violet (solid)
  - Mid-page: Medium orchid (semi-transparent)
- **Bottom-aligned:** Brown
  - Page edge: Saddle brown (solid)
  - Mid-page: Brown (semi-transparent)

---

## Colon Spacing Analysis

### Purpose

Detect inconsistencies in label:value pair spacing, which often indicates text has been edited/replaced.

### Implementation

**Function:** `detect_colon_spacing_anomalies(docling_payload)`
**Location:** `pdf_ocr_detector.py`

### Two-Case Pattern Detection

#### Case 1: Multi-Cell Patterns (Separate Cells)
**Example:** `"Name:"` in one cell, `"John Doe"` in another cell

**Analysis Method:** Spacing between cells
```python
spacing = value_bbox['x0'] - label_bbox['x1']
```

**Detection:**
1. Find all label cells ending with `:`
2. Locate value cell (same baseline, to the right, within max distance)
3. Calculate spacing between label's right edge and value's left edge
4. Compare to dominant spacing pattern

#### Case 2: Single-Cell Patterns (Same Cell) ✨ UPDATED
**Example:** `"PERIOD END DATE: 2024/11/27"` in single cell

**Analysis Method:** Alignment to baselines (not spacing)
- **Previous:** These were skipped entirely
- **Current:** Check if cell's left edge (`x0`) aligns to any left baseline

**Classification:**
- ✅ **Aligned to baseline:** Consistent (green) - Example: Long text spanning full width
- ❌ **Not aligned to baseline:** Deviation (red) - Example: Shortened date field

**Rationale:** In single-cell text, spacing is inherent to the text. The key indicator is whether the entire cell aligns with other elements.

### Spacing Pattern Analysis

**Function:** `analyze_colon_spacing(pairs)`

1. **Collect all spacing values** from colon pairs
2. **Bucket at 0.5pt precision** for clustering
3. **Find dominant cluster** (most frequent spacing value)
4. **Return:** Dominant spacing and cluster statistics

### Alignment Detection

**Function:** `detect_aligned_values_using_baselines(pairs, baselines)`

For multi-cell patterns, check if value bboxes align to detected baselines:

```python
# Check left alignment
for baseline_x in left_baselines:
    if abs(value_left_edge - baseline_x) <= TOLERANCE:
        # Value is left-aligned to baseline

# Check right alignment
for baseline_x in right_baselines:
    if abs(value_right_edge - baseline_x) <= TOLERANCE:
        # Value is right-aligned to baseline
```

**Tolerance:** `COLON_RIGHT_ALIGN_TOLERANCE` constant

### Content Type Analysis (NLP-based)

**Function:** `analyze_value_content_type(text)`

Detects value content type to assess suspiciousness:

| Content Type | Suspicion Multiplier | Detection Pattern |
|--------------|---------------------|-------------------|
| **Name** | 3.0x (VERY HIGH) | Title case, 2-5 words, mostly alphabetic |
| **Currency** | 2.5x (VERY HIGH) | $1,234.56, R123.45, €100.00, etc. |
| **Number/ID** | 2.0x (HIGH) | Pure digits, ID patterns (12-345-67) |
| **Date** | 1.0x (NORMAL) | YYYY-MM-DD, DD/MM/YYYY |
| **Text** | 1.0x (NORMAL) | Everything else |

**Usage:** When a deviation is detected, suspicion level affects warning message:
- **Multiplier ≥ 2.5:** "⚠️ HIGHLY SUSPICIOUS: [content type] misaligned!"
- **Multiplier > 1.0:** "⚠️ SUSPICIOUS: [content type] misaligned!"
- **Multiplier = 1.0:** Standard deviation message

---

## Integration and Classification

### Classification Priority

For multi-cell patterns:
1. **Left-aligned to baseline** → Consistent (green)
2. **Right-aligned to baseline** → Right-aligned (orange)
3. **Within spacing tolerance** → Consistent (green)
4. **Outside tolerance** → Deviation (red)

For single-cell patterns:
1. **Aligned to left baseline** → Consistent (green)
2. **Not aligned** → Deviation (red)

### Constants

```python
COLON_SPACING_TOLERANCE = 2.0  # pt - acceptable deviation from dominant spacing
COLON_RIGHT_ALIGN_TOLERANCE = 1.5  # pt - tolerance for baseline alignment
COLON_BASELINE_TOLERANCE = 2.0  # pt - same baseline detection
COLON_MAX_DISTANCE = 200.0  # pt - max distance between label and value
COLON_MIN_CLUSTER_SIZE = 3  # Minimum pairs to establish pattern
```

### Anomaly Data Structure

Each detected anomaly includes:

```python
{
    'page': int,
    'label': str,
    'value': str,
    'bbox': dict,  # Value bounding box
    'label_bbox': dict,
    'classification': 'consistent' | 'deviation' | 'right_aligned',
    'spacing': float,
    'dominant_spacing': float,
    'deviation': float,
    'reason': str,  # Human-readable explanation
    'content_type': str,  # 'name', 'currency', 'number', etc.
    'suspicion_multiplier': float,
    'is_same_cell': bool,
    'font_info': dict | None  # Font metadata
}
```

---

## Font Information Extraction

### Purpose

Provide font details for flagged items to help identify tampering (different fonts often indicate edits).

### Implementation

**Pipeline:**

1. **Extraction** (`detect_colon_spacing_anomalies`)
   ```python
   font_info = {
       'font': cell.get('font'),
       'font_size': cell.get('font_size'),
       'font_name': cell.get('font_name'),
       'font_family': cell.get('font_family')
   }
   ```

2. **Propagation** (`extract_colon_pairs`)
   - Attached to each pair dictionary
   - Passed through to anomaly data

3. **Display** (Streamlit UI)
   - Shows in "Highlighted Items on Page" section
   - Only for red/orange flagged items
   - Format: `Font: Name: Arial, Family: sans-serif, Size: 12pt`

---

## Visualization System

### Document Viewer Overlays

**Baseline Lines:**
- Drawn across entire page (vertical or horizontal)
- Color-coded by orientation and position
- Labeled with coordinate and count: `"L: 220.5pt (6)"`

**Colon Spacing Highlights:**
- **Green:** Consistent with pattern or aligned to baseline
- **Red:** Deviation from pattern (suspicious)
- **Orange:** Right-aligned exception

### Annotation Details Display

**Location:** Below document viewer
**Content:** Only red and orange items (suspicious only)

Each item shows:
- Text value
- Highlight color
- **Font information** (if available)
- Detailed reasons with explanations

### Debug Information

**Expander:** "🔍 Baseline Detection Debug Info"

Shows:
- Total cells and filtered cells count
- Baselines found count
- **Cluster statistics:** Top 10-20 clusters per edge type with counts
- **Rejected baselines:** Clusters below threshold with reasons
- **Sample cells:** First 30 cells with coordinates

**Purpose:** Helps diagnose why certain baselines aren't detected

### Alignment Logic Reports

- Every alignment run now emits a document-scoped JSON report under `Document Pipeline/Alignment logic/`.
- Filename pattern: `<pdf_stem>_alignment_logic.json`.
- Each report captures:
  - Document metadata (PDF path, Docling payload path, generation timestamp).
  - Per-page baseline stages (initial + block-refined) with cluster stats, rejected clusters, and coverage metrics.
  - Text-block clustering diagnostics (split decisions, reasons for accepted/rejected clusters, final block summaries).
  - Colon spacing analysis with dominant spacing, every colon pair's classification, and reasons for passes/failures.
  - Events explaining when data was missing (e.g., “no cells on page”, “insufficient colon pairs”).
- Use these files when asking Codex/Claude to explain why elements passed or failed—they contain both successes and failures.

---

## Future Requirements

### 0. Baseline & Text-Block Refinement Roadmap (2025-11-07)

1. **Tighter block cohesion**
   - After the initial vertical `cluster_text()` pass, re-run the grid in horizontal mode inside each cluster to split accidental multi-column clusters.
   - Reject clusters whose width variance or dominant-orientation share indicates they are mixtures; keep noisy regions from influencing downstream steps.
2. **Block-driven baseline seeding**
   - Promote finalized blocks to primary anchors (weight by line_count or bbox height) before considering any raw cell edges.
   - Allow raw cells to extend an anchor only if they land within ≈2 pt of that block-aligned edge; ignore isolated cell clusters without block support.
3. **Vertical baseline consolidation & coverage checks**
   - Merge left/right baselines that sit within 3 pt of one another when their supporting blocks overlap vertically by ≥30%.
   - Track coverage (percent of page height touched by aligned blocks) and discard any non-edge baseline whose coverage drops below 20%.
4. **Block ↔ baseline feedback loop**
   - After baseline consolidation, resnap block boxes and flag candidates that would need >5 pt movement; this keeps the two systems synchronized and highlights genuine anomalies.
5. **Instrumentation & visualization**
   - Surface per-baseline diagnostics (origin = block vs. raw, coverage, count) and emit viewer warnings when we exceed configured limits (e.g., >6 left baselines or average coverage <25%).

### 1. Horizontal Baseline-Based Text Alignment Detection

**Requirement ID:** FR-2025-11-06-01  
**Status:** Implemented (Version 4) - VERY GOOD precision on target documents

#### Problem

Text elements that do not align to horizontal baselines (table rows) should be flagged as suspicious.

#### Implemented Solution

**Function:** `detect_horizontal_alignment_anomalies(cells, baselines, page_num, page_metadata)`  
**Location:** `pdf_ocr_detector.py:2386-2608`

#### Current Behaviour (Version 4 – 2025-11-07)

1. **Baseline intrusion scoring**
   - Works only with well-supported bottom baselines (count ≥ 4)
   - Baseline must sit between 30%-70% of the text height and remain nearly symmetric (≤1.75pt difference)
   - Rejects candidates that have another baseline within 4pt below the text

2. **Segment aggregation**
   - Groups neighbours that share the same intruding baseline (≤14pt gaps)
   - Emits a single anomaly per block, e.g. `11 ALICE LN SANDHURST`
   - Stores combined bounding box plus component details for UI use

3. **NLP counter evidence**
   - Applies `analyze_value_content_type()` and a common-English lexicon
   - Generic phrases (e.g. "Opening balance") receive counter evidence:
     - Adds "Counter Evidence: Text unlikely to have been forged" → "Not a name/date/currency/number"
     - Highlights the item in light orange in the viewer instead of red
   - Proper nouns, city/streets, dates, currency, and numeric IDs stay high-risk (bright red)

4. **Key thresholds**
```python
MIN_BASELINE_COUNT = 4
MIN_TOP_OFFSET = 2.5
MIN_BOTTOM_OFFSET = 2.5
MIN_OFFSET_RATIO = 0.30
MAX_OFFSET_RATIO = 0.70
MAX_OFFSET_DIFF = 1.75
MIN_GAP_BELOW = 4.0
MAX_GROUP_GAP = 14.0
MIN_GROUP_CHAR_COUNT = 5
```

#### Output Format

**Visualization:**
- Red highlight = high-risk misalignment (names, addresses, dates, numbers, currency)
- Light orange highlight = low-risk/common words (with counter evidence)
- Items appear in "Highlighted item(s) on this page" with geometry + font data
- Counter evidence entry is appended for downgraded items

**Report Example:**
```
Item: "11 ALICE LN SANDHURST"
Font: Name: Arial, Size: 10pt
Type: Horizontal Alignment: MISALIGNMENT (Suspicious)
Details: Item at x0=150.3pt, expected baseline at x0=138.0pt
         Deviation: 12.3pt from dominant left edge
         Other items on same row are aligned to x0=138.0pt
```

#### Tuning History

- **Version 4 (current):** Baseline intrusion + NLP counter evidence → ONLY flags `11 ALICE LN SANDHURST` and `SANDTON`; adds light-orange highlights and counter evidence for phrases such as "Opening balance".
- **Version 3:** Baseline-through-middle + far-from-left heuristics — increased noise.
- **Version 2:** Threshold tweaks on Version 1 — still noisy.
- **Version 1:** Simple left baseline comparison — too many legitimate hits.

#### Recommended Next Steps

1. Build a regression harness covering both forged (expect red highlights) and legitimate (expect none or light orange) documents.
2. Expand the `COMMON_ENGLISH_WORDS` lexicon as new templates surface.
3. Surface numeric confidence metrics (e.g. baseline symmetry gap) inside the UI for analyst triage.

### 2. Text Block Grouping & Spacing Statistics

**Requirement ID:** FR-2025-11-09-01  
**Status:** Implemented (Initial release)

#### Purpose

Provide higher-level structural groupings to refine baseline detection and expose spacing anomalies inside coherent blocks of text.

#### Implementation Summary

- **Function:** `identify_text_blocks(cells, baselines)`  
  **Location:** `pdf_ocr_detector.py:2627-2854`
- **Inputs:** Consolidated Docling cells + detected vertical baselines.
- **Algorithm:**
  1. Classify each cell as left/right/center aligned using the closest baseline (≤3 pt tolerance).
  2. Cluster cells into lines by grouping y-centres within 1.5 pt (`LINE_CLUSTER_TOLERANCE`).
  3. Build blocks by chaining consecutive lines where the gap ≤ median page gap × 1.75, with safeguards for orientation changes.
  4. Each block stores:
     - Bounding box (used for violet overlay in the viewer)
     - Line count, per-block spacing values, mean, and 0.5 pt-rounded mode
     - Dominant alignment (`left`, `right`, `center`, or `mixed`)
     - Median supporting baseline value when available
- **Storage:** Blocks saved per page in `docling_payload["_text_blocks"]`.
- **Viewer Overlay:** Violet rectangles with light fill; details appear in “Highlighted item(s)” including spacing stats.

#### Next Steps

1. Monitor spacing distributions on additional documents to calibrate block-gap thresholds.
2. Feed block-level stats back into horizontal baseline tuning (e.g., suppress misalignment alerts when spacing mode is stable).

### 2a. Two-Stage Clustering for Text Block Detection

**Requirement ID:** FR-2025-11-07-01
**Status:** Implemented (Not perfect yet, under refinement)

#### Problem

The initial anisotropic DBSCAN approach (with horizontal scaling) produced blocks with better horizontal extent but failed to separate left/middle/right columns in some document sections, creating a single large block instead of distinct column blocks.

#### Solution: Two-Stage Clustering

A hierarchical approach that clusters in two dimensions separately:

1. **Stage 1 (Vertical):** Cluster cells by y-coordinates to find horizontal "bands" or rows
2. **Stage 2 (Horizontal):** Within each vertical band, cluster by x-coordinates to separate columns

This approach naturally creates the horizontal bands first, then splits them into columns, which better matches the document structure.

#### Implementation

**Function:** `identify_text_blocks_two_stage(cells, baselines, vertical_method, horizontal_method, vertical_eps, horizontal_eps, min_samples, debug)`
**Location:** `text_block_detection.py:835-1092`

**Algorithm:**

1. **Vertical Clustering (Stage 1)**
   - Extract y-coordinates (center_y) from all cells
   - Apply clustering method (DBSCAN or Agglomerative) with `vertical_eps` threshold
   - Creates horizontal "bands" of text at similar vertical positions
   - Noise points (DBSCAN label=-1) are filtered out

2. **Horizontal Clustering (Stage 2)**
   - For each vertical cluster from stage 1
   - Extract x-coordinates (center_x) from cells in that cluster
   - Apply clustering method with `horizontal_eps` threshold
   - Splits each horizontal band into left/middle/right column blocks
   - Each final block gets unique block_id

3. **Block Finalization**
   - Calculate bounding box for each final block
   - Determine dominant alignment using baseline proximity (≤3pt tolerance)
   - Count lines and extract text samples
   - Store spacing statistics

**Method Options:**
- **DBSCAN:** Better noise handling, density-based clustering
- **Agglomerative:** Deterministic, hierarchical clustering using single linkage

#### Smoke Test Results

**Test Coverage:** 24 combinations tested
- 4 method pairs: DBSCAN+DBSCAN, DBSCAN+Agglomerative, Agglomerative+DBSCAN, Agglomerative+Agglomerative
- Parameter variations: v_eps=[10.0, 12.0], h_eps=[30.0, 50.0, 70.0], min_samples=2

**Key Findings:**

| v_eps | h_eps | Method Pair | Vert Clusters | Final Blocks |
|-------|-------|-------------|---------------|--------------|
| 12.0  | 70.0  | dbscan+dbscan | 9 | 9 |
| 12.0  | 70.0  | dbscan+agglomerative | 9 | 9 |
| 12.0  | 70.0  | agglomerative+dbscan | 10 | 9 |
| 12.0  | 70.0  | agglomerative+agglomerative | 10 | 9 |
| 12.0  | 50.0  | (all methods) | 9-10 | 11 |
| 12.0  | 30.0  | (all methods) | 9-10 | 14 |
| 10.0  | 70.0  | (all methods) | 9-11 | 9-10 |

**Critical Discovery:** Method choice has minimal impact on results. Threshold values (especially `h_eps`) are the dominant factor.

**Optimal Parameters:**
```python
TWO_STAGE_VERTICAL_EPS = 12.0      # pt - vertical clustering threshold
TWO_STAGE_HORIZONTAL_EPS = 70.0    # pt - horizontal clustering threshold
TWO_STAGE_MIN_SAMPLES = 2          # minimum samples for DBSCAN
TWO_STAGE_VERTICAL_METHOD = "dbscan"       # Better noise handling
TWO_STAGE_HORIZONTAL_METHOD = "agglomerative"  # Deterministic
```

**Test Document:** "87026407408 – Mr I Fakude - False pay slip"
**Target:** 9 blocks with proper column separation

Sample blocks produced:
- Block with 4 lines: "HENCO", "MINERALS (PTY) LID", "EMP", "CODE:", "HM20927"
- Block with 9 lines: "SALARY", "SLIP", "PERIOD", "END", "DATE:2024/11/27"
- Block with 6 lines: "EMP", "CODE", "HM20927", "DEPARTMENT:", "MINING"

#### Configuration

**Constants in text_block_detection.py:**
```python
USE_TWO_STAGE = True   # Enable two-stage clustering
USE_DBSCAN = False     # Disable single-stage DBSCAN
```

**Integration in pdf_ocr_detector.py (lines 2273-2278):**
```python
if USE_TWO_STAGE:
    text_blocks = identify_text_blocks_two_stage(consolidated_cells, baselines, debug=final_block_debug)
elif USE_DBSCAN:
    text_blocks = identify_text_blocks_dbscan(consolidated_cells, baselines, debug=final_block_debug)
else:
    text_blocks = identify_text_blocks(consolidated_cells, baselines, debug=final_block_debug)
```

#### Output Structure

Each block includes:
```python
{
    'block_id': str,              # e.g., "block_0_0", "block_1_2"
    'bbox': dict,                 # Combined bounding box
    'line_count': int,            # Number of text lines
    'dominant_alignment': str,    # 'left', 'right', 'center', 'mixed'
    'texts': [str],              # Sample text content
    'vertical_cluster_id': int,   # Stage 1 cluster ID
    'horizontal_cluster_id': int  # Stage 2 cluster ID
}
```

#### Debug Output

The `debug` dictionary captures:
```python
{
    'inputs': {
        'total_cells': int,
        'vertical_clusters_found': int,  # Stage 1 output
        'horizontal_splits': int          # Stage 2 splits
    },
    'vertical_clusters': [
        {
            'cluster_id': int,
            'cell_count': int,
            'line_count': int,
            'texts': [str]
        }
    ],
    'blocks': [...]  # Final block details
}
```

#### Current Status

- **Implemented:** 2025-11-07
- **Smoke tested:** 24 parameter combinations
- **Optimal parameters:** Selected and configured
- **User feedback:** "Not perfect yet" - produces proper column separation but may need fine-tuning for edge cases

#### Comparison to Previous Approaches

| Approach | Strength | Weakness |
|----------|----------|----------|
| **Original `identify_text_blocks()`** | Simple, fast | Limited column separation |
| **Anisotropic DBSCAN** | Good horizontal extent | Single large block in some sections |
| **Two-Stage Clustering** | Natural band+column structure | Requires two passes, needs tuning |

#### Next Steps

1. Test on additional documents with varying layouts (multi-column, mixed orientations)
2. Consider adaptive threshold selection based on page dimensions
3. Add fallback logic for edge cases (very sparse pages, unusual layouts)
4. Evaluate hybrid approaches (e.g., two-stage for dense pages, anisotropic for sparse)

### 2b. Iterative Multi-Pass Clustering (Planned)

**Requirement ID:** FR-2025-11-08-01
**Status:** Planned (Not yet implemented)

#### Problem

Current two-stage clustering (v_eps=12.0, h_eps=70.0) still has two issues visible in test documents:

1. **Large Bottom Block:** One large purple block at bottom that should split into separate left/middle/right column blocks
   - Current h_eps=70.0 is too large for horizontal separation
   - Need tighter horizontal clustering threshold

2. **Missing Vertical Split:** "SALARY SLIP" and "6th FLOOR ROWMANS BUILDING" grouped together
   - Initial vertical pass groups them in same band
   - Need iterative refinement to catch vertical splits that emerge after horizontal splits

#### Proposed Solution: Iterative Dimensional Clustering

**Algorithm Overview:**

Current (Two-Stage):
```
Stage 1: Vertical clustering → horizontal bands
Stage 2: Horizontal clustering → column separation
DONE
```

Proposed (Iterative Multi-Pass):
```
Pass 1: Vertical clustering → horizontal bands
Pass 2: Horizontal clustering → column separation
Pass 3: Vertical clustering → sub-band separation
Pass 4: Horizontal clustering → sub-column separation
...
Continue until NO NEW SPLITS occur
```

**Key Concept:** Alternating vertical and horizontal passes allows each dimension to refine the clustering based on the previous dimension's results. This naturally discovers multi-level structure that emerges progressively.

#### Implementation Plan

**Function Signature:**
```python
def identify_text_blocks_iterative(
    cells: List[Dict],
    baselines: List[Dict],
    vertical_eps: float = 12.0,
    horizontal_eps: float = 40.0,  # REDUCED from 70.0
    min_samples: int = 2,
    max_iterations: int = 10,       # Safety limit
    min_block_cells: int = 2,       # Minimum cells per block
    debug: Optional[Dict] = None
) -> List[Dict]:
```

**Algorithm Steps:**

1. **Initialize:** Start with all cells as one "block"

2. **Iterate:**
   - Determine dimension: Odd iterations = vertical, Even = horizontal
   - For each current block:
     - Extract coordinates for clustering dimension (y for vertical, x for horizontal)
     - Apply DBSCAN clustering with appropriate eps
     - If clustering produces multiple sub-blocks → split occurred
     - Otherwise → keep original block
   - Track whether ANY splits occurred this iteration

3. **Stop when:**
   - No splits occurred in last iteration (converged), OR
   - Maximum iterations reached (safety), OR
   - All blocks below minimum size

4. **Finalize:** Convert blocks to final format with bboxes, alignment, spacing stats

**Stopping Criteria:**
- **Primary:** No new splits occurred in last pass (natural convergence)
- **Secondary:** Maximum iterations reached (safety: 10 iterations)
- **Tertiary:** All blocks below minimum size threshold

**Alternating Dimensions:**
```python
if iteration % 2 == 1:  # Odd iterations (1, 3, 5...)
    dimension = "vertical"   # Cluster by y-coordinate
else:                        # Even iterations (2, 4, 6...)
    dimension = "horizontal" # Cluster by x-coordinate
```

#### Proposed Parameters

```python
# Iterative clustering configuration
USE_ITERATIVE = True        # NEW - Use iterative multi-pass
USE_TWO_STAGE = False       # Disable two-stage
USE_DBSCAN = False          # Disable single-stage

# Parameters
ITERATIVE_VERTICAL_EPS = 12.0      # pt - keep current vertical threshold
ITERATIVE_HORIZONTAL_EPS = 40.0    # pt - REDUCED from 70.0 to 40.0
ITERATIVE_MIN_SAMPLES = 2          # DBSCAN minimum samples
ITERATIVE_MAX_ITERATIONS = 10      # Safety limit (typical: 3-5)
ITERATIVE_MIN_BLOCK_CELLS = 2      # Don't split blocks with <2 cells

# Method selection
ITERATIVE_METHOD = "dbscan"  # Use DBSCAN for both dimensions
```

**Why reduce h_eps to 40.0?**
- Smoke test showed h_eps=30.0 produces 14 blocks (likely too many)
- h_eps=50.0 produces 11 blocks
- h_eps=70.0 produces 9 blocks (current, insufficient splits)
- **Target:** h_eps=40.0 as middle ground for better column separation

#### Expected Convergence Behavior

**Example for "Mr I Fakude" document:**

| Pass | Dimension | Blocks Before | Blocks After | Splits? | Action |
|------|-----------|---------------|--------------|---------|--------|
| 1 | Vertical | 1 | 9 | Yes | Initial horizontal bands |
| 2 | Horizontal | 9 | 14 | Yes | Split bands into columns |
| 3 | Vertical | 14 | 15 | Yes | Split "SALARY SLIP" from address |
| 4 | Horizontal | 15 | 15 | No | **CONVERGED** |

**Final Result:** 15 blocks (vs current 9 blocks)

#### Expected Outcomes

**Issue #1 Fix: Bottom Block Split**
- **Before:** 1 large purple block at bottom
- **After:** 3-4 separate blocks (left column, middle column, right column)
- **Mechanism:** Reduced h_eps=40.0 + iterative refinement forces tighter horizontal clustering

**Issue #2 Fix: Vertical Refinement**
- **Before:** "SALARY SLIP" grouped with "6th FLOOR ROWMANS BUILDING"
- **After:** Separate blocks
- **Mechanism:**
  - Pass 1 (Vertical): Creates initial bands
  - Pass 2 (Horizontal): Splits columns
  - Pass 3 (Vertical): Re-evaluates vertical separation within columns → splits apart
  - Pass 4 (Horizontal): No new splits → STOP

#### Noise Handling Strategy

DBSCAN produces noise points (label=-1). Proposed handling:

**Option A (Conservative - Recommended):**
- Keep noise points in original block
- Apply minimum block size filter (min_block_cells=2)
- Prevents over-fragmentation

**Option B (Aggressive):**
- Create singleton blocks for noise points
- May cause too much fragmentation

**Recommendation:** Use Option A

#### Debug Output Structure

```python
debug = {
    "approach": "iterative",
    "inputs": {
        "cell_count": int,
        "vertical_eps": float,
        "horizontal_eps": float,
        "max_iterations": int
    },
    "iterations": [
        {
            "iteration": int,
            "dimension": "vertical" | "horizontal",
            "blocks_before": int,
            "blocks_after": int,
            "splits_occurred": bool,
            "blocks_split": int  # How many blocks were split this iteration
        }
    ],
    "convergence": {
        "converged": bool,
        "iterations_used": int,
        "final_blocks": int,
        "convergence_reason": "no_splits" | "max_iterations" | "min_size"
    },
    "blocks": [...]  # Final block details
}
```

#### Risk Mitigation

**Risk 1: Over-Fragmentation**
- Mitigation: `min_block_cells` threshold prevents single-cell blocks
- Monitor noise point handling
- Track iteration count in debug output

**Risk 2: Infinite Loops**
- Mitigation: Hard limit `max_iterations = 10`
- Explicit convergence check: `if not splits_occurred: break`
- Debug logging of each iteration

**Risk 3: Performance**
- Mitigation: Early stopping on convergence
- Typical convergence: 3-5 iterations
- Worst case: O(max_iterations × N log N) for N cells
- Expected: ~10-15ms for typical document (130 cells)

#### Implementation Phases

**Phase 1: Parameter Tuning (Quick Win)**
- Goal: Fix Issue #1 without algorithm changes
- Action: Reduce TWO_STAGE_HORIZONTAL_EPS from 70.0 → 40.0
- Change TWO_STAGE_HORIZONTAL_METHOD from "agglomerative" → "dbscan"
- Test: Run smoke test with h_eps=[35.0, 40.0, 45.0]
- Time: 30 minutes

**Phase 2: Iterative Algorithm (Full Solution)**
- Goal: Fix both Issues #1 and #2
- Action: Implement `identify_text_blocks_iterative()` function
- Location: `text_block_detection.py` (after line 1092)
- Create: `iterative_smoke_test.py` for parameter exploration
- Time: 2-3 hours

**Phase 3: Testing & Validation**
- Goal: Validate convergence and block quality
- Action: Run comprehensive parameter tests
- Test combinations:
  - vert_eps: [10.0, 12.0, 15.0]
  - horiz_eps: [35.0, 40.0, 45.0, 50.0]
  - max_iterations: [5, 10, 15]
- Time: 1 hour

**Total Estimated Time:** 3.5-4.5 hours

#### Files to Modify

**New Files:**
1. `iterative_smoke_test.py` - Parameter exploration script

**Modified Files:**
1. `text_block_detection.py`
   - Add `identify_text_blocks_iterative()` function
   - Add ITERATIVE_* configuration constants
   - Update mode selection flags

2. `pdf_ocr_detector.py`
   - Add `USE_ITERATIVE` conditional branch
   - Import iterative function

3. `ALIGNMENT_AND_COLON_LOGIC_SUMMARY.md`
   - Update this section with actual results
   - Document convergence behavior
   - Update comparison table

#### Comparison to Other Approaches

| Approach | Passes | Strength | Weakness | Expected Blocks |
|----------|--------|----------|----------|-----------------|
| **Original** | 1 | Simple, fast | Limited separation | 7-8 |
| **Anisotropic DBSCAN** | 1 | Good horizontal extent | Single large blocks | 8-9 |
| **Two-Stage** | 2 | Natural band+column | Fixed passes, insufficient splits | 9 |
| **Iterative** | 3-5 | Adaptive refinement, finds multi-level structure | More complex, needs tuning | 15+ |

#### Success Criteria

1. ✅ Bottom section splits into 3+ column blocks (not one large block)
2. ✅ "SALARY SLIP" isolated from address information below it
3. ✅ No over-fragmentation (no single-cell blocks)
4. ✅ Convergence occurs within 5 iterations (efficient)
5. ✅ Total blocks: 12-15 (increased from current 9)
6. ✅ Visual inspection: Purple blocks match logical document structure

#### Eliminated Blocks Tracking

**Implemented:** 2025-11-08

The iterative clustering now tracks all candidate blocks that were eliminated during finalization:

```python
debug["eliminated_blocks"] = [
    {
        "candidate_id": int,
        "cell_count": int,
        "line_count": int,
        "bbox": dict,
        "texts": [str],
        "elimination_reason": str  # e.g., "too_few_cells (< 2)", "too_few_lines (< 2)"
    }
]

debug["summary"]["statistics"] = {
    "candidate_blocks": 25,      # After final iteration
    "eliminated_blocks": 12,      # Filtered out
    "final_blocks_created": 13    # Accepted blocks
}
```

**Example eliminated blocks:**
- Header elements: "HENCO MINERALS (PTY) LID" (single line)
- Table headers: "EARNINGS", "AMOUNT", "DEDUCTIONS" (single line)
- Labels: "SALARY SLIP", "HOURS UNITS" (single line)

**Rationale:**
Eliminated blocks may be relevant for subsequent refinement steps:
- Some might need to be merged with final blocks
- Some might need to be split into separate blocks
- Information helps inform vertical baseline decisions

#### Next Steps

1. **Decision:** Start with Phase 1 (quick parameter tuning) or Phase 2 (full iterative implementation)?
2. **Parameter Selection:** Confirm h_eps target value (suggested: 40.0)
3. **Noise Handling:** Confirm Option A (conservative) vs Option B (aggressive)
4. **Testing:** Prepare test document set for validation

### 2c. Block-Baseline Refinement Logic (Future Requirement)

**Requirement ID:** FR-2025-11-08-02
**Status:** Planned (Not yet implemented)

#### Problem

Current implementation produces decent results but has opportunities for refinement:

1. **Block Refinement Needs:**
   - Some blocks may need to be merged (e.g., split header elements)
   - Some blocks may need to be separated (e.g., incorrectly grouped content)
   - Eliminated blocks (stored in debug output) may be relevant for these decisions

2. **Baseline Refinement Needs:**
   - Sometimes too many vertical baselines are detected
   - Need logic to determine which baselines to retain and which to drop
   - Block structure should inform baseline retention decisions

3. **Circular Dependency:**
   - Block decisions depend on baseline information (alignment, snapping)
   - Baseline decisions depend on block information (coverage, support)
   - Need iterative refinement to resolve this

#### Proposed Solution: Iterative Block-Baseline Refinement

**High-Level Algorithm:**

```
1. Run initial clustering (iterative or two-stage) → candidate blocks
2. Run baseline detection → candidate baselines
3. Refinement loop:
   a. Score each baseline by block support:
      - How many blocks align to it?
      - What percentage of page height covered?
      - Are blocks that align to it coherent?
   b. Filter baselines:
      - Keep baselines with strong block support
      - Drop baselines with weak/no block support
      - Merge baselines that are very close (< 3pt) with overlapping vertical coverage
   c. Re-evaluate blocks using refined baselines:
      - Re-snap block edges to updated baselines
      - Check if any blocks should merge (shared baseline, close proximity)
      - Check if any blocks should split (mixed alignments, gaps)
   d. If blocks or baselines changed → continue refinement
   e. If stable → converged
4. Return final blocks and baselines
```

**Baseline Scoring Metrics:**

```python
baseline_score = {
    "value": float,                    # x-coordinate
    "orientation": "left" | "right",
    "supporting_blocks": [block_id],   # Blocks aligned to this baseline
    "coverage_percentage": float,      # % of page height covered
    "avg_block_quality": float,        # Average line count or cell count
    "is_page_edge": bool,             # Near page margin
    "merge_candidates": [baseline_id], # Other baselines within 3pt
}
```

**Block Merge Criteria:**

Two blocks should be merged if:
1. Share the same dominant baseline (left or right)
2. Vertical gap between blocks < median_spacing × 2.0
3. No significant horizontal overlap with different alignment
4. Combined block meets minimum line count threshold

**Block Split Criteria:**

A block should be split if:
1. Contains cells with mixed alignments (left + right) spanning > 20% width difference
2. Contains vertical gap > median_spacing × 3.0 within block
3. Contains distinct sub-groups with different baselines

**Eliminated Block Recovery:**

```python
for eliminated_block in debug["eliminated_blocks"]:
    # Check if should merge with adjacent final block
    for final_block in final_blocks:
        if should_merge(eliminated_block, final_block):
            merge_blocks(eliminated_block, final_block)
            break
```

**Use Cases:**

1. **Header Consolidation:**
   - Eliminated: "HENCO" (1 line)
   - Eliminated: "MINERALS (PTY) LID" (1 line)
   - Adjacent final block: Company info
   - Action: Merge all three into single header block

2. **Baseline Pruning:**
   - Detected: 15 left baselines
   - Blocks support: 3 strong baselines (10+ blocks each)
   - Weak baselines: 12 baselines (1-2 blocks each)
   - Action: Keep 3 strong, drop 12 weak

3. **Block Splitting:**
   - Block has mixed left/right aligned content
   - Left portion: Labels at x=50pt
   - Right portion: Values at x=400pt
   - Action: Split into two blocks

#### Integration Points

**Input:**
- `final_blocks` from iterative clustering
- `eliminated_blocks` from debug output
- `baselines` from alignment detection
- Page metadata (width, height, margins)

**Output:**
- `refined_blocks` with updated bboxes and alignments
- `refined_baselines` with scores and support info
- `merge_log` documenting what was merged/split
- `baseline_filter_log` documenting retained/dropped baselines

**Performance:**
- Refinement loop typically converges in 2-3 iterations
- O(N × M) where N=blocks, M=baselines
- Expected: < 50ms for typical page

#### Implementation Priority

**Priority:** Medium-High
- Current results are "decent" but not perfect
- Refinement would improve accuracy
- Especially important for complex multi-column documents
- Can be implemented incrementally (baseline filtering first, then block merging)

**Dependencies:**
- Requires eliminated blocks tracking (✅ Implemented)
- Requires block-baseline association metadata (✅ Available in current implementation)
- Requires comprehensive test suite to validate refinements

### 3. Multi-Line Address Detection

**Requirement ID:** FR-2025-11-06-02
**Status:** Pending Implementation

#### Problem

When addresses span multiple lines, all lines should align consistently. Misalignment indicates potential tampering.

#### Proposed Solution

1. **Identify address patterns**
   - Multi-line text blocks
   - Consecutive items with similar x0 (left-aligned)
   - Related content (street → city → postal code)

2. **Check vertical alignment**
   - All lines should have same x0 (left edge)
   - Tolerance: 1-2pt

3. **Flag inconsistencies**
   - If one line has different x0 → suspicious

---

## Implementation Notes

### General Principles

1. **No Document-Specific Logic**
   - All thresholds and patterns are generalized
   - Works across different document types

2. **Baseline-Driven Detection**
   - Alignment to detected baselines is primary indicator
   - Spacing analysis is secondary

3. **Conservative Flagging**
   - Only flag clear anomalies
   - Use suspicion multipliers for high-value content (names, currency)

4. **Comprehensive Debugging**
   - All detection steps logged to debug info
   - Sample data provided for investigation

### Performance Considerations

- **Clustering precision:** 1pt balances accuracy and performance
- **Proximity filtering:** Reduces visual clutter without losing information
- **Minimum thresholds:** Prevent false positives from isolated elements

---

## Testing Recommendations

### Test Cases

1. **Legitimate Documents**
   - Should show mostly green highlights
   - Baselines should align with visible structure

2. **Tampered Documents**
   - Edited values should show as red (deviation)
   - Misaligned text should be flagged
   - Font differences should be visible

3. **Edge Cases**
   - Single-cell colon patterns
   - Right-aligned values (amounts, dates)
   - Multi-line addresses
   - Tables with varying alignments

### Validation

- Check baseline coordinates against visual inspection
- Verify spacing deviations match visible irregularities
- Confirm font information is accurate

---

## Changelog

### 2025-11-07: Two-Stage Clustering for Text Block Detection

**Implementation:** New two-stage clustering approach added to address column separation issues
- Created `identify_text_blocks_two_stage()` function in `text_block_detection.py`
- Implements hierarchical clustering: vertical first (y-coordinates), then horizontal (x-coordinates)
- Supports both DBSCAN and Agglomerative Hierarchical methods for each stage
- **Smoke test:** Tested 24 parameter combinations across 4 method pairs
- **Key finding:** Method choice has minimal impact; threshold values are critical
- **Optimal parameters:** v_eps=12.0pt, h_eps=70.0pt produces ~9 blocks with proper column separation
- **Configuration:** Selected DBSCAN (vertical) + Agglomerative (horizontal) for production
- **Status:** Implemented and integrated; produces better column separation than anisotropic DBSCAN
- **User feedback:** "Not perfect yet" - fine-tuning may be needed for edge cases

**Files Modified:**
- `text_block_detection.py`: Added two-stage clustering function and configuration constants
- `pdf_ocr_detector.py`: Integrated two-stage clustering with conditional selection logic
- `two_stage_smoke_test.py`: New smoke test script for parameter exploration

**Previous commits:**
- c5eab6b: "feat: Implement two-stage clustering (vertical then horizontal)"
- 9606782: "fix: Move text_block_detection imports to function start"

### 2025-11-06: Major Update (Part 2) - Horizontal Alignment Detection

**Version 3 (Refined Logic):** ❌ FAILED - Even more false positives
- Completely rewrote detection algorithm with two sophisticated patterns
- Pattern 1: Baseline runs through middle of bbox (not at edges) with no baseline below
- Pattern 2: Text far from baseline (15pt+) but has y-overlap (30%+) with aligned text
- Result: Produced more false positives than previous versions
- Conclusion: Geometric pattern matching alone is insufficient for this problem

**Version 2 (First Tuning):**
- Increased thresholds to reduce false positives
- Still too harsh, produced many false flags

**Version 1 (Initial Implementation):**
- Created `detect_horizontal_alignment_anomalies()` function
- Integrated horizontal alignment detection into main pipeline
- Added UI display for horizontal alignment anomalies in annotation details
- Too strict - flagged many legitimate items

### 2025-11-06: Major Update (Part 1)
- Added horizontal baseline detection (top/bottom)
- Increased clustering tolerance to 1pt
- Implemented proximity filtering for horizontal baselines
- Re-enabled single-cell colon analysis with baseline checks
- Added NLP-based content type detection
- Integrated font information extraction and display
- Enhanced debug information with sample cells

### Previous Versions
- Initial colon spacing detection
- Vertical baseline detection (left/right)
- Basic alignment anomaly reporting

---

## References

**Code Locations:**
- Baseline Detection: `pdf_ocr_detector.py` → `detect_alignment_baselines()`
- Horizontal Alignment: `pdf_ocr_detector.py` → `detect_horizontal_alignment_anomalies()`
- Colon Analysis: `pdf_ocr_detector.py` → `detect_colon_spacing_anomalies()`
- Content Type: `pdf_ocr_detector.py` → `analyze_value_content_type()`
- Visualization: `pdf_ocr_detector.py` → `render_pdf_with_annotations()`

**Related Documents:**
- `HIDDEN_TEXT_LIKE_FOR_LIKE_PLAN.md` - Hidden text detection plan
- `HIDDEN_TEXT_FLATTENING_SEGMENTATION.md` - Text flattening approach

---

**End of Document**
