# Docling Alignment Verification Logic

This Streamlit viewer overlays Docling-derived alignment diagnostics alongside existing hidden-text highlights.

## Inputs
- **Docling payload:** `parsed_pages` supplies word-level boxes, while `docling_document` provides structural metadata (tables, headers).
- **Hidden-text detection:** overlap analysis feeds the same viewer, letting us prioritise forged layers.
- **Thresholds** (configured near the top of `pdf_ocr_detector.py`):
  - `DOCLING_ALIGNMENT_SPACING_THRESHOLD = 3.0` pt
  - `DOCLING_ALIGNMENT_LEFT_THRESHOLD = 0.75` pt
  - `DOCLING_ALIGNMENT_RIGHT_THRESHOLD = 1.5` pt
  - `DOCLING_ALIGNMENT_VERTICAL_THRESHOLD = 1.0` pt
  - `IOU_THRESHOLD = 0.10` for hidden-text overlap detection

## Current Implementation Status (2025-11-05)

### ✅ Completed
- ✅ Cell/phrase consolidation logic (2025-11-05)
- ✅ Baseline detection and identification (2025-11-05)
- ✅ Baseline rendering in Document Viewer (2025-11-05)

### 🚧 In Progress
- Future: Implement anomaly detection based on baseline deviations
- Future: Define thresholds for baseline-based alignment checks

### 📊 Test Results (2025-11-05)

Tested with payload: `20251105_134512_87026407408 – Mr I Fakude - False pay slip_Docling_raw_payload.json`

**Cell Consolidation:**
- ✅ Successfully consolidates spatially and textually contained cells
- ✅ Test showed 2 cells removed from synthetic data (HOURS and UNITS contained in HOURS /UNITS)
- ✅ Correctly preserves separate cells at different locations

**Baseline Detection:**
- ✅ Detected 11 alignment baselines on page 0:
  - 3 LEFT baselines at page-left (44.0pt, 44.7pt, 45.0pt)
  - 1 RIGHT baseline at page-left (78.7pt)
  - 2 RIGHT baselines at mid-page (186.0pt, 186.7pt)
  - 4 LEFT baselines at mid-page (197.3pt, 342.7pt, 374.0pt, 406.0pt)
  - 1 RIGHT baseline at mid-page (444.0pt)

**Visualization:**
- ✅ Baselines rendered as vertical lines in Document Viewer
- ✅ Color-coded: cyan/blue for left-aligned, orange/yellow for right-aligned
- ✅ Labels show orientation, x-coordinate, and item count

### 📦 Archived
- Previous label→value same-row alignment checks (archived until baseline detection is complete)
- Previous vertical block alignment checks (archived until baseline detection is complete)

## New Alignment Baseline Approach

### 1. Cell/Phrase Consolidation
Before any alignment analysis, we consolidate cells to use parent containers instead of contained words:
- If a "word" cell is spatially contained within a "phrase" or larger "cell", we use the parent container
- Example: "HOURS" (x0=114.67, x1=140.0) is contained in "HOURS /UNITS" (x0=324.67, x1=378.0)
- Consolidation logic checks bbox containment and proximity in the JSON structure
- This reduces false positives from analyzing word fragments instead of complete text units

### 2. Baseline Detection
Before performing alignment checks, we identify alignment baselines across the document:

**Baseline Characteristics:**
1. **Orientation:** Is the baseline left-aligned or right-aligned?
2. **Value:** The x-coordinate of the baseline (in points)
3. **Page Position:** Is it page-left, page-right, or mid-page alignment?
4. **Proximity Block Y-values:** Y-coordinates of text blocks that align to this baseline

**Detection Algorithm:**
1. Cluster all left edges (x0) and right edges (x1) at 0.1pt precision
2. Identify dominant clusters (≥3 items with similar x-coordinates)
3. For each cluster:
   - Calculate mean x-coordinate → baseline value
   - Determine if left/right aligned based on edge type
   - Classify page position:
     - Page-left: x < page_width * 0.25
     - Page-right: x > page_width * 0.75
     - Mid-page: otherwise
   - Record y-coordinates of all aligned blocks
4. Filter out structural elements (table headers, section headers)

### 3. Baseline Visualization
Baselines are rendered in the Document Viewer:
- **Left-aligned baselines:** Vertical lines in cyan/blue
- **Right-aligned baselines:** Vertical lines in orange/yellow
- **Page position indicators:** Different line styles (solid for page-edges, dashed for mid-page)
- **Baseline labels:** Show x-coordinate value and number of aligned items
- Baselines overlay the document at their exact x-coordinates for visual verification

## Label → Value Alignment (Same Row) [ARCHIVED]
`detect_docling_alignment_anomalies` first checks explicit label/value pairs such as *Account number*, *Account holder*, and *Product name*:

1. Normalise every cell into `{text, bbox}` with axis-aligned coordinates.
2. For each target label, locate the value to its right whose baseline matches within ±0.6× the label height (minimum ±1.2 pt).
3. Measure three signals:
   - **Spacing anomaly:** gap between the label’s right edge and the value’s left edge.
   - **Right alignment shift:** deviation of the value’s right edge from the dominant right edge across the group (bucketed at 0.1 pt).
   - **Baseline mismatch:** vertical distance between label and value baselines.
4. Docling metadata (table/section headers) is matched via IoU so those regions are excluded from horizontal checks.
5. Threshold breaches trigger “Spacing anomaly”, “Right alignment shift”, or “Baseline mismatch” reasons.

## Vertical Blocks Below a Label (Multi-line Values)
For labels ending with `:`, we also analyse stacked values (addresses, etc.):

1. Gather text beneath the label (≥ 0.5 pt, within 200 pt) that overlaps the label’s column.
2. Filter out section headers, table headers, and blocks separated by > 45 pt.
3. Bucket left edges at 0.1 pt to find the dominant column (must sit within 40 pt of the label). We simultaneously bucket right edges and capture the label’s own right edge.
4. A line is considered aligned if **either** its left edge is within `DOCLING_ALIGNMENT_LEFT_THRESHOLD` of the dominant column **or** its right edge is within `DOCLING_ALIGNMENT_RIGHT_THRESHOLD` of either the dominant right edge or the label’s right edge.
5. Failure on both sides yields a magenta “Left alignment deviation” or “Right alignment deviation” highlight (whichever side is closer to compliance). This catches forged lines like `1246` and `ZA` while leaving right-aligned totals such as `R20,525.32` untouched.

## Rendering Priority
- Hidden-text overlays draw first as red (visible layer) plus blue (hidden layer). Alignment overlays skip any region that overlaps hidden-text bboxes, ensuring forged underlays remain highlighted in blue.
- Docling tables are outlined in light green to show table grids.
- `build_annotation_details` merges every reason so the “Highlighted Items on Page …” list explains each coloured region.

## Extending / Tuning
- Adjust `target_labels` inside `detect_docling_alignment_anomalies` to monitor new fields.
- Tune the thresholds above to relax or tighten sensitivity.
- Structural metadata matching automatically filters out table/section headers, reducing false positives in column headers and section titles.
