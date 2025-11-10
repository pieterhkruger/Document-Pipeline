# Text Block & Alignment Anomaly Detection Requirements

**Document Type:** Requirements & Use Cases
**Created:** 2025-11-08
**Status:** In Progress
**Purpose:** Comprehensive documentation of text block detection and alignment anomaly detection requirements based on real-world forgery detection scenarios

---

## Table of Contents

1. [Overview](#overview)
2. [Use Cases](#use-cases)
   - [UC-001: Right-Aligned Column Deviation in Bank Statement](#uc-001-right-aligned-column-deviation-in-bank-statement)
   - [UC-002: Redundant Vertical Baselines in Left-Aligned Column](#uc-002-redundant-vertical-baselines-in-left-aligned-column)
   - [UC-003: Table-Based Block Splitting for Mixed Alignment Columns](#uc-003-table-based-block-splitting-for-mixed-alignment-columns)
   - [UC-004: Header Misalignment Detection (Advanced Case)](#uc-004-header-misalignment-detection-advanced-case)
   - [UC-005: Block Refinement and Conflict Resolution](#uc-005-block-refinement-and-conflict-resolution)
   - [UC-006: Table-Informed Block Creation](#uc-006-table-informed-block-creation)
3. [Positive Test Cases (Correct Behavior)](#positive-test-cases-correct-behavior)
   - [PTC-001: Correct Table Structure Detection](#ptc-001-correct-table-structure-detection)
   - [PTC-002: Correct Earnings Section Blocking](#ptc-002-correct-earnings-section-blocking)
   - [PTC-003: Correct Numerical Value Blocking](#ptc-003-correct-numerical-value-blocking)
   - [PTC-004: Correct Transaction Table Row Separation](#ptc-004-correct-transaction-table-row-separation)
   - [PTC-005: Correct Income/Transaction Item Separation](#ptc-005-correct-incometransaction-item-separation)
   - [PTC-006: Problematic Multi-Line Address Fragmentation](#ptc-006-problematic-multi-line-address-fragmentation)
   - [PTC-007: Correct Multi-Column Table Structure](#ptc-007-correct-multi-column-table-structure)
   - [PTC-008: Correct Contact Information Blocking](#ptc-008-correct-contact-information-blocking)
   - [PTC-009: Correct Earnings/Deductions Table Blocking](#ptc-009-correct-earningsdeductions-table-blocking)
4. [Cross-Cutting Requirements](#cross-cutting-requirements)
5. [Implementation Priorities](#implementation-priorities)

---

## Overview

This document captures real-world use cases and requirements for text block detection and alignment anomaly detection in the context of document forgery detection. Each use case is based on actual processed documents and describes:

- **Current Behavior:** What the system does now
- **Problem:** What's wrong or missing
- **Desired Behavior:** What should happen
- **Implementation Requirements:** Technical changes needed
- **Detection Logic:** Algorithms or heuristics required

These requirements inform the development of:
- Text block clustering algorithms (iterative, two-stage, etc.)
- Baseline detection and refinement logic
- Alignment anomaly detection systems
- Block-baseline mutual refinement

---

## Use Cases

### UC-001: Right-Aligned Column Deviation in Bank Statement

**Document:** `False FNB Statements - Mrs. Winnie Sokhela.pdf`
**Category:** Alignment Anomaly Detection
**Priority:** HIGH
**Status:** Not Implemented

#### Problem Description

The document contains a clear columnar structure with a right-aligned "Amount" column containing numerical values. The text block detection currently snaps to the rightmost character ("Cr" in "19,631.85 Cr") instead of respecting the dominant vertical baseline for right-aligned numbers.

**Visual Evidence:** See attachment showing the Amount column with values:
```
100.00
60.15
400.00
105.74
30.00
122.00
650.00
90.10
100.00
50.00
100.00
32.60
30.00
12.00
12.00
100.05
100.18
118.73
100.00
28.12
99.65
19,631.85 Cr    ← ANOMALY: Deviates from right alignment
8,100.00
2,680.00
394.12
100.00
```

#### Current Behavior

1. **Block Snapping Issue:**
   - Text block snaps to the "Cr" of "19,631.85 Cr" (rightmost character)
   - Ignores the vertical baseline that correctly identifies right alignment of numbers
   - Result: Block bbox extends further right than the actual column

2. **Missing Alignment Type:**
   - Blocks don't have a "most likely alignment type" attribute
   - Cannot distinguish between left-aligned, right-aligned, center, or mixed blocks
   - Prevents alignment-based anomaly detection

3. **No Column Consistency Detection:**
   - System doesn't detect that this is a column with consistent right-aligned numbers
   - Doesn't identify "19,631.85 Cr" as deviating from the column pattern

4. **No Content Pattern Analysis:**
   - Doesn't recognize that ALL other cells are pure numbers
   - Doesn't flag "Cr" suffix as anomalous in numeric column
   - Missing: Content type consistency detection (all numbers vs. number+text)

#### Desired Behavior

1. **Baseline-Aware Block Snapping:**
   - Block should snap to the dominant vertical baseline for right-aligned content
   - Even if "Cr" extends beyond baseline, block edge should align to baseline
   - Baseline correctly identifies x=XXX as the right edge for this column

2. **Alignment Type Attribution:**
   - Each block should have `alignment_type` property: "left" | "right" | "center" | "mixed"
   - Right-aligned blocks identified by majority of cells aligning to right baseline
   - Confidence score for alignment type (e.g., 95% right-aligned)

3. **Column Deviation Detection:**
   - Detect strong vertical baseline with multiple supporting blocks
   - Identify when a block within that column deviates from alignment
   - Flag: "Block at y=XXX deviates 12.3pt from dominant right baseline"
   - Severity: HIGH (in financial document context)

4. **Content Pattern Anomaly Detection:**
   - Analyze cell content types within a column/block
   - Detect: 25/26 cells are pure numbers, 1/26 has "number + text"
   - Flag: "Cell '19,631.85 Cr' contains text suffix in otherwise numeric column"
   - Rationale: In banking context, "Cr" suffix is suspicious (likely manual edit)

#### Implementation Requirements

##### 1. Baseline-Aware Block Finalization

**Function:** `identify_text_blocks_iterative()` (and two-stage variant)
**Location:** `text_block_detection.py`

**Changes:**
```python
# Current: Snap to actual bbox edges
x0 = min(c["bbox"]["x0"] for c in cluster_cells)
x1 = max(c["bbox"]["x1"] for c in cluster_cells)

# Proposed: Snap to baseline if strong alignment detected
if dominant_alignment == "right" and alignment_baseline:
    # Check if majority of cells align to baseline (within tolerance)
    aligned_count = sum(1 for c in cluster_cells
                       if abs(c["bbox"]["x1"] - alignment_baseline) <= BASELINE_SNAP_TOLERANCE)

    if aligned_count / len(cluster_cells) >= 0.7:  # 70% threshold
        # Snap block right edge to baseline instead of max x1
        x1 = alignment_baseline
        block["snapped_to_baseline"] = True
        block["alignment_confidence"] = aligned_count / len(cluster_cells)
```

##### 2. Alignment Type Detection

**Add to block metadata:**
```python
block = {
    "block_id": int,
    "alignment_type": "left" | "right" | "center" | "mixed",
    "alignment_confidence": float,  # 0.0 to 1.0
    "alignment_baseline": float | None,
    "alignment_deviation_cells": [
        {
            "cell_text": str,
            "expected_edge": float,
            "actual_edge": float,
            "deviation": float
        }
    ]
}
```

**Algorithm:**
```python
def determine_alignment_type(cluster_cells, left_baselines, right_baselines):
    """
    Determine dominant alignment type for a block.

    Returns:
        alignment_type: str
        confidence: float
        baseline: float | None
        deviations: list
    """
    # Count cells aligned to each type
    left_aligned = []
    right_aligned = []
    center_aligned = []

    for cell in cluster_cells:
        x0, x1 = cell["bbox"]["x0"], cell["bbox"]["x1"]

        # Check left alignment
        for baseline in left_baselines:
            if abs(x0 - baseline) <= BASELINE_SNAP_TOLERANCE:
                left_aligned.append(cell)
                break

        # Check right alignment
        for baseline in right_baselines:
            if abs(x1 - baseline) <= BASELINE_SNAP_TOLERANCE:
                right_aligned.append(cell)
                break

        # Check center (neither left nor right)
        # ... implementation

    # Determine dominant type
    total = len(cluster_cells)
    left_pct = len(left_aligned) / total
    right_pct = len(right_aligned) / total

    if right_pct >= 0.7:
        return "right", right_pct, find_dominant_baseline(right_aligned, right_baselines)
    elif left_pct >= 0.7:
        return "left", left_pct, find_dominant_baseline(left_aligned, left_baselines)
    elif left_pct + right_pct < 0.3:
        return "center", 1.0 - (left_pct + right_pct), None
    else:
        return "mixed", 1.0, None
```

##### 3. Column Alignment Deviation Detection

**New Function:** `detect_column_alignment_anomalies()`
**Location:** `pdf_ocr_detector.py` (new anomaly detector)

**Algorithm:**
```python
def detect_column_alignment_anomalies(blocks, baselines):
    """
    Detect blocks that deviate from strong vertical baseline columns.

    For each strong right baseline:
        1. Find all blocks that align to it (>= 70% cells)
        2. Check vertical coverage (should span significant page height)
        3. For each block:
            - Check if any cells deviate from baseline
            - Calculate deviation magnitude
            - Flag high-confidence right-aligned blocks with deviations
    """
    anomalies = []

    # Group blocks by alignment baseline
    for baseline in right_baselines:
        aligned_blocks = [
            b for b in blocks
            if b["alignment_type"] == "right"
            and b["alignment_baseline"] == baseline
            and b["alignment_confidence"] >= 0.7
        ]

        if len(aligned_blocks) < 3:  # Need at least 3 blocks for pattern
            continue

        # Check vertical coverage
        coverage = calculate_vertical_coverage(aligned_blocks)
        if coverage < 0.3:  # Must cover >= 30% of page height
            continue

        # This is a strong column - check for deviations
        for block in aligned_blocks:
            if block.get("alignment_deviation_cells"):
                for dev_cell in block["alignment_deviation_cells"]:
                    if abs(dev_cell["deviation"]) > COLUMN_DEVIATION_THRESHOLD:
                        anomalies.append({
                            "type": "column_alignment_deviation",
                            "block_id": block["block_id"],
                            "baseline": baseline,
                            "cell_text": dev_cell["cell_text"],
                            "deviation": dev_cell["deviation"],
                            "severity": "HIGH",
                            "reason": f"Cell deviates {dev_cell['deviation']:.1f}pt from dominant right baseline in strong column"
                        })

    return anomalies
```

##### 4. Content Pattern Consistency Detection

**New Function:** `detect_content_pattern_anomalies()`
**Location:** `pdf_ocr_detector.py`

**Algorithm:**
```python
def detect_content_pattern_anomalies(blocks):
    """
    Detect cells with content patterns that deviate from column consistency.

    For blocks with strong alignment (right-aligned financial columns):
        1. Analyze content type of each cell (number, number+text, text, etc.)
        2. Identify dominant pattern (e.g., 90% pure numbers)
        3. Flag cells that deviate from pattern
        4. Higher severity for numeric+text in otherwise pure numeric column
    """
    anomalies = []

    for block in blocks:
        if block["alignment_type"] != "right":
            continue

        if block["alignment_confidence"] < 0.7:
            continue

        # Analyze content types
        content_types = []
        for cell in block["cells"]:
            text = cell.get("text", "").strip()
            ctype = classify_content_type(text)
            content_types.append({
                "text": text,
                "type": ctype,
                "bbox": cell["bbox"]
            })

        # Find dominant pattern
        type_counts = Counter(c["type"] for c in content_types)
        dominant_type, dominant_count = type_counts.most_common(1)[0]
        consistency = dominant_count / len(content_types)

        if consistency >= 0.8:  # 80% same type
            # Flag deviations
            for ct in content_types:
                if ct["type"] != dominant_type:
                    anomalies.append({
                        "type": "content_pattern_deviation",
                        "block_id": block["block_id"],
                        "cell_text": ct["text"],
                        "expected_type": dominant_type,
                        "actual_type": ct["type"],
                        "severity": "HIGH" if dominant_type == "number" and "text" in ct["type"] else "MEDIUM",
                        "reason": f"Cell contains {ct['type']} in column with {consistency:.0%} {dominant_type} cells",
                        "suspicion_note": "Text suffix in numeric column (e.g., 'Cr', 'Dr') often indicates manual editing"
                    })

    return anomalies

def classify_content_type(text):
    """Classify cell content type."""
    text = text.strip()

    # Remove common number formatting
    clean = text.replace(",", "").replace(" ", "")

    # Pure number (including decimals, negatives)
    if re.match(r'^-?\d+\.?\d*$', clean):
        return "number"

    # Number with currency symbol
    if re.match(r'^[$€£R]?\s*-?\d+\.?\d*$', clean):
        return "number_with_currency"

    # Number with text suffix (e.g., "19,631.85 Cr")
    if re.match(r'^-?\d+\.?\d*\s+[A-Za-z]+$', text):
        return "number_with_text_suffix"

    # Number with text prefix
    if re.match(r'^[A-Za-z]+\s+-?\d+\.?\d*$', text):
        return "number_with_text_prefix"

    # Pure text
    return "text"
```

#### Integration with Docling Table Information

**Requirement:** Use Docling's table detection to support column-based analysis

**Changes:**
```python
# In pdf_ocr_detector.py

def detect_docling_alignment_anomalies(docling_payload, pdf_path):
    # ... existing code ...

    # NEW: Extract table information from Docling
    tables = extract_docling_tables(docling_payload, page_no)

    # Use table structure to:
    # 1. Identify columns more reliably
    # 2. Get column headers (e.g., "Amount", "Date", etc.)
    # 3. Associate text blocks with table columns
    # 4. Apply column-specific content validation

    for table in tables:
        for col_idx, column in enumerate(table["columns"]):
            header = column.get("header", "")
            cells = column.get("cells", [])

            # For numeric columns (e.g., "Amount", "Balance")
            if is_numeric_column(header):
                # All cells should be numbers or number+currency
                # Flag any with unexpected text suffixes
                anomalies.extend(
                    detect_numeric_column_anomalies(cells, header)
                )
```

#### Expected Output

**Anomaly Report:**
```json
{
  "anomalies": [
    {
      "type": "column_alignment_deviation",
      "block_id": 7,
      "page": 0,
      "baseline": 450.2,
      "cell_text": "19,631.85 Cr",
      "bbox": {"x0": 420.5, "y0": 245.3, "x1": 465.8, "y1": 255.1},
      "deviation": 15.6,
      "severity": "HIGH",
      "reason": "Cell right edge at 465.8pt deviates 15.6pt from dominant right baseline at 450.2pt in strong column",
      "visualization_color": "red"
    },
    {
      "type": "content_pattern_deviation",
      "block_id": 7,
      "page": 0,
      "cell_text": "19,631.85 Cr",
      "expected_type": "number",
      "actual_type": "number_with_text_suffix",
      "pattern_consistency": 0.96,
      "severity": "HIGH",
      "reason": "Cell contains number_with_text_suffix in column with 96% number cells",
      "suspicion_note": "Text suffix 'Cr' in numeric column often indicates manual editing or forgery",
      "visualization_color": "red"
    }
  ]
}
```

#### Testing Validation

**Test Document:** `False FNB Statements - Mrs. Winnie Sokhela.pdf`

**Expected Results:**
1. ✅ Block for Amount column has `alignment_type: "right"`, `alignment_confidence: 0.96`
2. ✅ Block snaps to x1=450.2 (baseline) instead of x1=465.8 (Cr edge)
3. ✅ "19,631.85 Cr" flagged as column alignment deviation (15.6pt)
4. ✅ "19,631.85 Cr" flagged as content pattern deviation (text suffix in numeric column)
5. ✅ Both anomalies highlighted in RED in document viewer
6. ✅ Anomaly details shown in "Highlighted Items" section with severity and reasoning

---

## Positive Test Cases (Correct Behavior)

This section documents examples of **correct** text block detection and baseline identification. These cases must continue to work correctly after implementing the use cases above. Any new logic should preserve this correct behavior to avoid regressions.

### PTC-001: Correct Table Structure Detection

**Document:** `85385859429 – Mr MP Madiya – False pay slip.pdf`
**Category:** Text Block Detection
**Status:** ✅ Working Correctly

#### Visual Evidence

![Employee Information Table](Screenshots show purple blocks correctly identifying table structure)

**Screenshot 1:** Employee information table with multiple rows and columns

#### Current Behavior (Correct)

The system correctly:

1. **Separates table rows into individual blocks**
   - Each row of employee data forms a distinct block
   - Example rows:
     - "12894 MAMPHO STREET" / "Dt Engaged" / "2023/06/05"
     - "MIDRAND" / "Occup" / "SUPERVISOR"
     - "JOHANNESBURG 1685" / "ID Number" / "8605135836086"
     - "MR MATOME P MADIYA" / "Bank" / "ABSA BANK"
     - "PETER" / "Account No" / "4106364207"
     - "MPC4UG" / "Branch Code" / "632005"

2. **Maintains column structure**
   - Left column (addresses, names): Left-aligned blocks
   - Middle column (labels): Properly aligned blocks
   - Right column (values): Correctly identified blocks

3. **Detects vertical baselines accurately**
   - Cyan baselines: Left alignment edges
   - Orange baselines: Right alignment edges
   - Each column has its own baseline(s)

4. **Preserves multi-line cell content**
   - "MASTER PAVING CONSTRUCTION" remains in one block
   - "12894 MAMPHO STREET" grouped with address components

#### Why This Is Correct

- Each row represents a logical unit (one piece of information)
- Column boundaries are respected (no cross-column blocks)
- Alignment is consistent within each column
- Baselines accurately represent the dominant alignments

#### What Must Be Preserved

✅ **Row-based blocking:** Don't merge rows vertically unless they're part of the same logical block
✅ **Column separation:** Don't merge cells from different columns horizontally
✅ **Baseline accuracy:** Don't add redundant baselines within consistent columns
✅ **Multi-line handling:** Keep multi-line cells together in one block

---

### PTC-002: Correct Earnings Section Blocking

**Document:** `85385859429 – Mr MP Madiya – False pay slip.pdf`
**Category:** Text Block Detection
**Status:** ✅ Working Correctly

#### Visual Evidence

**Screenshot 2:** Earnings section with numerical values

#### Current Behavior (Correct)

The system correctly:

1. **Separates label blocks from value blocks**
   - "Taxable Earnings" in one block
   - "R 360,000.00" in another block
   - "Co. contributions" separate
   - "R 177.12" separate

2. **Maintains alignment consistency**
   - Label column: Left-aligned
   - Value column: Right-aligned (consistent with "R" prefix)

3. **Detects column boundaries**
   - Clear separation between labels and amounts
   - No cross-column merging

#### Why This Is Correct

- Labels and values serve different purposes (should be in different blocks)
- Right-alignment of amounts is preserved
- Column structure matches document layout

#### What Must Be Preserved

✅ **Label/Value separation:** Keep labels and amounts in separate blocks
✅ **Right-alignment detection:** Continue detecting right-aligned numeric columns
✅ **Column boundaries:** Don't merge across the label/value boundary

---

### PTC-003: Correct Numerical Value Blocking

**Document:** `85385859429 – Mr MP Madiya – False pay slip.pdf`
**Category:** Text Block Detection
**Status:** ✅ Working Correctly

#### Visual Evidence

**Screenshot 3:** Numerical values section

#### Current Behavior (Correct)

The system correctly:

1. **Creates individual blocks for each value**
   - "R 4,783.08"
   - "R 177.12"
   - "R 787.10"
   - "R 627.20"

2. **Maintains vertical spacing**
   - Each value is separated despite being in the same column
   - Spacing indicates these are separate line items

3. **Preserves right-alignment**
   - All values align to the same right edge
   - Currency symbol and amount together

#### Why This Is Correct

- Each value represents a distinct line item
- Vertical separation indicates different categories/purposes
- Right-alignment is semantically meaningful (numeric data)

#### What Must Be Preserved

✅ **Individual value blocks:** Don't merge all right-aligned values into one large block
✅ **Spacing-based separation:** Respect vertical gaps as block boundaries
✅ **Alignment consistency:** Maintain right-alignment for numeric columns

---

### PTC-Summary: Key Correct Behaviors

The following behaviors are currently working correctly and **must not regress**:

| Behavior | Description | Example |
|----------|-------------|---------|
| **Table row blocking** | Each table row forms a distinct block | Employee info rows |
| **Column separation** | Cells in different columns stay in different blocks | Address column vs. Label column |
| **Multi-line cells** | Multi-line content within one cell stays together | "MASTER PAVING CONSTRUCTION" |
| **Baseline accuracy** | One baseline per consistent alignment within a column | Left baseline for address column |
| **Label/Value split** | Labels and their values in separate blocks | "Taxable Earnings" vs. "R 360,000.00" |
| **Vertical spacing** | Gaps between blocks respected | Individual earning amounts |
| **Alignment preservation** | Left/right/center alignment detected correctly | Right-aligned amounts |

**Testing Strategy:** When implementing UC-002 through UC-006, run tests on this document to ensure:
1. Same number of final blocks (or justify changes)
2. Same block boundaries (within 2pt tolerance)
3. Same alignment classifications (left/right/center)
4. Same number of baselines for consistently-aligned columns

---

### PTC-004: Correct Transaction Table Row Separation

**Document:** `85393423298 – Mr G Motau - legitimate Capitec Bank statements.pdf`
**Category:** Text Block Detection
**Status:** ✅ Working Correctly

#### Visual Evidence

**Screenshot 3:** Transaction summary table and fee breakdown

#### Current Behavior (Correct)

The system correctly:

1. **Separates each transaction row into individual label/value blocks**
   - "Card Payments" | "-R7,551.40" (separate blocks)
   - "Cash Withdrawals" | "-R3,920.00" (separate blocks)
   - "Digital Payments" | "-R1,360.00" (separate blocks)
   - "Send Cash" | "-R750.00" (separate blocks)
   - "Transfer" | "-R219.66" (separate blocks)
   - "Fees" | "-R216.08" (separate blocks)

2. **Maintains table structure in fee breakdown**
   - "Fee Summary" header block: "-R216.08"
   - Individual fee items separated:
     - "Cash Withdrawal Fee" | "R110.00"
     - "SMS Notification Fee" | "-R36.40"
     - "Cash Sent Fee" | "-R30.00"
     - "Monthly Account Admin Fee" | "-R22.50"
     - "Cash Deposit Fee (Notes)" | "-R5.18"
     - "Immediate Payment Fee" | "-R5.00"
     - "Other Fees" | "-R7.00"

3. **Respects vertical spacing between rows**
   - Each transaction type is a distinct line item
   - Gaps between rows indicate separate entries
   - No unwanted merging of vertically stacked items

#### Why This Is Correct

- Each transaction represents a separate category (not a multi-line description)
- Label-value separation allows for proper column alignment
- Vertical spacing is semantically meaningful (different transaction types)
- Table structure requires row-by-row blocking

#### What Must Be Preserved

✅ **Row-level separation:** Don't merge vertically adjacent table rows into one block
✅ **Label-value independence:** Keep labels and values in separate blocks
✅ **Spacing semantics:** Respect vertical gaps as row boundaries
✅ **Table structure:** Maintain individual row blocks for tabular data

---

### PTC-005: Correct Income/Transaction Item Separation

**Document:** `85393423298 – Mr G Motau - legitimate Capitec Bank statements.pdf`
**Category:** Text Block Detection
**Status:** ✅ Working Correctly

#### Visual Evidence

**Screenshot 4:** Income and deposit items

#### Current Behavior (Correct)

The system correctly:

1. **Creates individual blocks for each income/transaction item**
   - "Salary" | "R36,314.78" (separate blocks)
   - "Other Income" | "R12,435.00" (separate blocks)
   - "Cash Deposit" | "R370.00" (separate blocks)
   - "Transfer" | "R120.00" (separate blocks)
   - "Interest" | "R1.64" (separate blocks)

2. **Maintains alignment consistency**
   - Labels: Left-aligned column
   - Amounts: Right-aligned column (with "R" prefix)

3. **Preserves row structure**
   - Each income source is a distinct block
   - No merging of vertically stacked income types

#### Why This Is Correct

- Each income source represents a separate financial category
- Amounts are right-aligned (semantically meaningful for numeric data)
- Row-by-row structure matches tabular layout
- Vertical separation indicates different income streams

#### What Must Be Preserved

✅ **Category separation:** Each income/transaction type in its own row block
✅ **Alignment detection:** Right-aligned amounts preserved
✅ **No vertical merging:** Income items stay separate despite vertical proximity
✅ **Row independence:** Each row represents distinct financial data

---

### PTC-006: Problematic Multi-Line Address Fragmentation

**Document:** `85393423298 – Mr G Motau - legitimate Capitec Bank statements.pdf`
**Category:** Text Block Detection - **PROBLEM CASE**
**Status:** ⚠️ **Needs Improvement** (Related to UC-005)

#### Visual Evidence

**Screenshot 1:** Capitec Bank address
**Screenshot 2:** Customer address

#### Current Behavior (Incorrect)

The system currently creates **separate blocks for each line** of multi-line addresses:

**Example 1 - Bank Address:**
- "Capitec Bank Limited" (Block 1)
- "5 Neutron Road" (Block 2)
- "Techno Park" (Block 3)
- "Stellenbosch" (Block 4)
- "7600" (Block 5)

**Example 2 - Customer Address:**
- "MR GONTSE MOTAU" (Block 1)
- "96 LEBANON" (Block 2)
- "WARD 12" (Block 3)
- "GABPANEI" (Block 4)
- "RETORIA" (Block 5)
- "0190" (Block 6)

#### Desired Behavior

**Each complete address should be ONE block:**
- Bank address: All 5 lines in ONE block
- Customer address: All 6 lines in ONE block

#### Why Merging Is Needed

1. **Semantic coherence:** All lines represent a single logical entity (one address)
2. **Consistent alignment:** All lines are left-aligned to the same baseline
3. **Tight vertical spacing:** Line spacing is consistent (~12-14pt), indicating continuity
4. **No intervening content:** No gaps or other content between address lines
5. **Typographical grouping:** Font, size, and style are consistent

#### Detection Criteria for Address Block Merging

**Should merge vertically adjacent blocks when:**
1. Same alignment (all left-aligned to same baseline within 2pt)
2. Vertical gap < 2.0 × typical line spacing
3. Horizontal alignment consistent (left edges within 2pt)
4. No competing content between blocks
5. Similar font properties (size, family within tolerance)
6. Combined block has reasonable line count (2-10 lines)

#### Relationship to Use Cases

This problem is addressed by:
- **UC-005: Block Refinement and Conflict Resolution** - Adjacent block merging logic
  - Criterion: "Share the same dominant baseline"
  - Criterion: "Vertical gap < median_spacing × 2.0"
  - Criterion: "Combined block meets minimum line count"

#### Impact on Testing

⚠️ **This is a KNOWN ISSUE** - not correct behavior to preserve
- When UC-005 is implemented, these addresses SHOULD merge into single blocks
- Current behavior (5-6 separate blocks) is INCORRECT
- Target behavior: 1 block per address

### PTC-007: Correct Multi-Column Table Structure

**Document:** `87026407408 – Mr I Fakude - False pay slip.pdf`
**Category:** Text Block Detection
**Status:** ✅ Working Correctly

#### Visual Evidence

**Screenshot 1:** Employee information table (top section)
**Screenshot 3:** Employee detail close-up

#### What the System Does Correctly

The system correctly handles a **complex 3-column table layout** with employee information:

1. **Row-based blocking**
   - Row 1: "HENCO MINERALS (PTY) LTD" | "SALARY SLIP"
   - Row 2: "EMP.CODE: HM20927" | "DEPARTMENT: MINING DIVISION" | "PERIOD END DATE:2024/11/27"
   - Row 3: "EMP.NAME: MR INNOCENT FAKUDE" | "JOB TITLE: FITTER" | "DATE ENGAGED:2024/11/01"
   - Row 4: "KNOWN AS: INNOCENT" | "PAYMENT METHOD: AGE" | "ID NUMBER: 1611763731088"
   - Row 5: "MPRESSELTON" | "EMPL. TYPE: PERMANENT" | "ACCOUNT TYPE: SAVINGS"
   - Row 6: "ERMELO 2350" | (blank) | "BANK: CAPITEC BANK"

2. **Column separation**
   - Left column: Names, addresses, locations
   - Middle column: Labels and field names
   - Right column: Values and data

3. **Label-value pairs within cells**
   - "EMP.CODE: HM20927" - kept together in one block
   - "DEPARTMENT: MINING DIVISION" - kept together in one block
   - "PERIOD END DATE:2024/11/27" - kept together in one block

#### What Must Be Preserved

✅ **Multi-column table handling:** Properly separate left/middle/right columns
✅ **Row independence:** Each table row forms distinct blocks
✅ **Compound labels:** Label:Value pairs within same cell stay together
✅ **Horizontal separation:** Don't merge across column boundaries
✅ **Vertical separation:** Don't merge rows vertically unless they're part of same logical block
✅ **Mixed content types:** Handle text, dates, numbers, codes consistently

#### Why This Matters

This layout is **common in payslips and forms**:
- Left column: Personal information (names, addresses)
- Middle column: Field labels (semi-structured)
- Right column: Field values (structured data)

Correct blocking enables:
- Extracting field-value pairs
- Detecting tampering in specific cells
- Maintaining table structure for analysis

---

### PTC-008: Correct Contact Information Blocking

**Document:** `87026407408 – Mr I Fakude - False pay slip.pdf`
**Category:** Text Block Detection
**Status:** ✅ Working Correctly

#### Visual Evidence

**Screenshot 2:** Address and contact section

#### What the System Does Correctly

The system correctly separates **contact information elements** into individual blocks:

1. **Individual phone number blocks**
   - "SWITCH BOARD: 0871471135" (separate block)
   - "HR DIRECT TEL: 0101574929" (separate block)

2. **Individual email block**
   - "EMAIL: hr@henccminerals.co.za" (separate block)

3. **Address line separation**
   - "2191" (separate block - postal code)

4. **Header separation**
   - "SALARY SLIP" (separate block)
   - "PERIOD END DATE:2024/11/27" (separate block)

#### What Must Be Preserved

✅ **Contact item independence:** Each phone/email/address element in separate block
✅ **Label-value coupling:** "SWITCH BOARD: 0871471135" stays together
✅ **No vertical merging:** Phone numbers don't merge despite close vertical proximity
✅ **Consistent alignment:** All items left-aligned but remain separate blocks

#### Why This Matters

Contact information needs to be **individually extractable**:
- Phone numbers can be validated separately
- Email addresses can be checked for domain consistency
- Tampering often affects individual contact fields
- Enables field-level anomaly detection

---

### PTC-009: Correct Earnings/Deductions Table Blocking

**Document:** `87026407408 – Mr I Fakude - False pay slip.pdf`
**Category:** Text Block Detection
**Status:** ✅ Working Correctly

#### Visual Evidence

**Screenshot 4:** Earnings and tax section

#### What the System Does Correctly

The system correctly handles **tabular financial data** with label-value separation:

1. **Individual earning/deduction lines as separate label-value pairs**
   - "TAXABLE EARNING" | "32 000.00" (separate blocks)
   - "PERKS" | "0.00" (separate blocks)
   - "TAX" | "5 359.33" (separate blocks)

2. **Section headers isolated**
   - "CURRENT PERIOD" (separate block)

3. **Summary items separated**
   - "TOTAL PERKS" | "0.00" (separate blocks)
   - "CO. CONTRIBUTION" | "0.00" (separate blocks)
   - "ANNUAL LEAVE DUE" | "1.25" (separate blocks)

#### What Must Be Preserved

✅ **Label-value separation:** Financial labels and amounts stay in separate blocks
✅ **Row independence:** Each earning/deduction line is a separate row
✅ **Right-alignment detection:** Numeric values correctly identified as right-aligned
✅ **Section headers:** Headers like "CURRENT PERIOD" isolated from data rows
✅ **Decimal precision:** Values with decimals (0.00, 5 359.33, 1.25) handled correctly
✅ **No vertical merging:** Similar amounts don't merge into one block

#### Why This Matters

Financial tables are **critical for fraud detection**:
- Each line item must be independently verifiable
- Amounts must be extractable for calculation validation
- Right-alignment is a strong indicator of numeric columns
- Tampering often affects individual line items (amounts, labels)
- Enables line-item-level anomaly detection

**Example fraud scenarios detected:**
- Changed amount in one earning line
- Added/removed deduction line
- Misaligned amount (should be right-aligned, but isn't)

---

### PTC-Summary: Updated Key Correct Behaviors

| Behavior | Description | Example | PTCs |
|----------|-------------|---------|------|
| **Table row blocking** | Each table row forms a distinct block | Employee info rows | PTC-001, PTC-007 |
| **Column separation** | Cells in different columns stay in different blocks | Address column vs. Label column | PTC-001, PTC-007 |
| **Multi-line cells** | Multi-line content within one cell stays together | "MASTER PAVING CONSTRUCTION" | PTC-001 |
| **Baseline accuracy** | One baseline per consistent alignment within a column | Left baseline for address column | PTC-001 |
| **Label/Value split** | Labels and their values in separate blocks | "Taxable Earnings" vs. "R 360,000.00" | PTC-002, PTC-008, PTC-009 |
| **Vertical spacing** | Gaps between blocks respected | Individual earning amounts | PTC-003 |
| **Alignment preservation** | Left/right/center alignment detected correctly | Right-aligned amounts | PTC-003, PTC-009 |
| **Contact item independence** | Each phone/email/address element in separate block | Phone numbers, emails | PTC-008 |
| **Multi-column tables** | Complex 3+ column layouts properly separated | Employee information table | PTC-007 |
| **Financial data blocking** | Earnings/deductions in separate row blocks | Earning line items | PTC-009 |

**Total Positive Test Cases:** 9 (PTC-001 through PTC-009)
- ✅ **Working Correctly:** 8 cases (PTC-001 through PTC-005, PTC-007 through PTC-009)
- ⚠️ **Known Issues:** 1 case (PTC-006 - Multi-line address fragmentation)

**Testing Strategy:** When implementing UC-002 through UC-006, run tests on all 3 documents to ensure:
1. Same number of final blocks (or justify changes)
2. Same block boundaries (within 2pt tolerance)
3. Same alignment classifications (left/right/center)
4. Same number of baselines for consistently-aligned columns
5. No regressions on any of the 8 working test cases

---

## Cross-Cutting Requirements

*(To be populated as more use cases are added)*

---

## Implementation Priorities

### Phase 1: Foundation (High Priority)
1. **Baseline Confidence Scoring** (UC-002) - Add confidence metrics (support, coverage, consistency) to all baselines
2. **Baseline Filtering & Merging** (UC-002) - Filter weak baselines and merge redundant ones
3. **Alignment Type Detection** (UC-001) - Add alignment_type, confidence, baseline to blocks
4. **Baseline-Aware Snapping** (UC-001) - Snap blocks to baselines when strong alignment detected
5. **Content Type Classification** (UC-001) - Implement classify_content_type() function

### Phase 2: Mutual Refinement (High Priority)
6. **Baseline-to-Block Refinement** (UC-002) - Snap baselines to strong block edges
7. **Column Alignment Deviation** (UC-001) - Detect deviations from strong vertical baselines
8. **Content Pattern Consistency** (UC-001) - Detect content type deviations within columns

### Phase 3: Integration & Block Refinement (High Priority)
9. **Docling Table Extraction** (UC-003) - Extract table structure with column boundaries
10. **Alignment Conflict Detection** (UC-003) - Detect blocks with mixed left/right alignment
11. **Table-Based Block Splitting** (UC-003) - Split blocks that span multiple table columns
12. **Vertical Baseline Split Detection** (UC-003) - Use internal baselines as split triggers
13. **Block Confidence Scoring** (UC-005) - Calculate quality metrics for all blocks
14. **Nested Block Detection & Removal** (UC-005) - Detect and remove engulfed blocks using IoU
15. **Weak Evidence Block Removal** (UC-005) - Filter blocks with low cell density / vertical span
16. **Adjacent Block Merging** (UC-005) - Merge same-alignment adjacent blocks in tables
17. **Iterative Block-Baseline Refinement** (UC-001 + UC-002) - Circular refinement loop
18. **Multi-Page Consistency** - Track column patterns across pages

### Phase 4: Visualization & Advanced Detection (Lower Priority)
19. **Confidence-Based Visualization** (UC-002) - Render baselines with opacity based on confidence
20. **Anomaly Severity Scoring** (UC-001) - Add ML-based confidence for anomaly severity
21. **False Positive Reduction** - Tune thresholds based on document type
22. **Header-Data Alignment Analysis** (UC-004) - Advanced detection for header misalignment (radar only)

### Rationale

**Phase 1 starts with baselines (UC-002)** because:
- Baseline quality directly impacts block detection quality
- Filtering weak baselines reduces noise early in the pipeline
- Block detection relies on good baselines as input

**Phase 2 implements mutual refinement** because:
- Blocks and baselines inform each other
- Both UC-001 and UC-002 refinement logic needed for complete solution
- Anomaly detection requires high-quality blocks with alignment types

**Phase 3 integrates external data, implements block splitting and refinement (UC-003 + UC-005)** because:
- Docling tables provide authoritative structure for column boundaries
- Table-based splitting requires baseline confidence (Phase 1) and alignment detection (Phase 2)
- Alignment conflict detection identifies blocks needing to be split
- Block splitting improves granularity, but creates conflicts (nested, weak blocks)
- UC-005 refinement resolves these conflicts: merges adjacent blocks, removes nested/weak blocks
- Order matters: Split first (UC-003), then refine (UC-005)
- Iterative refinement (UC-001 + UC-002) depends on clean blocks from UC-005

**Phase 4 adds polish and advanced detection (UC-004)** because:
- Visualization improvements enhance user experience
- ML-based scoring (requires training data from earlier phases)
- Document-type-specific tuning reduces false positives
- Header-data alignment (UC-004) is complex with many edge cases - needs more examples before implementation

---

**End of Use Case UC-001**

---

### UC-002: Redundant Vertical Baselines in Left-Aligned Column

**Document:** `False FNB Statements - Mrs. Winnie Sokhela.pdf`
**Category:** Baseline Confidence & Redundancy Detection
**Priority:** HIGH
**Status:** Not Implemented

#### Problem Description

The document contains a "Description" column with left-aligned text elements. Multiple vertical baselines are detected running through the text block, but many of these baselines have no clear rationale when examining the page vertically. These redundant baselines create noise and prevent clean column boundary detection.

**Key Observations:**
- Description column is **LEFT aligned** (not right or center)
- Text block's right edge is already correct for left alignment
- Multiple vertical lines (baselines) run through the same text block
- Scrolling up/down reveals no consistent support for redundant baselines
- The vertical baseline should snap **TO** the text block (not vice versa)

#### Current Behavior

1. **Overdetection of Baselines:**
   - System detects many vertical baselines within a single text block
   - No distinction between strong baselines (high support) vs weak ones (coincidental)
   - All baselines treated equally regardless of evidence quality

2. **No Confidence Scoring:**
   - Baselines lack confidence metrics (support count, vertical coverage, consistency)
   - Cannot identify which baselines are reliable vs noise
   - No quantitative basis for filtering weak baselines

3. **No Parameter Tuning:**
   - Cannot adjust detection sensitivity based on baseline quality
   - Fixed thresholds apply to all baselines equally
   - No ability to say "only keep baselines with >80% confidence"

4. **Block-to-Baseline Snapping Only:**
   - Current UC-001 focuses on blocks snapping to baselines
   - Missing inverse: baselines should snap to strong block edges
   - For left-aligned blocks, left baseline should align to block's left edge

#### Desired Behavior

1. **Baseline Confidence Scoring:**
   - Each baseline has a confidence score (0.0 to 1.0)
   - Metrics: support count, vertical coverage, alignment consistency
   - Example: "Left baseline at x=120.5 has 0.92 confidence (23 cells, 65% coverage)"

2. **Confidence-Based Filtering:**
   - Configurable threshold: `MIN_BASELINE_CONFIDENCE = 0.7`
   - Eliminate baselines below threshold
   - Report in debug: "Filtered 8 weak baselines (confidence < 0.7)"

3. **Baseline-to-Block Refinement:**
   - Strong text blocks refine nearby baselines
   - Left-aligned block → snap left baseline to block's left edge
   - Right-aligned block → snap right baseline to block's right edge
   - Reduces redundant baselines caused by minor alignment variations

4. **Baseline Merging:**
   - Detect baselines within tolerance (e.g., 2pt apart)
   - Merge if they have similar support and coverage
   - Keep the stronger one (higher confidence) or merged average

5. **Visualization of Confidence:**
   - Baselines rendered with opacity based on confidence
   - High confidence (0.9+): Solid line
   - Medium confidence (0.7-0.9): Semi-transparent
   - Low confidence (<0.7): Dotted or hidden

#### Implementation Requirements

##### 1. Baseline Confidence Scoring

**Function:** `calculate_baseline_confidence()`
**Location:** `alignment_detection.py` (new function)

**Algorithm:**
```python
def calculate_baseline_confidence(baseline, cells, page_height):
    """
    Calculate confidence score for a baseline.

    Metrics:
    - Support: How many cells align to this baseline?
    - Coverage: What percentage of page height is covered?
    - Consistency: How tight is the alignment (std dev)?

    Returns:
        confidence: float (0.0 to 1.0)
        metrics: dict with details
    """
    position = baseline["position"]
    orientation = baseline["orientation"]  # "left" or "right"
    tolerance = baseline.get("tolerance", 2.0)

    # Find supporting cells
    supporting_cells = []
    for cell in cells:
        bbox = cell["bbox"]
        cell_edge = bbox["x0"] if orientation == "left" else bbox["x1"]

        if abs(cell_edge - position) <= tolerance:
            supporting_cells.append(cell)

    # Metric 1: Support count (normalized)
    support_count = len(supporting_cells)
    support_score = min(support_count / 10.0, 1.0)  # 10+ cells = full score

    # Metric 2: Vertical coverage
    if supporting_cells:
        y_min = min(c["bbox"]["y0"] for c in supporting_cells)
        y_max = max(c["bbox"]["y1"] for c in supporting_cells)
        coverage = (y_max - y_min) / page_height
        coverage_score = min(coverage / 0.5, 1.0)  # 50% coverage = full score
    else:
        coverage_score = 0.0

    # Metric 3: Alignment consistency (lower std dev = higher score)
    if support_count >= 2:
        edges = [
            c["bbox"]["x0"] if orientation == "left" else c["bbox"]["x1"]
            for c in supporting_cells
        ]
        std_dev = np.std(edges)
        consistency_score = max(0.0, 1.0 - (std_dev / 5.0))  # 5pt std = 0 score
    else:
        consistency_score = 1.0 if support_count == 1 else 0.0

    # Weighted average
    confidence = (
        0.4 * support_score +
        0.3 * coverage_score +
        0.3 * consistency_score
    )

    metrics = {
        "support_count": support_count,
        "coverage": coverage_score,
        "consistency": consistency_score,
        "std_dev": std_dev if support_count >= 2 else None,
    }

    return confidence, metrics
```

##### 2. Baseline Filtering

**Function:** `filter_baselines_by_confidence()`
**Location:** `alignment_detection.py`

**Algorithm:**
```python
def filter_baselines_by_confidence(
    baselines: List[Dict[str, Any]],
    cells: List[Dict[str, Any]],
    page_height: float,
    min_confidence: float = 0.7
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter baselines by confidence score.

    Returns:
        strong_baselines: List of baselines above threshold
        weak_baselines: List of baselines below threshold (for debugging)
    """
    strong_baselines = []
    weak_baselines = []

    for baseline in baselines:
        confidence, metrics = calculate_baseline_confidence(
            baseline, cells, page_height
        )

        # Add confidence to baseline
        baseline["confidence"] = confidence
        baseline["confidence_metrics"] = metrics

        if confidence >= min_confidence:
            strong_baselines.append(baseline)
        else:
            weak_baselines.append(baseline)

    return strong_baselines, weak_baselines
```

##### 3. Baseline Merging

**Function:** `merge_redundant_baselines()`
**Location:** `alignment_detection.py`

**Algorithm:**
```python
def merge_redundant_baselines(
    baselines: List[Dict[str, Any]],
    merge_tolerance: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Merge baselines that are very close together (within tolerance).

    Strategy:
    - Group baselines by orientation (left vs right)
    - For each group, find pairs within merge_tolerance
    - Merge pairs: keep stronger one or weighted average
    """
    if not baselines:
        return []

    # Separate by orientation
    left_baselines = [b for b in baselines if b["orientation"] == "left"]
    right_baselines = [b for b in baselines if b["orientation"] == "right"]

    merged = []

    for group in [left_baselines, right_baselines]:
        # Sort by position
        group_sorted = sorted(group, key=lambda b: b["position"])

        i = 0
        while i < len(group_sorted):
            current = group_sorted[i]

            # Find all baselines within tolerance
            cluster = [current]
            j = i + 1
            while j < len(group_sorted):
                if abs(group_sorted[j]["position"] - current["position"]) <= merge_tolerance:
                    cluster.append(group_sorted[j])
                    j += 1
                else:
                    break

            # Merge cluster
            if len(cluster) == 1:
                merged.append(current)
            else:
                # Keep strongest baseline or weighted average
                strongest = max(cluster, key=lambda b: b.get("confidence", 0.0))

                # Weighted average position
                total_confidence = sum(b.get("confidence", 0.5) for b in cluster)
                avg_position = sum(
                    b["position"] * b.get("confidence", 0.5)
                    for b in cluster
                ) / total_confidence

                merged_baseline = {
                    "position": avg_position,
                    "orientation": current["orientation"],
                    "confidence": max(b.get("confidence", 0.0) for b in cluster),
                    "merged_from": len(cluster),
                }
                merged.append(merged_baseline)

            i = j if len(cluster) > 1 else i + 1

    return merged
```

##### 4. Baseline-to-Block Refinement

**Function:** `refine_baselines_with_blocks()`
**Location:** `alignment_detection.py`

**Algorithm:**
```python
def refine_baselines_with_blocks(
    baselines: List[Dict[str, Any]],
    blocks: List[Dict[str, Any]],
    snap_tolerance: float = 3.0
) -> List[Dict[str, Any]]:
    """
    Refine baseline positions using strong text blocks.

    For high-confidence blocks with clear alignment:
    - Left-aligned block → snap nearby left baseline to block's left edge
    - Right-aligned block → snap nearby right baseline to block's right edge

    This is the INVERSE of block-to-baseline snapping (UC-001).
    """
    refined_baselines = baselines.copy()

    for block in blocks:
        alignment_type = block.get("alignment_type")
        alignment_confidence = block.get("alignment_confidence", 0.0)

        # Only refine with high-confidence blocks
        if alignment_confidence < 0.8:
            continue

        bbox = block["bbox"]

        if alignment_type == "left":
            # Find nearby left baselines
            block_left = bbox["x0"]

            for baseline in refined_baselines:
                if baseline["orientation"] == "left":
                    if abs(baseline["position"] - block_left) <= snap_tolerance:
                        # Snap baseline to block edge
                        baseline["position"] = block_left
                        baseline["snapped_to_block"] = block["block_id"]

        elif alignment_type == "right":
            # Find nearby right baselines
            block_right = bbox["x1"]

            for baseline in refined_baselines:
                if baseline["orientation"] == "right":
                    if abs(baseline["position"] - block_right) <= snap_tolerance:
                        # Snap baseline to block edge
                        baseline["position"] = block_right
                        baseline["snapped_to_block"] = block["block_id"]

    return refined_baselines
```

##### 5. Integration with Detection Pipeline

**Changes to:** `pdf_ocr_detector.py`

```python
# In detect_docling_alignment_anomalies()

# STEP 1: Detect baselines (existing)
baselines = detect_alignment_baselines(consolidated_cells, page_no, {})

# STEP 2: Calculate confidence scores (NEW)
page_height = get_page_height(page_no)  # Need to add this
for baseline in baselines:
    confidence, metrics = calculate_baseline_confidence(
        baseline, consolidated_cells, page_height
    )
    baseline["confidence"] = confidence
    baseline["confidence_metrics"] = metrics

# STEP 3: Filter weak baselines (NEW)
strong_baselines, weak_baselines = filter_baselines_by_confidence(
    baselines, consolidated_cells, page_height,
    min_confidence=0.7  # Configurable parameter
)

# STEP 4: Merge redundant baselines (NEW)
merged_baselines = merge_redundant_baselines(strong_baselines, merge_tolerance=2.0)

# STEP 5: Detect text blocks using strong baselines
text_blocks = identify_text_blocks_iterative(
    consolidated_cells, merged_baselines, debug={}
)

# STEP 6: Refine baselines with blocks (NEW - circular refinement)
refined_baselines = refine_baselines_with_blocks(
    merged_baselines, text_blocks, snap_tolerance=3.0
)

# STEP 7: Store for visualization
debug_output = {
    "original_baselines": len(baselines),
    "weak_baselines_filtered": len(weak_baselines),
    "baselines_after_merge": len(merged_baselines),
    "baselines_after_refinement": len(refined_baselines),
    "weak_baselines": weak_baselines,  # For debugging
}
```

##### 6. Configuration Parameters

**Add to:** `alignment_detection.py` (constants section)

```python
# Baseline Confidence Thresholds
MIN_BASELINE_CONFIDENCE = 0.7           # Filter baselines below this
BASELINE_MERGE_TOLERANCE = 2.0          # Merge baselines within this distance
BASELINE_SNAP_TOLERANCE = 3.0           # Snap baselines to blocks within this

# Confidence Metric Weights
BASELINE_SUPPORT_WEIGHT = 0.4           # Weight for cell count
BASELINE_COVERAGE_WEIGHT = 0.3          # Weight for vertical coverage
BASELINE_CONSISTENCY_WEIGHT = 0.3       # Weight for alignment tightness

# Support thresholds
BASELINE_MIN_SUPPORT_FOR_FULL_SCORE = 10    # 10+ cells = max support score
BASELINE_MIN_COVERAGE_FOR_FULL_SCORE = 0.5  # 50% page = max coverage score
BASELINE_MAX_STDDEV_FOR_FULL_SCORE = 5.0    # 5pt std dev = 0 consistency score
```

#### Expected Output

**Debug Information:**
```json
{
  "baseline_processing": {
    "original_baselines_detected": 15,
    "weak_baselines_filtered": 8,
    "baselines_after_merge": 5,
    "baselines_after_block_refinement": 5,

    "weak_baselines": [
      {
        "position": 125.3,
        "orientation": "left",
        "confidence": 0.42,
        "confidence_metrics": {
          "support_count": 2,
          "coverage": 0.15,
          "consistency": 0.85,
          "std_dev": 0.8
        },
        "reason": "Low support and coverage"
      }
    ],

    "strong_baselines": [
      {
        "position": 72.0,
        "orientation": "left",
        "confidence": 0.91,
        "confidence_metrics": {
          "support_count": 18,
          "coverage": 0.68,
          "consistency": 0.94,
          "std_dev": 0.3
        },
        "snapped_to_block": 3,
        "merged_from": 2
      },
      {
        "position": 450.2,
        "orientation": "right",
        "confidence": 0.96,
        "confidence_metrics": {
          "support_count": 23,
          "coverage": 0.72,
          "consistency": 0.98,
          "std_dev": 0.1
        }
      }
    ]
  }
}
```

**Visualization:**
- Strong baselines (confidence ≥ 0.9): Solid green line
- Medium baselines (0.7 ≤ confidence < 0.9): Semi-transparent blue line
- Weak baselines (confidence < 0.7): Not rendered (filtered out)
- Baseline tooltip: "Left baseline at 72.0pt (confidence: 0.91, support: 18 cells, coverage: 68%)"

#### Testing Validation

**Test Document:** `False FNB Statements - Mrs. Winnie Sokhela.pdf`

**Expected Results:**
1. ✅ All baselines have confidence scores and metrics
2. ✅ Weak baselines in Description column filtered out (e.g., 8 of 15 removed)
3. ✅ Strong left baseline at Description column's left edge (confidence ≥ 0.9)
4. ✅ Redundant baselines within 2pt merged together
5. ✅ Left baseline snaps to left-aligned Description block's left edge
6. ✅ Debug output shows: original count → filtered count → merged count
7. ✅ Visualization shows only strong baselines with opacity based on confidence
8. ✅ Configurable `MIN_BASELINE_CONFIDENCE` parameter works (test with 0.6, 0.7, 0.8)

#### Relationship to UC-001

**UC-001**: Block-to-Baseline Snapping
- Blocks snap TO baselines (baseline is authority)
- Use case: Right-aligned Amount column

**UC-002**: Baseline-to-Block Refinement
- Baselines snap TO blocks (block is authority)
- Use case: Left-aligned Description column with strong block edges

**Integration**: Both should work together:
1. Detect baselines → calculate confidence
2. Filter weak baselines
3. Merge redundant baselines
4. Detect text blocks using strong baselines (blocks may snap to baselines per UC-001)
5. Refine baselines using high-confidence blocks (baselines snap to blocks per UC-002)
6. Optional: Iterate steps 4-5 for mutual refinement

---

**End of Use Case UC-002**

---

### UC-003: Table-Based Block Splitting for Mixed Alignment Columns

**Document:** `85385859429 – Mr MP Madiya – False pay slip.pdf`
**Category:** Table Structure Analysis & Block Splitting
**Priority:** HIGH
**Status:** Not Implemented

#### Problem Description

A single text block contains both an "Amount" column (right-aligned numbers with heading) and a "Description" column (left-aligned text with heading). While acceptable in isolation, the vertical baselines and table structure indicate these should be separate blocks due to opposing alignment types.

**Visual Evidence (Attachment 1):**
- Purple block spans both Amount and Description columns
- Amount column: Right-aligned numbers (e.g., "R 30,000.00", "3000.00")
- Description column: Left-aligned text (e.g., "TAX", "U.I.F", "PENSION FUND", "MEDICAL AID")
- Vertical baselines clearly separate the two columns
- Table structure has distinct column boundaries

**Key Observations:**
- Current clustering treats this as one cohesive block (horizontally continuous)
- Vertical baselines run between the columns, indicating separation
- Docling table detection likely identifies this as a 2-column table
- Different alignment types (left vs right) within same block
- Column headers present: "Amount" (right-aligned) and "Description" (left-aligned)

#### Current Behavior

1. **Single Block for Multi-Column Table:**
   - Clustering algorithm creates one block spanning both columns
   - Horizontal proximity triggers grouping (h_eps threshold)
   - No consideration of table structure or column boundaries
   - No consideration of opposing alignment types

2. **No Table-Informed Splitting:**
   - Vertical baselines detected but not used for block splitting
   - Docling table information available but not integrated
   - No logic to say "this block spans multiple table columns → split it"

3. **No Alignment Conflict Detection:**
   - System doesn't detect that block contains both left and right aligned content
   - No flag for "mixed alignment within block"
   - Mixed alignment is a strong signal for block splitting

4. **Missing Column Header Detection:**
   - Column headers ("Amount", "Description") not used to inform structure
   - Headers could provide alignment hints (right-aligned header → right-aligned data)

#### Desired Behavior

1. **Docling Table Integration:**
   - Use Docling's table detection as authoritative structure
   - If block spans multiple table columns → split by column boundaries
   - Each table column becomes a separate text block
   - Preserve table metadata (column index, header text)

2. **Vertical Baseline as Split Trigger:**
   - Strong vertical baselines between content → split candidates
   - If vertical baseline has high confidence and runs through middle of block → split
   - Use baseline confidence (from UC-002) to determine split necessity

3. **Alignment Conflict Detection:**
   - Analyze cells within block for alignment homogeneity
   - If block contains ≥70% right-aligned AND ≥70% left-aligned cells → flag as mixed
   - Mixed alignment → strong signal to split
   - Calculate alignment distribution: `{left: 45%, right: 50%, center: 5%}`

4. **Column Header-Informed Splitting:**
   - Detect column headers within or above block
   - Use header alignment to predict column data alignment
   - "Amount" header (right-aligned) → expect right-aligned data below
   - "Description" header (left-aligned) → expect left-aligned data below

5. **Output Structure:**
   - Block 1: Amount column (right-aligned, with header)
   - Block 2: Description column (left-aligned, with header)
   - Each block has `table_column_id` metadata
   - Each block has homogeneous alignment type

#### Implementation Requirements

##### 1. Docling Table Extraction

**Function:** `extract_docling_tables()`
**Location:** `pdf_ocr_detector.py` (new function)

**Algorithm:**
```python
def extract_docling_tables(docling_payload, page_no):
    """
    Extract table structure from Docling payload.

    Returns:
        tables: List of table metadata with column boundaries
    """
    parsed_pages = docling_payload.get("parsed_pages", {})
    page_data = parsed_pages.get(str(page_no), {})

    # Docling stores tables in page data
    tables = page_data.get("tables", [])

    extracted = []
    for table_idx, table in enumerate(tables):
        # Extract column information
        columns = []
        for col_idx, column in enumerate(table.get("columns", [])):
            col_info = {
                "column_index": col_idx,
                "header": column.get("header", ""),
                "cells": column.get("cells", []),
                "bbox": calculate_column_bbox(column["cells"]),
                "alignment": infer_column_alignment(column),
            }
            columns.append(col_info)

        extracted.append({
            "table_id": table_idx,
            "bbox": table.get("bbox"),
            "columns": columns,
            "row_count": len(table.get("rows", [])),
        })

    return extracted

def infer_column_alignment(column):
    """Infer alignment from column header or cell content."""
    header = column.get("header", "").lower()

    # Header text hints
    if "amount" in header or "total" in header or "balance" in header:
        return "right"
    elif "description" in header or "name" in header or "item" in header:
        return "left"

    # Analyze cell content
    cells = column.get("cells", [])
    numeric_cells = sum(1 for c in cells if is_numeric(c.get("text", "")))

    if numeric_cells / len(cells) >= 0.8:  # 80% numeric
        return "right"
    else:
        return "left"
```

##### 2. Table-Based Block Splitting

**Function:** `split_blocks_by_table_structure()`
**Location:** `text_block_detection.py` (new function)

**Algorithm:**
```python
def split_blocks_by_table_structure(
    blocks: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    min_overlap: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Split text blocks that span multiple table columns.

    Args:
        blocks: Text blocks from clustering
        tables: Table structure from Docling
        min_overlap: Minimum bbox overlap to associate block with table

    Returns:
        split_blocks: Blocks split by table column boundaries
    """
    split_blocks = []

    for block in blocks:
        block_bbox = block["bbox"]

        # Find overlapping table
        overlapping_table = None
        for table in tables:
            if bbox_overlap_ratio(block_bbox, table["bbox"]) >= min_overlap:
                overlapping_table = table
                break

        if not overlapping_table:
            # No table overlap - keep original block
            split_blocks.append(block)
            continue

        # Check how many table columns this block spans
        spanning_columns = []
        for column in overlapping_table["columns"]:
            col_bbox = column["bbox"]
            if bbox_overlap_ratio(block_bbox, col_bbox) >= 0.5:
                spanning_columns.append(column)

        if len(spanning_columns) <= 1:
            # Block is within single column - keep original
            block["table_id"] = overlapping_table["table_id"]
            block["table_column_id"] = spanning_columns[0]["column_index"] if spanning_columns else None
            split_blocks.append(block)
        else:
            # Block spans multiple columns - SPLIT
            for column in spanning_columns:
                col_bbox = column["bbox"]

                # Extract cells from original block that fall in this column
                column_cells = [
                    cell for cell in block.get("cells", [])
                    if bbox_overlap_ratio(cell["bbox"], col_bbox) >= 0.5
                ]

                if not column_cells:
                    continue

                # Create new block for this column
                new_block = {
                    "block_id": f"{block['block_id']}_col{column['column_index']}",
                    "bbox": calculate_bbox(column_cells),
                    "cells": column_cells,
                    "line_count": count_lines(column_cells),
                    "cell_count": len(column_cells),
                    "dominant_alignment": column["alignment"],  # Use column alignment
                    "table_id": overlapping_table["table_id"],
                    "table_column_id": column["column_index"],
                    "table_column_header": column["header"],
                    "split_from_block": block["block_id"],
                    "texts": [c.get("text", "") for c in column_cells[:10]],
                }
                split_blocks.append(new_block)

    return split_blocks
```

##### 3. Alignment Conflict Detection

**Function:** `detect_alignment_conflicts()`
**Location:** `text_block_detection.py`

**Algorithm:**
```python
def detect_alignment_conflicts(
    block: Dict[str, Any],
    baselines: List[Dict[str, Any]],
    conflict_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Detect if a block has mixed/conflicting alignments.

    Args:
        block: Text block to analyze
        baselines: Vertical baselines for alignment detection
        conflict_threshold: Minimum percentage for each type to flag conflict

    Returns:
        conflict_info: Alignment distribution and conflict status
    """
    cells = block.get("cells", [])
    if not cells:
        return {"has_conflict": False}

    # Count alignments
    left_aligned = 0
    right_aligned = 0
    center_aligned = 0

    for cell in cells:
        bbox = cell["bbox"]

        # Check left baselines
        for baseline in baselines:
            if baseline["orientation"] == "left":
                if abs(bbox["x0"] - baseline["position"]) <= 3.0:
                    left_aligned += 1
                    break

        # Check right baselines
        for baseline in baselines:
            if baseline["orientation"] == "right":
                if abs(bbox["x1"] - baseline["position"]) <= 3.0:
                    right_aligned += 1
                    break

    total = len(cells)
    left_pct = left_aligned / total
    right_pct = right_aligned / total
    center_pct = 1.0 - (left_pct + right_pct)

    # Conflict if multiple alignment types exceed threshold
    alignment_types = []
    if left_pct >= conflict_threshold:
        alignment_types.append("left")
    if right_pct >= conflict_threshold:
        alignment_types.append("right")
    if center_pct >= conflict_threshold:
        alignment_types.append("center")

    has_conflict = len(alignment_types) >= 2

    return {
        "has_conflict": has_conflict,
        "alignment_distribution": {
            "left": left_pct,
            "right": right_pct,
            "center": center_pct,
        },
        "conflicting_types": alignment_types if has_conflict else [],
        "suggestion": "split_by_alignment" if has_conflict else "no_action",
    }
```

##### 4. Vertical Baseline Split Detection

**Function:** `detect_baseline_split_candidates()`
**Location:** `alignment_detection.py`

**Algorithm:**
```python
def detect_baseline_split_candidates(
    blocks: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]],
    min_confidence: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Detect blocks that should be split based on vertical baselines running through them.

    Args:
        blocks: Text blocks to analyze
        baselines: Vertical baselines with confidence scores
        min_confidence: Minimum baseline confidence to trigger split

    Returns:
        split_candidates: List of blocks with split recommendations
    """
    candidates = []

    for block in blocks:
        bbox = block["bbox"]
        x0, x1 = bbox["x0"], bbox["x1"]

        # Find baselines that run through middle of block (not at edges)
        internal_baselines = []
        for baseline in baselines:
            if baseline.get("confidence", 0.0) < min_confidence:
                continue

            pos = baseline["position"]

            # Check if baseline is internal (not at block edges)
            edge_tolerance = 5.0  # pt
            if (pos > x0 + edge_tolerance) and (pos < x1 - edge_tolerance):
                internal_baselines.append(baseline)

        if internal_baselines:
            # This block has strong vertical baseline(s) running through it
            candidates.append({
                "block": block,
                "split_baselines": internal_baselines,
                "recommendation": "split_by_baseline",
                "reason": f"{len(internal_baselines)} strong vertical baseline(s) run through block interior",
            })

    return candidates
```

##### 5. Integration Pipeline

**Changes to:** `pdf_ocr_detector.py`

```python
# In detect_docling_alignment_anomalies()

# EXISTING: Baseline detection and filtering (UC-002)
baselines = detect_alignment_baselines(consolidated_cells, page_no, {})
strong_baselines, weak_baselines = filter_baselines_by_confidence(baselines, ...)

# EXISTING: Text block detection
text_blocks = identify_text_blocks_iterative(consolidated_cells, strong_baselines, debug={})

# NEW: Extract table structure from Docling
tables = extract_docling_tables(docling_payload, page_no)

# NEW: Detect alignment conflicts within blocks
for block in text_blocks:
    conflict_info = detect_alignment_conflicts(block, strong_baselines)
    block["alignment_conflict"] = conflict_info

# NEW: Detect baseline split candidates
split_candidates = detect_baseline_split_candidates(text_blocks, strong_baselines)

# NEW: Split blocks by table structure
split_blocks = split_blocks_by_table_structure(text_blocks, tables, min_overlap=0.7)

# Store results
debug_output = {
    "tables_detected": len(tables),
    "blocks_before_split": len(text_blocks),
    "blocks_after_split": len(split_blocks),
    "alignment_conflicts_detected": sum(1 for b in text_blocks if b.get("alignment_conflict", {}).get("has_conflict")),
    "baseline_split_candidates": len(split_candidates),
}
```

#### Expected Output

**Before Split:**
```json
{
  "block_id": "block_3",
  "bbox": {"x0": 50.0, "y0": 200.0, "x1": 350.0, "y1": 450.0},
  "line_count": 15,
  "dominant_alignment": "mixed",
  "texts": ["Amount", "R 30,000.00", "3000.00", "Description", "TAX", "U.I.F"],
  "alignment_conflict": {
    "has_conflict": true,
    "alignment_distribution": {"left": 0.47, "right": 0.53},
    "conflicting_types": ["left", "right"],
    "suggestion": "split_by_alignment"
  }
}
```

**After Split:**
```json
[
  {
    "block_id": "block_3_col0",
    "bbox": {"x0": 50.0, "y0": 200.0, "x1": 150.0, "y1": 450.0},
    "line_count": 8,
    "dominant_alignment": "right",
    "table_id": 0,
    "table_column_id": 0,
    "table_column_header": "Amount",
    "split_from_block": "block_3",
    "texts": ["Amount", "R 30,000.00", "3000.00", "..."],
    "alignment_conflict": {"has_conflict": false}
  },
  {
    "block_id": "block_3_col1",
    "bbox": {"x0": 200.0, "y0": 200.0, "x1": 350.0, "y1": 450.0},
    "line_count": 7,
    "dominant_alignment": "left",
    "table_id": 0,
    "table_column_id": 1,
    "table_column_header": "Description",
    "split_from_block": "block_3",
    "texts": ["Description", "TAX", "U.I.F", "PENSION FUND", "..."],
    "alignment_conflict": {"has_conflict": false}
  }
]
```

#### Testing Validation

**Test Document:** `85385859429 – Mr MP Madiya – False pay slip.pdf`

**Expected Results:**
1. ✅ Docling table structure extracted with 2 columns (Amount, Description)
2. ✅ Single block spanning both columns detected
3. ✅ Alignment conflict detected (47% left, 53% right)
4. ✅ Block split into 2 blocks (one per column)
5. ✅ Amount block: right-aligned, contains header + numbers
6. ✅ Description block: left-aligned, contains header + text
7. ✅ Each block has `table_column_id` metadata
8. ✅ Visualization shows 2 separate purple blocks (not 1)

---

**End of Use Case UC-003**

---

### UC-004: Header Misalignment Detection (Advanced Case)

**Document:** `85385859429 – Mr MP Madiya – False pay slip.pdf`
**Category:** Advanced Anomaly Detection
**Priority:** MEDIUM (Future Consideration)
**Status:** Not Implemented (Radar Only)

#### Problem Description

In a table with right-aligned amounts, the column header ("Amount") is not perfectly aligned with the data values below it. This creates a subtle misalignment that is somewhat suspicious, though counterevidence exists (horizontal alignment intact, consistent font).

**Visual Evidence (Attachment 2):**
- Column header: "Amount"
- Data values: "R 4,783.08", "R 177.12", "R 787.10", "R 627.20"
- Header appears slightly offset from data column alignment
- Horizontal baseline through data is intact
- Font appears consistent between header and data

**Key Observations:**
- **Suspicious:** Header not aligned to column data baseline
- **Counterevidence:**
  - Horizontal alignment preserved (values on same y-line)
  - Font consistency maintained
  - May be intentional design choice (centered header over right-aligned data)

#### Current Behavior

No detection logic exists for header-to-data alignment mismatches.

#### Desired Behavior (Future)

1. **Header-Data Alignment Analysis:**
   - For each table column, compare header alignment to data alignment
   - Right-aligned data → expect right-aligned header (or centered)
   - Left-aligned data → expect left-aligned header (or centered)
   - Flag: "Header alignment differs from data alignment"

2. **Confidence Scoring:**
   - Consider counterevidence:
     - Font consistency: ✓ → reduce suspicion
     - Horizontal alignment: ✓ → reduce suspicion
     - Baseline deviation magnitude: Small → lower severity
   - Final score: LOW, MEDIUM, or HIGH suspicion

3. **Context-Aware Detection:**
   - Centered headers over aligned data → common pattern, LOW suspicion
   - Completely misaligned header (left header, right data) → HIGH suspicion
   - Subtle offset → MEDIUM suspicion

#### Implementation Notes

**Complexity:** HIGH
- Requires distinguishing between intentional design and forgery
- Many false positives likely (centered headers are common)
- Needs sophisticated context analysis

**Dependencies:**
- UC-001: Alignment type detection
- UC-003: Table structure extraction
- Font analysis (already available in system)
- Baseline detection

**Recommendation:**
- Keep in radar for future consideration
- Implement UC-001, UC-002, UC-003 first
- Gather more examples of this pattern
- May require machine learning for reliable detection

#### Potential Future Algorithm

```python
def detect_header_data_alignment_mismatch(table, baselines):
    """
    Detect misalignment between column headers and data.

    Note: Advanced case - many edge cases and false positives.
    """
    anomalies = []

    for column in table["columns"]:
        header = column["header"]
        data_cells = column["cells"][1:]  # Exclude header

        # Get header alignment
        header_alignment = detect_cell_alignment(header, baselines)

        # Get data alignment (majority)
        data_alignments = [detect_cell_alignment(c, baselines) for c in data_cells]
        data_alignment = most_common(data_alignments)

        # Check for mismatch
        if header_alignment != data_alignment and header_alignment != "center":
            # Calculate counterevidence
            font_consistent = check_font_consistency(header, data_cells)
            horizontal_aligned = check_horizontal_alignment(data_cells)
            deviation_magnitude = calculate_alignment_deviation(header, data_cells, baselines)

            # Score suspicion
            suspicion = "MEDIUM"
            if font_consistent and horizontal_aligned and deviation_magnitude < 5.0:
                suspicion = "LOW"  # Likely design choice
            elif not font_consistent or deviation_magnitude > 15.0:
                suspicion = "HIGH"  # More suspicious

            anomalies.append({
                "type": "header_data_alignment_mismatch",
                "column": column["column_index"],
                "header_alignment": header_alignment,
                "data_alignment": data_alignment,
                "suspicion": suspicion,
                "counterevidence": {
                    "font_consistent": font_consistent,
                    "horizontal_aligned": horizontal_aligned,
                    "deviation_magnitude": deviation_magnitude,
                },
            })

    return anomalies
```

#### Status

**Current:** Not planned for immediate implementation
**Reason:** High complexity, many false positives expected, requires more examples
**Action:** Document as future consideration, revisit after UC-001/002/003 implemented

---

**End of Use Case UC-004**

---

### UC-005: Block Refinement and Conflict Resolution

**Document:** `85385859429 – Mr MP Madiya – false ABSA statements..pdf`
**Category:** Block Quality & Conflict Resolution
**Priority:** HIGH
**Status:** Not Implemented

#### Problem Description

The current clustering produces multiple block-level issues that require refinement:

1. **Adjacent Blocks That Should Merge** - Separate Amount and Balance column blocks with same right alignment should combine into single block
2. **Engulfed/Nested Blocks** - Smaller blocks completely contained within larger blocks (doesn't make sense)
3. **Competing Block Hierarchies** - Sometimes the large block should be kept, sometimes the smaller blocks make more sense (need confidence scoring)
4. **Weak Evidence Long-Distance Blocks** - Narrow box spanning from "absa" logo down to "Uncleared Cheques" with weak evidence due to distance

**Visual Evidence:**

**Screenshot 1:** Amount and Balance columns
- Two separate purple blocks for Amount and Balance columns
- Both have right alignment
- Both are in same table structure
- Should be combined into single right-aligned block

**Screenshots 2 & 3:** Engulfed blocks
- Small text blocks completely inside larger blocks
- Large block: full table area
- Small blocks: individual cells or rows
- Small blocks are redundant (engulfed by larger block)

**Screenshot 4:** Competing hierarchies
- Same area has both large block AND smaller blocks
- Sometimes large block is correct (coherent table)
- Sometimes smaller blocks are correct (distinct sections)
- Need confidence metric to decide which to keep

**Screenshot 4:** Weak long-distance block
- Narrow purple box from "absa" logo to "Uncleared Cheques"
- Vertical distance is large (~300-400pt)
- Evidence for grouping is weak (just horizontal alignment?)
- Significant IoU with better blocks
- Weak block should be removed

#### Current Behavior

1. **No Block Merging:**
   - Adjacent blocks with same alignment not merged
   - Table columns remain as separate blocks even when coherent
   - Missing: Logic to merge compatible adjacent blocks

2. **No Nested Block Detection:**
   - Smaller blocks inside larger blocks are both kept
   - Creates redundant/conflicting block representations
   - Missing: IoU-based nesting detection

3. **No Block Confidence Scoring:**
   - All blocks treated equally
   - No way to prefer large block vs small blocks
   - Missing: Quality metrics (cell count, alignment coherence, table membership)

4. **No Weak Evidence Filtering:**
   - Long-distance blocks with weak support are kept
   - Large vertical span with few cells not flagged as suspicious
   - Missing: Evidence strength calculation (cells-per-vertical-distance ratio)

#### Desired Behavior

1. **Block Merging for Adjacent Same-Alignment Blocks:**
   - Detect adjacent blocks with same alignment type
   - If both are in same table (per Docling), check for merge eligibility
   - Merge if: same alignment, adjacent (gap < threshold), same table
   - Example: Amount + Balance columns → single right-aligned block

2. **Nested Block Elimination:**
   - Calculate IoU (Intersection over Union) for all block pairs
   - If IoU(small, large) > 0.8 and area(small) < 0.5 × area(large) → nested
   - Keep larger block, remove smaller (engulfed) block
   - Exception: If small block has higher confidence score, keep both

3. **Block Confidence Scoring:**
   - Score blocks based on quality metrics:
     - Cell count (more cells = higher confidence)
     - Alignment coherence (% cells matching dominant alignment)
     - Table membership (inside Docling table = higher)
     - Cell density (cells per area)
     - Vertical span reasonableness (not too tall for few cells)
   - Use confidence to resolve conflicts (nested blocks, overlapping blocks)

4. **Weak Evidence Block Removal:**
   - Calculate evidence strength: cells_count / vertical_span
   - If strength < threshold AND IoU with stronger block > 0.3 → remove weak block
   - Example: 5 cells spanning 400pt = 0.0125 cells/pt (very weak)
   - Compare to typical block: 20 cells spanning 100pt = 0.2 cells/pt (strong)

5. **Conflict Resolution Strategy:**
   ```
   For overlapping blocks A and B:
   - If IoU > 0.8: Nested → keep higher confidence block
   - If IoU > 0.3 and confidence_A > confidence_B × 1.5: Remove B (weak evidence)
   - If adjacent and same alignment and same table: Merge
   - Otherwise: Keep both
   ```

#### Implementation Requirements

##### 1. Block Confidence Scoring

**Function:** `calculate_block_confidence()`
**Location:** `text_block_detection.py` (new function)

**Algorithm:**
```python
def calculate_block_confidence(
    block: Dict[str, Any],
    tables: List[Dict[str, Any]],
    page_height: float
) -> float:
    """
    Calculate confidence score for a text block.

    Metrics:
    - Cell count (more is better)
    - Alignment coherence (dominant alignment %)
    - Table membership (in table = bonus)
    - Cell density (cells per area)
    - Vertical span reasonableness

    Returns:
        confidence: float (0.0 to 1.0)
    """
    cell_count = block.get("cell_count", 0)
    bbox = block["bbox"]
    width = bbox["x1"] - bbox["x0"]
    height = bbox["y1"] - bbox["y0"]
    area = width * height

    # Metric 1: Cell count (normalized)
    cell_score = min(cell_count / 20.0, 1.0)  # 20+ cells = full score

    # Metric 2: Alignment coherence
    alignment_confidence = block.get("alignment_confidence", 0.5)
    coherence_score = alignment_confidence

    # Metric 3: Table membership
    table_bonus = 0.0
    for table in tables:
        table_bbox = table["bbox"]
        if bbox_overlap_ratio(bbox, table_bbox) >= 0.7:
            table_bonus = 0.2  # 20% bonus for being in table
            break

    # Metric 4: Cell density
    density = cell_count / area if area > 0 else 0
    # Typical density: 0.0001 to 0.001 cells/pt²
    density_score = min(density / 0.0005, 1.0)

    # Metric 5: Vertical span reasonableness
    vertical_span = height
    cells_per_point = cell_count / vertical_span if vertical_span > 0 else 0
    # Typical: 0.05 to 0.5 cells/pt
    span_score = min(cells_per_point / 0.1, 1.0)

    # Weighted average
    confidence = (
        0.25 * cell_score +
        0.25 * coherence_score +
        0.15 * density_score +
        0.15 * span_score +
        table_bonus  # Bonus, not weighted
    )

    return min(confidence, 1.0)
```

##### 2. Nested Block Detection

**Function:** `detect_nested_blocks()`
**Location:** `text_block_detection.py`

**Algorithm:**
```python
def detect_nested_blocks(
    blocks: List[Dict[str, Any]],
    iou_threshold: float = 0.8,
    size_ratio_threshold: float = 0.5
) -> List[Tuple[int, int, str]]:
    """
    Detect blocks that are nested (engulfed) within other blocks.

    Args:
        blocks: List of text blocks
        iou_threshold: Minimum IoU to consider nested
        size_ratio_threshold: Max size ratio (small/large) to consider engulfed

    Returns:
        conflicts: List of (small_idx, large_idx, "engulfed") tuples
    """
    conflicts = []

    for i, block_a in enumerate(blocks):
        for j, block_b in enumerate(blocks):
            if i >= j:
                continue

            bbox_a = block_a["bbox"]
            bbox_b = block_b["bbox"]

            area_a = (bbox_a["x1"] - bbox_a["x0"]) * (bbox_a["y1"] - bbox_a["y0"])
            area_b = (bbox_b["x1"] - bbox_b["x0"]) * (bbox_b["y1"] - bbox_b["y0"])

            # Determine which is smaller
            if area_a < area_b:
                small_idx, large_idx = i, j
                small_area, large_area = area_a, area_b
                small_bbox, large_bbox = bbox_a, bbox_b
            else:
                small_idx, large_idx = j, i
                small_area, large_area = area_b, area_a
                small_bbox, large_bbox = bbox_b, bbox_a

            # Check if small is engulfed by large
            iou = calculate_iou(small_bbox, large_bbox)
            size_ratio = small_area / large_area

            if iou >= iou_threshold and size_ratio <= size_ratio_threshold:
                conflicts.append((small_idx, large_idx, "engulfed"))

    return conflicts
```

##### 3. Weak Evidence Block Detection

**Function:** `detect_weak_evidence_blocks()`
**Location:** `text_block_detection.py`

**Algorithm:**
```python
def detect_weak_evidence_blocks(
    blocks: List[Dict[str, Any]],
    min_strength: float = 0.05,
    iou_threshold: float = 0.3
) -> List[Tuple[int, str]]:
    """
    Detect blocks with weak evidence (low cell density, large vertical span).

    Args:
        blocks: List of text blocks
        min_strength: Minimum cells-per-vertical-distance ratio
        iou_threshold: Minimum IoU with other blocks to flag for removal

    Returns:
        weak_blocks: List of (block_idx, reason) tuples
    """
    weak_blocks = []

    for i, block in enumerate(blocks):
        bbox = block["bbox"]
        cell_count = block.get("cell_count", 0)
        vertical_span = bbox["y1"] - bbox["y0"]

        # Calculate evidence strength
        strength = cell_count / vertical_span if vertical_span > 0 else 0

        if strength < min_strength:
            # Check if overlaps with stronger blocks
            has_stronger_overlap = False

            for j, other in enumerate(blocks):
                if i == j:
                    continue

                other_bbox = other["bbox"]
                other_span = other_bbox["y1"] - other_bbox["y0"]
                other_strength = other.get("cell_count", 0) / other_span if other_span > 0 else 0

                iou = calculate_iou(bbox, other_bbox)

                if iou >= iou_threshold and other_strength > strength * 1.5:
                    has_stronger_overlap = True
                    break

            if has_stronger_overlap:
                weak_blocks.append((
                    i,
                    f"Weak evidence: {cell_count} cells / {vertical_span:.1f}pt = {strength:.4f} cells/pt"
                ))

    return weak_blocks
```

##### 4. Block Merging for Adjacent Blocks

**Function:** `merge_adjacent_blocks()`
**Location:** `text_block_detection.py`

**Algorithm:**
```python
def merge_adjacent_blocks(
    blocks: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    max_gap: float = 20.0
) -> List[Dict[str, Any]]:
    """
    Merge adjacent blocks with same alignment in same table.

    Args:
        blocks: List of text blocks
        tables: Table structure from Docling
        max_gap: Maximum horizontal gap to consider adjacent

    Returns:
        merged_blocks: Blocks after merging
    """
    merged = []
    merged_flags = [False] * len(blocks)

    for i, block_a in enumerate(blocks):
        if merged_flags[i]:
            continue

        # Start merge group with this block
        merge_group = [i]
        alignment_a = block_a.get("dominant_alignment")
        table_a = get_block_table(block_a, tables)

        # Find adjacent blocks to merge
        for j, block_b in enumerate(blocks):
            if i == j or merged_flags[j]:
                continue

            alignment_b = block_b.get("dominant_alignment")
            table_b = get_block_table(block_b, tables)

            # Check merge eligibility
            if alignment_a != alignment_b:
                continue
            if table_a != table_b:  # Must be in same table (or both not in table)
                continue

            # Check adjacency (horizontal)
            bbox_a = block_a["bbox"]
            bbox_b = block_b["bbox"]

            # Horizontal gap
            gap = min(
                abs(bbox_a["x1"] - bbox_b["x0"]),
                abs(bbox_b["x1"] - bbox_a["x0"])
            )

            # Vertical overlap
            y_overlap = min(bbox_a["y1"], bbox_b["y1"]) - max(bbox_a["y0"], bbox_b["y0"])
            y_total = max(bbox_a["y1"] - bbox_a["y0"], bbox_b["y1"] - bbox_b["y0"])
            overlap_ratio = y_overlap / y_total if y_total > 0 else 0

            if gap <= max_gap and overlap_ratio >= 0.5:
                merge_group.append(j)

        # Merge the group
        if len(merge_group) > 1:
            merged_block = merge_block_group([blocks[idx] for idx in merge_group])
            merged.append(merged_block)
            for idx in merge_group:
                merged_flags[idx] = True
        else:
            merged.append(block_a)
            merged_flags[i] = True

    return merged

def get_block_table(block, tables):
    """Find which table this block belongs to."""
    bbox = block["bbox"]
    for table_idx, table in enumerate(tables):
        if bbox_overlap_ratio(bbox, table["bbox"]) >= 0.7:
            return table_idx
    return None  # Not in any table
```

##### 5. Conflict Resolution Pipeline

**Function:** `resolve_block_conflicts()`
**Location:** `text_block_detection.py`

**Algorithm:**
```python
def resolve_block_conflicts(
    blocks: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    page_height: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Resolve all block conflicts: nested, weak evidence, merging.

    Pipeline:
    1. Calculate confidence scores for all blocks
    2. Detect and resolve nested blocks
    3. Detect and remove weak evidence blocks
    4. Merge adjacent compatible blocks

    Returns:
        refined_blocks: Blocks after conflict resolution
        resolution_log: Details of what was changed
    """
    # Step 1: Calculate confidence
    for block in blocks:
        block["confidence"] = calculate_block_confidence(block, tables, page_height)

    # Step 2: Detect nested blocks
    nested_conflicts = detect_nested_blocks(blocks, iou_threshold=0.8)

    # Resolve: Keep higher confidence block
    remove_indices = set()
    for small_idx, large_idx, reason in nested_conflicts:
        small_conf = blocks[small_idx]["confidence"]
        large_conf = blocks[large_idx]["confidence"]

        if large_conf >= small_conf:
            remove_indices.add(small_idx)
        else:
            remove_indices.add(large_idx)

    # Step 3: Detect weak evidence blocks
    weak_blocks = detect_weak_evidence_blocks(blocks, min_strength=0.05)
    for block_idx, reason in weak_blocks:
        remove_indices.add(block_idx)

    # Remove flagged blocks
    filtered_blocks = [b for i, b in enumerate(blocks) if i not in remove_indices]

    # Step 4: Merge adjacent blocks
    merged_blocks = merge_adjacent_blocks(filtered_blocks, tables, max_gap=20.0)

    # Build resolution log
    resolution_log = {
        "original_blocks": len(blocks),
        "nested_conflicts_resolved": len(nested_conflicts),
        "weak_blocks_removed": len(weak_blocks),
        "blocks_after_filtering": len(filtered_blocks),
        "blocks_after_merging": len(merged_blocks),
        "removed_blocks": [
            {
                "index": idx,
                "reason": next((r for i, r in weak_blocks if i == idx), "nested/engulfed"),
                "bbox": blocks[idx]["bbox"],
                "texts": blocks[idx].get("texts", [])[:3],
            }
            for idx in remove_indices
        ],
    }

    return merged_blocks, resolution_log
```

##### 6. Integration Pipeline

**Changes to:** `pdf_ocr_detector.py`

```python
# In detect_docling_alignment_anomalies()

# EXISTING: Baseline detection, filtering, text blocks
baselines = detect_alignment_baselines(consolidated_cells, page_no, {})
strong_baselines, weak_baselines = filter_baselines_by_confidence(baselines, ...)
text_blocks = identify_text_blocks_iterative(consolidated_cells, strong_baselines, debug={})

# EXISTING (UC-003): Table extraction and splitting
tables = extract_docling_tables(docling_payload, page_no)
split_blocks = split_blocks_by_table_structure(text_blocks, tables, min_overlap=0.7)

# NEW (UC-005): Block conflict resolution
refined_blocks, resolution_log = resolve_block_conflicts(
    split_blocks, tables, page_height
)

# Store results
debug_output = {
    "blocks_initial": len(text_blocks),
    "blocks_after_split": len(split_blocks),
    "blocks_after_refinement": len(refined_blocks),
    "resolution_log": resolution_log,
}
```

#### Expected Output

**Before Refinement:**
```json
{
  "blocks": [
    {"block_id": "block_1", "bbox": {...}, "texts": ["Amount"], "cell_count": 5},
    {"block_id": "block_2", "bbox": {...}, "texts": ["Balance"], "cell_count": 5},
    {"block_id": "block_3_large", "bbox": {...}, "texts": ["Transaction History"], "cell_count": 20},
    {"block_id": "block_3_small", "bbox": {...}, "texts": ["NAME"], "cell_count": 2},
    {"block_id": "block_4_weak", "bbox": {...}, "texts": ["absa", "Uncleared"], "cell_count": 3, "vertical_span": 400}
  ]
}
```

**After Refinement:**
```json
{
  "blocks": [
    {
      "block_id": "block_1_merged",
      "bbox": {...},
      "texts": ["Amount", "Balance"],
      "cell_count": 10,
      "dominant_alignment": "right",
      "confidence": 0.85,
      "merged_from": ["block_1", "block_2"]
    },
    {
      "block_id": "block_3_large",
      "bbox": {...},
      "texts": ["Transaction History"],
      "cell_count": 20,
      "confidence": 0.92
    }
  ],
  "resolution_log": {
    "original_blocks": 5,
    "nested_conflicts_resolved": 1,
    "weak_blocks_removed": 1,
    "blocks_after_filtering": 3,
    "blocks_after_merging": 2,
    "removed_blocks": [
      {"index": 3, "reason": "nested/engulfed", "texts": ["NAME"]},
      {"index": 4, "reason": "Weak evidence: 3 cells / 400.0pt = 0.0075 cells/pt", "texts": ["absa", "Uncleared"]}
    ]
  }
}
```

#### Testing Validation

**Test Document:** `85385859429 – Mr MP Madiya – false ABSA statements..pdf`

**Expected Results:**
1. ✅ Amount + Balance blocks merged into single right-aligned block
2. ✅ Engulfed small blocks (screenshots 2 & 3) removed
3. ✅ Block confidence scores calculated for all blocks
4. ✅ Weak "absa" → "Uncleared Cheques" block removed (low cells/pt ratio + IoU overlap)
5. ✅ Competing hierarchies resolved based on confidence (screenshot 4)
6. ✅ Resolution log shows: merged count, removed count, reasons
7. ✅ Visualization shows refined blocks (no nested/weak blocks)

#### Relationship to Other Use Cases

**UC-003**: Table-Based Block Splitting
- UC-003 splits blocks that span multiple columns
- UC-005 merges blocks that should be together (inverse operation)
- Both use Docling table structure as authority

**UC-002**: Baseline Confidence & Filtering
- Same concept (confidence-based filtering) applied to blocks instead of baselines
- Both use evidence strength metrics (support, coverage, consistency)

**Integration**: UC-005 runs AFTER UC-003:
1. UC-003: Split multi-column blocks → more granular blocks
2. UC-005: Merge adjacent same-alignment blocks → consolidate coherent regions
3. UC-005: Remove nested/weak blocks → clean up conflicts

---

**End of Use Case UC-005**

---

### UC-006: Table-Informed Block Creation

**Document:** `85393423298 – Mr G Motau - False pay slip.pdf`
**Category:** Table Structure Integration & Block Creation
**Priority:** HIGH
**Status:** Not Implemented

#### Problem Description

Text blocks created by clustering do not respect table column structure, leading to blocks that span across columns or one large block covering an entire table. Docling provides table structure information that could guide block creation, but this must be done without overfitting hyperparameters to specific documents.

**Visual Evidence:**

**Screenshot 1:** Employee information table
- Table columns: "Employee Code", "Pay Method", "Identity Number", "Branch Code", "Account Number"
- Purple text blocks span horizontally across multiple columns
- Blocks don't align with column boundaries
- Should have separate blocks for each column (or logical groupings)

**Screenshot 2:** Financial data table
- Large table with label-value pairs (left-aligned labels, right-aligned values)
- Labels: "Total earnings", "Nett pay", "YTD Totals", "Taxable earnings", "Tax paid", etc.
- One massive purple block covers entire table area
- Should have multiple blocks: one for labels column, one for values column (or rows)

**Key Observations:**
- Clustering algorithm doesn't "see" table structure
- Horizontal proximity (h_eps=40.0) causes cells in different columns to cluster together
- Large tables with consistent vertical spacing create one big vertical band
- Need table-aware clustering, not just geometric clustering
- Must maintain consistent hyperparameters across all documents (avoid overfitting)

#### Current Behavior

1. **Pure Geometric Clustering:**
   - Clustering uses only x,y coordinates
   - No awareness of table column/row structure
   - Proximity-based grouping ignores semantic boundaries (columns)

2. **Table Structure Ignored:**
   - Docling provides rich table metadata (columns, rows, cells)
   - This information is not used during clustering
   - Only used AFTER clustering for splitting (UC-003)

3. **No Hybrid Approach:**
   - Either pure clustering OR table splitting
   - Missing: Table-guided clustering (use table as prior/constraint)

4. **Overfitting Risk:**
   - Could tune h_eps differently per document
   - BUT this would overfit to specific layouts
   - Need generalizable approach

#### Desired Behavior

1. **Table-Guided Clustering:**
   - If Docling detects table at location → use table structure to constrain clustering
   - Cells in different table columns should not cluster together (even if geometrically close)
   - Cells in same table column are more likely to cluster together

2. **Hybrid Approach:**
   - Start with table structure from Docling
   - Apply clustering WITHIN each column separately
   - Merge columns if they have same alignment and are logically related

3. **Fallback to Pure Clustering:**
   - Areas not in tables → use pure geometric clustering (existing approach)
   - Ensures consistent behavior across all document types
   - Table detection is optional enhancement, not requirement

4. **Consistent Hyperparameters:**
   - Use same v_eps=12.0, h_eps=40.0 for all documents
   - Table structure acts as constraint, not parameter override
   - Avoids overfitting to specific document layouts

5. **Column-Aware Distance Metric:**
   ```python
   # If cells are in same table:
   if cell_a.table_id == cell_b.table_id:
       if cell_a.column_id != cell_b.column_id:
           distance = INFINITY  # Different columns → never cluster
       else:
           distance = euclidean(cell_a, cell_b)  # Same column → normal distance
   else:
       distance = euclidean(cell_a, cell_b)  # Not in table → normal distance
   ```

#### Implementation Requirements

##### 1. Table-Aware Cell Metadata

**Function:** `enrich_cells_with_table_info()`
**Location:** `pdf_ocr_detector.py` (new function)

**Algorithm:**
```python
def enrich_cells_with_table_info(
    cells: List[Dict[str, Any]],
    tables: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Add table metadata to cells that are inside tables.

    For each cell:
    - Check which table it belongs to (bbox overlap)
    - Determine which column it's in
    - Determine which row it's in
    - Add table_id, column_id, row_id to cell metadata

    Args:
        cells: OCR cells from Docling
        tables: Table structure from Docling

    Returns:
        enriched_cells: Cells with table metadata added
    """
    enriched = []

    for cell in cells:
        cell_bbox = cell["bbox"]
        cell_enriched = cell.copy()

        # Default: not in any table
        cell_enriched["table_id"] = None
        cell_enriched["column_id"] = None
        cell_enriched["row_id"] = None

        # Find containing table
        for table_idx, table in enumerate(tables):
            table_bbox = table["bbox"]

            if bbox_contains(table_bbox, cell_bbox):
                cell_enriched["table_id"] = table_idx

                # Find containing column
                for col_idx, column in enumerate(table["columns"]):
                    col_bbox = column["bbox"]
                    if bbox_overlap_ratio(cell_bbox, col_bbox) >= 0.5:
                        cell_enriched["column_id"] = col_idx
                        break

                # Find containing row (optional)
                for row_idx, row in enumerate(table.get("rows", [])):
                    row_bbox = row.get("bbox")
                    if row_bbox and bbox_overlap_ratio(cell_bbox, row_bbox) >= 0.5:
                        cell_enriched["row_id"] = row_idx
                        break

                break  # Cell can only be in one table

        enriched.append(cell_enriched)

    return enriched
```

##### 2. Table-Aware Distance Function

**Function:** `table_aware_distance()`
**Location:** `text_block_detection.py` (new function)

**Algorithm:**
```python
def table_aware_distance(
    cell_a: Dict[str, Any],
    cell_b: Dict[str, Any],
    dimension: str  # "vertical" or "horizontal"
) -> float:
    """
    Calculate distance between cells with table-awareness.

    If both cells are in the same table but different columns:
    - Return INFINITY for horizontal clustering (prevent cross-column clustering)
    - Return normal distance for vertical clustering (allow rows across columns)

    Args:
        cell_a, cell_b: Cells to compare
        dimension: "vertical" (y-axis) or "horizontal" (x-axis)

    Returns:
        distance: Float or np.inf
    """
    table_a = cell_a.get("table_id")
    table_b = cell_b.get("table_id")
    col_a = cell_a.get("column_id")
    col_b = cell_b.get("column_id")

    # Both cells in same table?
    if table_a is not None and table_a == table_b:
        # Different columns?
        if col_a is not None and col_b is not None and col_a != col_b:
            # For horizontal clustering: prevent cross-column grouping
            if dimension == "horizontal":
                return np.inf
            # For vertical clustering: allow (to detect rows)
            # Fall through to normal distance

    # Normal euclidean distance
    bbox_a = cell_a["bbox"]
    bbox_b = cell_b["bbox"]

    if dimension == "vertical":
        # Y-coordinate distance
        center_a = (bbox_a["y0"] + bbox_a["y1"]) / 2.0
        center_b = (bbox_b["y0"] + bbox_b["y1"]) / 2.0
        return abs(center_a - center_b)
    else:  # horizontal
        # X-coordinate distance
        center_a = (bbox_a["x0"] + bbox_a["x1"]) / 2.0
        center_b = (bbox_b["x0"] + bbox_b["x1"]) / 2.0
        return abs(center_a - center_b)
```

##### 3. Table-Aware DBSCAN Clustering

**Function:** `dbscan_with_table_awareness()`
**Location:** `text_block_detection.py` (new function)

**Algorithm:**
```python
def dbscan_with_table_awareness(
    cells: List[Dict[str, Any]],
    dimension: str,  # "vertical" or "horizontal"
    eps: float,
    min_samples: int = 2
) -> np.ndarray:
    """
    Run DBSCAN clustering with table-aware distance metric.

    Args:
        cells: Cells to cluster (must have table metadata)
        dimension: "vertical" or "horizontal"
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        labels: Cluster labels for each cell
    """
    if not cells:
        return np.array([])

    # Build custom distance matrix
    n = len(cells)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = table_aware_distance(cells[i], cells[j], dimension)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Run DBSCAN with precomputed distance matrix
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed"
    )
    labels = clustering.fit_predict(distance_matrix)

    return labels
```

##### 4. Modified Iterative Clustering

**Function:** `identify_text_blocks_iterative()` (MODIFIED)
**Location:** `text_block_detection.py`

**Changes:**
```python
def identify_text_blocks_iterative(
    cells: List[Dict[str, Any]],
    baselines: List[Dict[str, Any]],
    tables: List[Dict[str, Any]] = None,  # NEW PARAMETER
    vertical_eps: float = ITERATIVE_VERTICAL_EPS,
    horizontal_eps: float = ITERATIVE_HORIZONTAL_EPS,
    min_samples: int = ITERATIVE_MIN_SAMPLES,
    max_iterations: int = ITERATIVE_MAX_ITERATIONS,
    min_block_cells: int = ITERATIVE_MIN_BLOCK_CELLS,
    use_table_awareness: bool = True,  # NEW PARAMETER
    debug: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Identify text blocks using iterative multi-pass clustering.

    NEW: Supports table-aware clustering if tables are provided.
    """
    # NEW: Enrich cells with table metadata
    if use_table_awareness and tables:
        cells = enrich_cells_with_table_info(cells, tables)
        debug["table_awareness"] = True
    else:
        debug["table_awareness"] = False

    # ... existing initialization ...

    # Main iteration loop
    for iteration in range(1, max_iterations + 1):
        dimension = "vertical" if iteration % 2 == 1 else "horizontal"
        eps = vertical_eps if dimension == "vertical" else horizontal_eps

        # ... existing split detection logic ...

        for cluster_cells in current_blocks:
            if len(cluster_cells) < min_block_cells:
                continue

            # NEW: Use table-aware DBSCAN if enabled
            if use_table_awareness and tables:
                labels = dbscan_with_table_awareness(
                    cluster_cells, dimension, eps, min_samples
                )
            else:
                # Original DBSCAN (coordinate-only)
                coords = extract_coords(cluster_cells, dimension)
                labels = dbscan(coords, eps, min_samples)

            # ... rest of existing logic ...
```

##### 5. Integration Pipeline

**Changes to:** `pdf_ocr_detector.py`

```python
# In detect_docling_alignment_anomalies()

# STEP 1: Extract table structure (UC-003)
tables = extract_docling_tables(docling_payload, page_no)

# STEP 2: Baseline detection and filtering (UC-002)
baselines = detect_alignment_baselines(consolidated_cells, page_no, {})
strong_baselines, weak_baselines = filter_baselines_by_confidence(baselines, ...)

# STEP 3: Table-aware text block detection (UC-006 - NEW)
text_blocks = identify_text_blocks_iterative(
    consolidated_cells,
    strong_baselines,
    tables=tables,  # Pass table structure
    use_table_awareness=True,  # Enable table-aware clustering
    debug={}
)

# STEP 4: Table-based block splitting (UC-003)
# NOTE: This may be redundant if UC-006 works well
split_blocks = split_blocks_by_table_structure(text_blocks, tables, min_overlap=0.7)

# STEP 5: Block conflict resolution (UC-005)
refined_blocks, resolution_log = resolve_block_conflicts(split_blocks, tables, page_height)
```

##### 6. Configuration & Tuning

**Add to:** `text_block_detection.py` (constants section)

```python
# Table-Aware Clustering
USE_TABLE_AWARENESS = True                    # Enable table-aware clustering
TABLE_COLUMN_CROSS_PENALTY = np.inf          # Distance penalty for cross-column clustering
TABLE_OVERLAP_THRESHOLD = 0.5                # Min overlap to assign cell to column

# Hyperparameters remain consistent across all documents
ITERATIVE_VERTICAL_EPS = 12.0                # No change
ITERATIVE_HORIZONTAL_EPS = 40.0              # No change
ITERATIVE_MIN_SAMPLES = 2                    # No change
```

**Key Design Principle:**
- Hyperparameters (eps values) stay constant across all documents
- Table awareness is a constraint/enhancement, not parameter tuning
- Documents without tables fall back to pure geometric clustering

#### Expected Output

**Before Table-Awareness (Current):**
```json
{
  "blocks": [
    {
      "block_id": "block_1",
      "bbox": {"x0": 50, "y0": 100, "x1": 800, "y1": 120},
      "texts": ["Employee Code", "9060", "Pay Method", "ACB"],
      "dominant_alignment": "mixed",
      "spans_multiple_columns": true
    },
    {
      "block_id": "block_2",
      "bbox": {"x0": 50, "y0": 300, "x1": 800, "y1": 500},
      "texts": ["Total earnings", "46,358.88", "Nett pay", "36,207.18", ...],
      "dominant_alignment": "mixed",
      "cell_count": 50,
      "is_giant_block": true
    }
  ]
}
```

**After Table-Awareness (UC-006):**
```json
{
  "blocks": [
    {
      "block_id": "block_1_col0",
      "bbox": {"x0": 50, "y0": 100, "x1": 200, "y1": 120},
      "texts": ["Employee Code"],
      "dominant_alignment": "left",
      "table_id": 0,
      "column_id": 0
    },
    {
      "block_id": "block_1_col1",
      "bbox": {"x0": 250, "y0": 100, "x1": 350, "y1": 120},
      "texts": ["9060"],
      "dominant_alignment": "left",
      "table_id": 0,
      "column_id": 1
    },
    {
      "block_id": "block_1_col2",
      "bbox": {"x0": 400, "y0": 100, "x1": 550, "y1": 120},
      "texts": ["Pay Method"],
      "dominant_alignment": "left",
      "table_id": 0,
      "column_id": 2
    },
    {
      "block_id": "block_1_col3",
      "bbox": {"x0": 600, "y0": 100, "x1": 700, "y1": 120},
      "texts": ["ACB"],
      "dominant_alignment": "left",
      "table_id": 0,
      "column_id": 3
    },
    {
      "block_id": "block_2_labels",
      "bbox": {"x0": 50, "y0": 300, "x1": 400, "y1": 500},
      "texts": ["Total earnings", "Nett pay", "YTD Totals", ...],
      "dominant_alignment": "left",
      "table_id": 1,
      "column_id": 0
    },
    {
      "block_id": "block_2_values",
      "bbox": {"x0": 600, "y0": 300, "x1": 800, "y1": 500},
      "texts": ["46,358.88", "36,207.18", "Amount", ...],
      "dominant_alignment": "right",
      "table_id": 1,
      "column_id": 1
    }
  ],
  "debug": {
    "table_awareness": true,
    "cross_column_clusters_prevented": 127,
    "hyperparameters": {
      "v_eps": 12.0,
      "h_eps": 40.0,
      "note": "Same hyperparameters used for all documents"
    }
  }
}
```

#### Testing Validation

**Test Document:** `85393423298 – Mr G Motau - False pay slip.pdf`

**Expected Results:**
1. ✅ Screenshot 1: Employee info table splits into separate column blocks
2. ✅ Screenshot 2: Financial table splits into labels column + values column (not one giant block)
3. ✅ Hyperparameters (v_eps=12.0, h_eps=40.0) consistent across all documents
4. ✅ Documents without tables still work correctly (fallback to pure geometric)
5. ✅ Debug output shows: `table_awareness: true`, cross-column prevention count
6. ✅ Each block has table_id, column_id metadata when applicable
7. ✅ Visualization shows blocks aligned with table columns

**Overfitting Prevention:**
- Test on multiple documents with different table structures
- Verify same hyperparameters work across all test cases
- Monitor false negatives (tables that should be detected but aren't)
- Monitor false positives (non-tables treated as tables)

#### Relationship to Other Use Cases

**UC-003**: Table-Based Block Splitting
- UC-003: Post-clustering split (fix blocks that span columns)
- UC-006: Pre-clustering constraint (prevent spanning in first place)
- UC-006 is more efficient (prevents problem vs fixing it)
- UC-003 may still be needed as backup/redundancy

**UC-005**: Block Conflict Resolution
- UC-006 creates cleaner initial blocks
- UC-005 still needed for merging adjacent same-column blocks
- UC-005 nested block detection less needed (UC-006 reduces nesting)

**Integration Order:**
1. UC-006: Table-aware clustering → column-aligned blocks
2. UC-003: Split any remaining multi-column blocks (backup)
3. UC-005: Merge adjacent blocks, remove nested/weak blocks

#### Design Rationale

**Why table-aware distance metric?**
- Clean separation of concerns: hyperparameters stay constant, table structure adds constraints
- Works with existing DBSCAN implementation (precomputed distance matrix)
- Graceful degradation: no tables → normal clustering

**Why not just use UC-003 splitting?**
- Prevention better than cure: cleaner blocks from the start
- UC-003 splitting can create artifacts (partial cells, ragged edges)
- Table-aware clustering produces semantically meaningful blocks

**Why keep same hyperparameters?**
- Avoid overfitting to specific document layouts
- Easier to maintain and debug
- Table structure provides document-specific adaptation without parameter tuning

---

**End of Use Case UC-006**

---

*Additional use cases will be added below as provided...*

---

## Implementation Plan & Regression Strategy

**Created:** 2025-11-08
**Purpose:** Comprehensive plan to implement all pending use cases while ensuring no regressions on positive test cases

### Current Implementation Status

**✅ Implemented:**
- **UC-006:** Table-Informed Block Creation
  - Table-aware clustering using INFINITY distance penalty for cross-column cells
  - Functions: `extract_table_metadata()`, `enrich_cells_with_table_info()`, `table_aware_distance()`
  - Integrated into `identify_text_blocks_iterative()` with `use_table_awareness=True`
  - Test file: [test_table_aware.py](test_table_aware.py)

**❌ Not Implemented:**
- **UC-001:** Right-Aligned Column Deviation in Bank Statement
- **UC-002:** Redundant Vertical Baselines in Left-Aligned Column
- **UC-003:** Table-Based Block Splitting for Mixed Alignment Columns
- **UC-004:** Header Misalignment Detection (Low Priority - Future/Radar)
- **UC-005:** Block Refinement and Conflict Resolution

### Regression Test Coverage

**Test Documents (PTCs):**
1. **85385859429 – Mr MP Madiya – False pay slip.pdf**
   - PTC-001: Correct Table Structure Detection ✅
   - PTC-002: Correct Earnings Section Blocking ✅
   - PTC-003: Correct Numerical Value Blocking ✅

2. **85393423298 – Mr G Motau - legitimate Capitec Bank statements.pdf**
   - PTC-004: Correct Transaction Table Row Separation ✅
   - PTC-005: Correct Income/Transaction Item Separation ✅
   - PTC-006: Problematic Multi-Line Address Fragmentation ⚠️ (Known Issue - UC-005 should fix)

3. **87026407408 – Mr I Fakude - False pay slip.pdf**
   - PTC-007: Correct Multi-Column Table Structure ✅
   - PTC-008: Correct Contact Information Blocking ✅
   - PTC-009: Correct Earnings/Deductions Table Blocking ✅

**Total:** 9 test cases (8 working correctly, 1 known issue)

### Key Decisions & Questions

Before proceeding with implementation, the following decisions must be documented:

#### Question 1: Implementation Order and Dependencies

**Issue:** UC-001 (alignment anomaly detection) depends on high-quality baselines from UC-002 (baseline refinement). Should we implement UC-002 first?

**Options:**
- **Option A:** UC-002 → UC-001 (baseline refinement first, then anomaly detection)
  - Pro: UC-001 gets clean baselines as input
  - Pro: Logical dependency order
  - Con: UC-001 benefits delayed

- **Option B:** UC-001 + UC-002 together (mutual refinement)
  - Pro: Can implement iterative refinement loop from the start
  - Pro: Faster time to complete both
  - Con: More complex initial implementation

- **Option C:** UC-001 with current baselines, then UC-002 refinement later
  - Pro: Quick initial implementation of UC-001
  - Con: May produce false positives with noisy baselines
  - Con: Rework needed when UC-002 is added

**✅ DECISION:** Option A - UC-002 → UC-001 (baseline refinement first)
- Implement baseline confidence scoring, filtering, and merging first
- Then implement alignment anomaly detection using refined baselines
- Rationale: Clean baselines reduce false positives in anomaly detection

#### Question 2: UC-003 Priority with UC-006 Implemented

**Issue:** UC-006 (table-aware clustering) already prevents blocks from spanning table columns using INFINITY distance. Is UC-003 (post-clustering splitting) still needed?

**Options:**
- **Option A:** Implement UC-003 as backup/redundancy
  - Pro: Catches cases where UC-006 fails (e.g., table not detected by Docling)
  - Pro: Defense in depth
  - Con: Additional complexity

- **Option B:** Skip UC-003, rely on UC-006
  - Pro: Simpler codebase
  - Pro: UC-006 is more elegant (prevention vs cure)
  - Con: No fallback if UC-006 fails

- **Option C:** Defer UC-003 to Phase 2 (implement only if needed)
  - Pro: Start simple, add complexity only if empirically needed
  - Pro: Test UC-006 thoroughly first
  - Con: May need to revisit later

**✅ DECISION:** Option A - Implement UC-003 as backup/redundancy
- UC-006 table-aware clustering is primary mechanism
- UC-003 provides fallback when table information is not available from Docling
- Defense in depth: catches edge cases where UC-006 misses splits
- Rationale: Not all documents have detectable table structure; UC-003 ensures robustness

#### Question 3: UC-005 Priority for Regression Prevention

**Issue:** UC-005 (block refinement) addresses:
- Adjacent block merging (e.g., multi-line addresses in PTC-006)
- Nested block removal
- Weak evidence filtering
- Block confidence scoring

PTC-006 (multi-line address fragmentation) is a known issue that UC-005 should fix. Should UC-005 be high priority to resolve this?

**Options:**
- **Option A:** High priority - Implement UC-005 early
  - Pro: Fixes PTC-006 (multi-line addresses)
  - Pro: Improves block quality overall
  - Con: Complex implementation (merging, nesting, confidence)

- **Option B:** Medium priority - After UC-001 and UC-002
  - Pro: UC-001/UC-002 are more critical for anomaly detection
  - Pro: Can use refined baselines from UC-002 in UC-005
  - Con: PTC-006 remains broken longer

- **Option C:** Phased implementation - Block merging first, then rest
  - Pro: Fix PTC-006 quickly (just implement adjacent merging)
  - Pro: Incrementally add complexity
  - Con: Partial implementation may cause other issues

**✅ DECISION:** Option B - Medium priority (after UC-001 and UC-002)
- UC-001/UC-002 are more critical for anomaly detection capability
- Can use refined baselines from UC-002 in UC-005 merging logic
- PTC-006 (multi-line addresses) will remain fragmented until UC-005 is implemented
- Rationale: Establish foundation (baselines + anomaly detection) before refinement

#### Question 4: Regression Testing Thresholds

**Issue:** When implementing new logic, how much variation in block count/boundaries is acceptable before flagging a regression?

**Proposed Thresholds:**
- **Block count:** ±2 blocks (e.g., 13 blocks → 11-15 blocks acceptable)
  - Rationale: Minor merging/splitting may occur with refinements
  - Exception: PTC-006 should merge from 5-6 blocks → 1-2 blocks (intentional improvement)

- **Block bounding boxes:** ±3pt per edge (within baseline tolerance)
  - Rationale: Snapping to baselines may adjust edges slightly
  - Exception: Large changes (>10pt) require justification

- **Alignment classifications:** Exact match (left/right/center/mixed)
  - Rationale: Alignment type should be deterministic
  - Exception: Blocks transitioning from "mixed" to "left" or "right" (improvement)

- **Baseline count:** ±20% (e.g., 10 baselines → 8-12 baselines acceptable)
  - Rationale: UC-002 will filter weak baselines (intentional reduction)
  - Exception: Drastic reductions (>50%) require investigation

**✅ DECISION:** Thresholds accepted as starting point
- Block count: ±2 blocks acceptable
- Block bounding boxes: ±3pt per edge acceptable
- Alignment classifications: Exact match required (with improvements allowed)
- Baseline count: ±20% acceptable
- **Note:** Thresholds will be reviewed and adjusted based on empirical results during implementation
- **Monitoring:** Track all deviations and justify changes exceeding thresholds

#### Question 5: Automated Regression Testing

**Issue:** Should we create automated regression tests that run on all PTC documents before merging changes?

**Proposal:**
```python
# test_regression_suite.py

PTC_DOCUMENTS = [
    {
        "name": "85385859429 – Mr MP Madiya – False pay slip.pdf",
        "test_cases": ["PTC-001", "PTC-002", "PTC-003"],
        "expected_blocks": 9,  # baseline
        "expected_block_range": (7, 11),  # acceptable range
    },
    {
        "name": "85393423298 – Mr G Motau - legitimate Capitec Bank statements.pdf",
        "test_cases": ["PTC-004", "PTC-005"],
        "expected_blocks": 15,
        "expected_block_range": (13, 17),
    },
    {
        "name": "87026407408 – Mr I Fakude - False pay slip.pdf",
        "test_cases": ["PTC-007", "PTC-008", "PTC-009"],
        "expected_blocks": 13,
        "expected_block_range": (11, 15),
    },
]

def test_regression():
    """Run clustering on all PTC documents and verify no regressions."""
    for doc in PTC_DOCUMENTS:
        blocks = run_clustering(doc["name"])
        assert doc["expected_block_range"][0] <= len(blocks) <= doc["expected_block_range"][1]
        # Additional assertions for bbox, alignment, etc.
```

**✅ DECISION:** Yes - Implement automated regression test suite
- Create `test_regression_suite.py` with all 3 PTC documents
- Run automatically before merging changes to main branch
- Include assertions for: block counts, bbox tolerances, alignment types, baseline counts
- Generate comparison reports showing before/after differences
- Store baseline results for each phase implementation
- Rationale: Essential for preventing regressions across 9 positive test cases

### Finalized Implementation Phases

Based on decisions above, the implementation will proceed in the following order:

#### Phase 1: Foundation - Baseline Quality (UC-002)

**Goal:** Improve baseline detection quality to support accurate anomaly detection

**Tasks:**
1. Implement baseline confidence scoring (support count, coverage, consistency)
2. Implement baseline filtering (remove weak baselines below threshold)
3. Implement baseline merging (combine redundant baselines within tolerance)
4. Implement baseline-to-block refinement (snap baselines to strong block edges)

**Deliverables:**
- Modified `detect_alignment_baselines()` in `alignment_detection.py`
- New functions: `calculate_baseline_confidence()`, `filter_baselines_by_confidence()`, `merge_redundant_baselines()`
- Debug output includes baseline confidence scores
- Visualization renders baselines with opacity based on confidence

**Regression Tests:**
- Run on all 3 PTC documents
- Verify block counts within acceptable range
- Verify no change to alignment classifications
- Document baseline count changes (expected reduction)

**Estimated Effort:** 6-8 hours

#### Phase 2: Alignment Attribution & Anomaly Detection (UC-001)

**Goal:** Add alignment type detection and detect alignment-based anomalies

**Tasks:**
1. Add alignment type attribution to blocks (left/right/center/mixed)
2. Implement baseline-aware block snapping (snap to baselines, not actual bbox)
3. Implement column alignment deviation detection
4. Implement content pattern consistency detection
5. Integrate with document viewer for anomaly visualization

**Deliverables:**
- Modified block structure with `alignment_type`, `alignment_confidence`, `alignment_baseline`
- Modified `identify_text_blocks_iterative()` to include alignment attribution
- New function: `detect_column_alignment_anomalies()` in `pdf_ocr_detector.py`
- New function: `detect_content_pattern_anomalies()`
- New function: `classify_content_type()`
- Anomaly report JSON structure

**Regression Tests:**
- Run on all 3 PTC documents
- Verify block bboxes within ±3pt tolerance
- Verify alignment types are correct (left/right as expected)
- Verify no false positive anomalies on PTC documents

**Estimated Effort:** 8-10 hours

#### Phase 3: Block Refinement (UC-005)

**Goal:** Improve block quality through merging, nesting removal, and weak evidence filtering

**Tasks:**
1. Implement block confidence scoring
2. Implement adjacent block merging (fix PTC-006 multi-line addresses)
3. Implement nested block detection and removal (IoU-based)
4. Implement weak evidence block filtering
5. Implement conflict resolution strategy

**Deliverables:**
- New function: `calculate_block_confidence()` in `text_block_detection.py`
- New function: `merge_adjacent_blocks()`
- New function: `remove_nested_blocks()`
- New function: `filter_weak_evidence_blocks()`
- New function: `resolve_block_conflicts()`
- Modified `identify_text_blocks_iterative()` to include refinement step

**Regression Tests:**
- Run on all 3 PTC documents
- Verify PTC-006 multi-line addresses merge correctly (5-6 blocks → 1-2 blocks)
- Verify no over-merging on PTC-001 through PTC-009
- Verify block counts within acceptable range

**Estimated Effort:** 10-12 hours

#### Phase 4: Table-Based Splitting (UC-003)

**Goal:** Add post-clustering table-based splitting as backup to UC-006

**Decision:** Implement as fallback mechanism for documents without detectable table structure

**Tasks:**
1. Implement table-based block splitting detection
2. Implement vertical baseline as split trigger
3. Implement alignment conflict detection
4. Integration with UC-006 table metadata

**Deliverables:**
- New function: `split_blocks_by_table_columns()` in `text_block_detection.py`
- New function: `detect_alignment_conflicts()`
- Modified pipeline to run splitting after initial clustering

**Regression Tests:**
- Run on all 3 PTC documents
- Verify no over-splitting on correctly blocked tables
- Document split decisions in debug output

**Estimated Effort:** 6-8 hours

### Testing Strategy

**Regression Test Execution:**
1. Before implementing each phase, capture baseline results for all PTC documents:
   - Block count
   - Block bboxes
   - Alignment types
   - Baseline count
   - Visual screenshots

2. After implementing each phase:
   - Run clustering on all PTC documents
   - Compare results to baseline
   - Document changes and justify deviations
   - Update screenshots if intentional improvements

3. Regression criteria:
   - Block counts within defined ranges
   - Bboxes within ±3pt tolerance
   - Alignment types preserved or improved (mixed → left/right)
   - No new false positives
   - PTC-006 must improve (multi-line address merging)

**Smoke Test Execution:**
1. Create parameterized smoke tests for each phase
2. Test on documents NOT in PTC set (generalization)
3. Document parameter sensitivity and edge cases

### Implementation Decisions - FINALIZED

All key decisions have been made and documented above:

1. **✅ UC-002 vs UC-001 Order:** UC-002 first (baseline refinement before anomaly detection)
2. **✅ UC-003 Priority:** Implement as backup/redundancy for when table information unavailable
3. **✅ UC-005 Priority:** Medium priority (after UC-001/UC-002)
4. **✅ Regression Thresholds:** Accepted as starting point, adjust as needed
5. **✅ Automated Testing:** Yes, implement `test_regression_suite.py`
6. **✅ Additional Test Documents:** Work with 3 PTC documents for now

**Status:** Plan finalized and ready for implementation

### Risk Mitigation

**Risk 1: Over-Merging (UC-005)**
- Mitigation: Conservative merging criteria (same alignment + same table + gap < threshold)
- Mitigation: Visual inspection of merged blocks
- Fallback: Configurable merge threshold

**Risk 2: Over-Filtering Baselines (UC-002)**
- Mitigation: Configurable confidence threshold (default: 0.7, tunable: 0.5-0.9)
- Mitigation: Debug output tracks filtered baselines
- Fallback: Ability to disable filtering if too aggressive

**Risk 3: False Positive Anomalies (UC-001)**
- Mitigation: High deviation thresholds (e.g., >10pt deviation to flag)
- Mitigation: Content pattern detection requires high consistency (>80% same type)
- Fallback: Severity levels (LOW/MEDIUM/HIGH) for user filtering

**Risk 4: Table Detection Failures (UC-006 dependency)**
- Mitigation: UC-003 as backup splitting mechanism
- Mitigation: Graceful degradation to pure geometric clustering
- Monitoring: Track Docling table detection rate

---

## Implementation Readiness Summary

**Status:** ✅ Plan finalized and ready for implementation

**Total Estimated Effort:** 30-38 hours across 4 phases

**Phase Breakdown:**
- Phase 1 (UC-002): 6-8 hours - Baseline confidence scoring and filtering
- Phase 2 (UC-001): 8-10 hours - Alignment attribution and anomaly detection
- Phase 3 (UC-005): 10-12 hours - Block refinement (merging, nesting, weak evidence)
- Phase 4 (UC-003): 6-8 hours - Table-based splitting as backup

**Testing Infrastructure:**
- Automated regression test suite: `test_regression_suite.py`
- 3 PTC documents with 9 positive test cases
- Baseline capture before each phase
- Regression validation with defined thresholds

**Risk Mitigation:**
- Configurable thresholds for all filtering/merging operations
- Debug output tracking all decisions
- Visual inspection at each phase
- Fallback mechanisms for edge cases

**Next Steps:**
1. Capture baseline results for all 3 PTC documents (current state)
2. Begin Phase 1: UC-002 implementation (baseline confidence scoring)
3. Run regression tests after Phase 1
4. Proceed to Phase 2, 3, 4 sequentially with regression testing between each

---

**End of Implementation Plan & Regression Strategy**
