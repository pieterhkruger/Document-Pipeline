# Hidden Text Detection – Flattening Segmentation Findings

## Summary
- The hidden-text detector compares Docling payloads extracted before and after PDF flattening.
- For `85385859429 – Mr MP Madiya – false ABSA statements..pdf`, flattening forces Docling to re-run OCR and emit **word-level** tokens, while the original payload preserves **phrase-level** spans pulled from the native PDF structure.
- Because `find_cell_in_visible_payload` (`pdf_ocr_detector.py`) requires an exact normalized text match **before** checking bounding boxes, any original span containing multiple words is reported as “not in flattened,” even when the corresponding words exist in the visible payload.
- The payloads themselves are correct; the mismatch stems from segmentation changes introduced during flattening.

## Evidence
| Metric (page 0) | Original payload | Visible-only payload | Notes |
| --------------- | ---------------- | -------------------- | ----- |
| Cell count | 119 | 207 | 1.74× increase after flattening |
| Average words per non-blank cell | 1.96 | 1.32 | 32% drop (multi-word spans split) |
| Multi-word cell share | 42.0% | 15.5% | Majority of phrases split into single tokens |
| Cells with `from_ocr = True` | 6 | 207 | Docling switches from native text extraction to OCR |
| Original cells missing exact match | 389 / 394 checked | — | Almost every span is flagged hidden |

Comparison document (`False Pay Slips – MR SIPHIWE M DHLAMIN.pdf`) that does **not** surface the issue:
| Metric (page 0) | Original payload | Visible-only payload | Notes |
| --------------- | ---------------- | -------------------- | ----- |
| Cell count | 118 | 86 | 0.73× (flattening trims duplicated content) |
| Average words per non-blank cell | 3.12 | 2.58 | 17% drop (modest change) |
| Multi-word cell share | 64.0% | 57.0% | Segmentation largely preserved |
| Cells with `from_ocr = True` | 0 | 86 | Full OCR, but without drastic token splitting |

The problematic case combines **massive cell proliferation** with a **sharp drop** in average words per cell, signalling that the matching routine will misclassify legitimate text as hidden.

## Planned Detection Logic
Goal: deterministically flag payload pairs where flattening introduces the “word-splitting” pattern so we can route them to more advanced matching later.

1. **Compute per-payload statistics** (ignoring blank cells):
   - `original_cell_count`, `visible_cell_count`
   - `average_words_original`, `average_words_visible`
   - `multi_word_share_original`, `multi_word_share_visible`
2. **Derive ratios**:
   - `cell_count_ratio = visible / original`
   - `average_word_ratio = average_words_visible / average_words_original` (guarded against division by zero)
3. **Flag as a segmentation mismatch** when both hold:
   - `cell_count_ratio ≥ 1.40` (≥40% more cells after flattening)
   - `average_word_ratio ≤ 0.75` (≤75% of the original average words per cell)
   These thresholds cover the ABSA case (1.74 ratio & 0.67 average-word ratio) while excluding the payslip example (0.73 ratio & 0.83 average-word ratio).
4. **Surface diagnostics** along with the boolean flag so downstream logic can decide whether to switch to enhanced matching.

This adds an inexpensive, deterministic classification step without touching UI components. Later work can hook the flag into alternative matching workflows.
