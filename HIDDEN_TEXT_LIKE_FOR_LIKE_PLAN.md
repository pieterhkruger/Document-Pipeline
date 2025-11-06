# Hidden Text Matching Parity Plan

## Goal
- Ensure original and flattened Docling payloads are compared on equivalent granularities.
- Reduce false “not_in_flattened” flags by matching word-to-word when both payloads expose word cells, otherwise fall back to phrase-level comparison with tolerance for OCR variances.

## Approach Overview
1. **Granularity Assessment**
   - Inspect `parsed_pages` cells for each payload.
   - Determine whether a page provides reliable word-level data (e.g., multiple short spans, `from_ocr` indicators).
2. **Like-for-like Matching**
   - When both payloads have word-level tokens: match tokens individually, using bounding-box overlap and normalized text (Levenshtein similarity ≥ threshold).
   - When one payload is phrase-level: aggregate overlapping words into phrases before comparing.
3. **Bounding Box Tolerance**
   - Allow configurable IoU/offset tolerances to account for OCR shifts.
4. **Result Attribution**
   - Track matching provenance (word-level vs phrase-level) to aid debugging.

## Implementation Checklist
- [x] Utility: payload granularity classifier (word vs phrase).
- [x] Word/phrase matching with Levenshtein fallback and bbox tolerance.
- [x] Integrate like-for-like logic into `find_cell_in_visible_payload` and comparison loops.
- [ ] Unit-style smoke test using existing payload pairs (e.g., `_ 3 month bank statements`).
- [ ] Telemetry: extend detection metadata to record match mode counts.

## Progress Log
- *(done - awaiting commit)* Document created outlining strategy. Commit: `docs: outline hidden text like-for-like plan` (d05e51c).
- *(in progress)* Implemented granularity analysis helpers and fuzzy matching integration in `pdf_ocr_detector.py`. Next: add tests/telemetry.

> **Reminder:** Update the checklist and progress log as each task completes so we can resume seamlessly if tokens run out.
