"""
Alignment Logic Reporter
========================

Collects detailed debugging data for alignment-related detections
and writes structured JSON reports per processed document.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class AlignmentLogicReporter:
    """Accumulates per-document diagnostic information."""

    def __init__(
        self,
        doc_label: str,
        pdf_path: Optional[Path],
        output_dir: Path,
    ) -> None:
        self.doc_label = doc_label or "document"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {
            "document": {
                "label": self.doc_label,
                "pdf_path": str(pdf_path) if pdf_path else None,
                "payloads": {},
                "generated_at": datetime.utcnow().isoformat() + "Z",
            },
            "pages": {},
            "notes": [],
        }
        self._dirty = False
        self._output_path: Optional[Path] = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_page(self, page_no: int) -> Dict[str, Any]:
        page_key = str(page_no)
        if page_key not in self.data["pages"]:
            self.data["pages"][page_key] = {
                "overview": {},
                "baselines": {},
                "text_blocks": {},
                "colon_spacing": {},
                "highlighted_items": [],
                "events": [],
            }
        return self.data["pages"][page_key]

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
        return slug or "document"

    # ------------------------------------------------------------------ #
    # Document-level metadata
    # ------------------------------------------------------------------ #
    def add_payload_reference(self, name: str, path: Optional[Path]) -> None:
        self.data["document"]["payloads"][name] = str(path) if path else None
        self._dirty = True

    def add_note(self, note: str) -> None:
        self.data["notes"].append(note)
        self._dirty = True

    # ------------------------------------------------------------------ #
    # Page-level logging
    # ------------------------------------------------------------------ #
    def record_page_overview(self, page_no: int, **overview: Any) -> None:
        page = self._ensure_page(page_no)
        page["overview"].update(overview)
        self._dirty = True

    def record_baseline_stage(
        self,
        page_no: int,
        stage: str,
        baselines: Any,
        debug_info: Optional[Dict[str, Any]],
    ) -> None:
        page = self._ensure_page(page_no)
        page["baselines"][stage] = {
            "baselines": baselines,
            "debug": debug_info,
        }
        self._dirty = True

    def record_text_block_stage(
        self,
        page_no: int,
        stage: str,
        debug_payload: Dict[str, Any],
    ) -> None:
        page = self._ensure_page(page_no)
        page["text_blocks"][stage] = debug_payload
        self._dirty = True

    def record_colon_analysis(
        self,
        page_no: int,
        colon_payload: Dict[str, Any],
    ) -> None:
        page = self._ensure_page(page_no)
        page["colon_spacing"] = colon_payload
        self._dirty = True

    def record_event(
        self,
        page_no: int,
        category: str,
        payload: Dict[str, Any],
    ) -> None:
        page = self._ensure_page(page_no)
        page["events"].append({"category": category, **payload})
        self._dirty = True

    def record_highlighted_item(
        self,
        page_no: int,
        item_type: str,
        item_data: Dict[str, Any],
    ) -> None:
        """
        Record an anomaly/highlighted item that appears in the UI.

        Args:
            page_no: Page number (0-indexed)
            item_type: Type of highlight (e.g., "colon_spacing_deviation", "horizontal_misalignment", "text_block")
            item_data: Dictionary containing item details (bbox, reason, classification, etc.)
        """
        page = self._ensure_page(page_no)
        page["highlighted_items"].append({
            "type": item_type,
            **item_data
        })
        self._dirty = True

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def write(self) -> Optional[Path]:
        """
        Write the accumulated report to disk if there were any changes.
        Returns the path to the generated JSON file or None if nothing was written.
        """
        if not self._dirty:
            return self._output_path

        slug = self._slugify(self.doc_label)
        self._output_path = self.output_dir / f"{slug}_alignment_logic.json"
        with self._output_path.open("w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)

        self._dirty = False
        return self._output_path


__all__ = ["AlignmentLogicReporter"]
