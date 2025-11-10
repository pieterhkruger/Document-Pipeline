"""
Spatial grid utilities for accelerated bbox lookups and clustering.

This module centralizes the uniform spatial grid approach that was
previously embedded inside `pdf_ocr_detector.py` and prototype notebooks.
It exposes a `SpatialGrid` dataclass that can:
    * Generate overlap pairs without missing any intersections.
    * Estimate typical spacing along horizontal/vertical directions.
    * Perform simple single-link clustering for text runs with direction-aware
      constraints (projection overlap + gap thresholds).

The implementation is based on the GPT‑5 framework captured in
`Clustering_idea.py` and has been adapted for production use.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Literal, Optional, Tuple

Box = Tuple[float, float, float, float]  # (x0, y0, x1, y1)
OverlapTuple = Tuple[int, int, float]


def box_area(box: Box) -> float:
    """Return the positive area (clamped at zero) for a bbox tuple."""
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return width * height


def intersection_area(a: Box, b: Box) -> float:
    """Return area of intersection (strict >0 if overlapping)."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def one_d_overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    """Positive length of the 1D overlap between [a0, a1] and [b0, b1]."""
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0.0, hi - lo)


@dataclass
class SpatialGrid:
    """
    Uniform spatial grid for bbox queries.

    Parameters
    ----------
    boxes:
        List of axis-aligned bounding boxes (x0, y0, x1, y1).
    cell_w / cell_h:
        Optional overrides for grid cell size. Defaults to the median width
        / height of provided boxes, which is robust against outliers.
    """

    boxes: List[Box]
    cell_w: Optional[float] = None
    cell_h: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.boxes:
            # Minimal scaffolding so helpers continue to work
            self._grid: Dict[Tuple[int, int], List[int]] = {}
            self._cell_ranges: List[Tuple[int, int, int, int]] = []
            self._minx = self._miny = 0.0
            self._cw = self._ch = 1.0
            self._areas: List[float] = []
            return

        widths = [max(1e-12, box[2] - box[0]) for box in self.boxes]
        heights = [max(1e-12, box[3] - box[1]) for box in self.boxes]
        self._cw = self.cell_w or median(widths)
        self._ch = self.cell_h or median(heights)
        self._minx = min(box[0] for box in self.boxes)
        self._miny = min(box[1] for box in self.boxes)

        self._grid = defaultdict(list)
        self._cell_ranges = []
        for idx, box in enumerate(self.boxes):
            xr, yr = self._cells_for_box(box)
            self._cell_ranges.append((*xr, *yr))
            for cx in range(xr[0], xr[1] + 1):
                for cy in range(yr[0], yr[1] + 1):
                    self._grid[(cx, cy)].append(idx)

        self._areas = [box_area(box) for box in self.boxes]

    # ------------------------------------------------------------------ #
    # Core helpers
    # ------------------------------------------------------------------ #
    def _cells_for_box(self, box: Box) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        cw, ch, minx, miny = self._cw, self._ch, self._minx, self._miny
        cx0 = int((box[0] - minx) // cw)
        cx1 = int((box[2] - minx) // cw)
        cy0 = int((box[1] - miny) // ch)
        cy1 = int((box[3] - miny) // ch)
        return (cx0, cx1), (cy0, cy1)

    def co_cell_candidates(self, idx: int) -> Iterable[int]:
        """Yield indices that share at least one grid cell with `idx`."""
        cx0, cx1, cy0, cy1 = self._cell_ranges[idx]
        seen: set[int] = set()
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for other in self._grid.get((cx, cy), ()):
                    if other not in seen:
                        seen.add(other)
                        yield other

    # ------------------------------------------------------------------ #
    # Overlaps
    # ------------------------------------------------------------------ #
    def overlapping_pairs(
        self,
        threshold: float = 0.25,
        mode: Literal["min", "iou", "a", "b"] = "min",
    ) -> List[OverlapTuple]:
        """
        Return (i, j, score) with score >= threshold.

        Parameters
        ----------
        threshold:
            Minimum overlap score.
        mode:
            How to normalize the intersection area.
                * "min": intersection / min(area(a), area(b))
                * "iou": intersection / union
                * "a":   intersection / area(a)
                * "b":   intersection / area(b)
        """
        if not self.boxes:
            return []

        results: List[OverlapTuple] = []
        seen: set[Tuple[int, int]] = set()

        for idxs in self._grid.values():
            if len(idxs) < 2:
                continue

            for pos_i in range(len(idxs)):
                i = idxs[pos_i]
                box_i = self.boxes[i]
                for pos_j in range(pos_i + 1, len(idxs)):
                    j = idxs[pos_j]

                    key = (i, j) if i < j else (j, i)
                    if key in seen:
                        continue
                    seen.add(key)

                    box_j = self.boxes[j]

                    # Fast reject using axis projections
                    if box_i[2] <= box_j[0] or box_j[2] <= box_i[0]:
                        continue
                    if box_i[3] <= box_j[1] or box_j[3] <= box_i[1]:
                        continue

                    inter = intersection_area(box_i, box_j)
                    if inter <= 0.0:
                        continue

                    if mode == "min":
                        denom = min(self._areas[i], self._areas[j])
                    elif mode == "iou":
                        denom = self._areas[i] + self._areas[j] - inter
                    elif mode == "a":
                        denom = self._areas[i]
                    elif mode == "b":
                        denom = self._areas[j]
                    else:
                        raise ValueError(f"Unknown overlap mode: {mode}")

                    score = inter / denom if denom > 0.0 else 0.0
                    if score >= threshold:
                        results.append((key[0], key[1], score))

        return results

    # ------------------------------------------------------------------ #
    # Spacing estimation + clustering
    # ------------------------------------------------------------------ #
    def estimate_typical_spacing(
        self,
        direction: Literal["vertical", "horizontal"] = "vertical",
        align_frac: float = 0.35,
        look_cells_radius: int = 1,
    ) -> float:
        """
        Estimate typical gap between nearest neighbors in the given direction.

        Returns
        -------
        float
            Median of nearest-neighbor gaps (0.0 if insufficient data).
        """
        if not self.boxes:
            return 0.0

        gaps: List[float] = []

        for i, box_i in enumerate(self.boxes):
            best_gap: Optional[float] = None
            cx0, cx1, cy0, cy1 = self._cell_ranges[i]
            cand: set[int] = set()

            for cx in range(cx0 - look_cells_radius, cx1 + look_cells_radius + 1):
                for cy in range(cy0 - look_cells_radius, cy1 + look_cells_radius + 1):
                    cand.update(self._grid.get((cx, cy), ()))

            for j in cand:
                if j == i:
                    continue
                box_j = self.boxes[j]

                if direction == "vertical":
                    overlap = one_d_overlap_len(box_i[0], box_i[2], box_j[0], box_j[2])
                    min_width = min(box_i[2] - box_i[0], box_j[2] - box_j[0])
                    if min_width <= 0 or overlap < align_frac * min_width:
                        continue

                    if box_j[1] >= box_i[3]:
                        gap = box_j[1] - box_i[3]
                    elif box_i[1] >= box_j[3]:
                        gap = box_i[1] - box_j[3]
                    else:
                        continue  # overlapping vertically − gap ~= 0
                else:
                    overlap = one_d_overlap_len(box_i[1], box_i[3], box_j[1], box_j[3])
                    min_height = min(box_i[3] - box_i[1], box_j[3] - box_j[1])
                    if min_height <= 0 or overlap < align_frac * min_height:
                        continue

                    if box_j[0] >= box_i[2]:
                        gap = box_j[0] - box_i[2]
                    elif box_i[0] >= box_j[2]:
                        gap = box_i[0] - box_j[2]
                    else:
                        continue

                if gap <= 0:
                    continue
                if best_gap is None or gap < best_gap:
                    best_gap = gap

            if best_gap is not None:
                gaps.append(best_gap)

        return median(gaps) if gaps else 0.0

    def cluster_text(
        self,
        direction: Literal["vertical", "horizontal"] = "vertical",
        cutoff: Optional[float] = None,
        spacing_multiplier: float = 1.25,
        align_frac: float = 0.35,
        look_cells_radius: int = 1,
    ) -> List[List[int]]:
        """
        Single-link agglomerative clustering for text runs.

        Boxes connect when they:
            * Align in the orthogonal axis by at least `align_frac`
              fraction of the smaller side.
            * Have a gap <= cutoff along the chosen direction.

        Parameters
        ----------
        cutoff:
            Absolute maximum gap. If None, derived from the typical spacing
            multiplied by `spacing_multiplier`. When no spacing data exists,
            the cutoff defaults to `inf` which effectively connects aligned
            neighbors (useful for single-line clusters).
        """
        n = len(self.boxes)
        if n == 0:
            return []

        parent = list(range(n))
        rank = [0] * n

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        if cutoff is None:
            typical = self.estimate_typical_spacing(
                direction=direction,
                align_frac=align_frac,
                look_cells_radius=look_cells_radius,
            )
            cutoff = spacing_multiplier * typical if typical > 0 else float("inf")

        for i, box_i in enumerate(self.boxes):
            cx0, cx1, cy0, cy1 = self._cell_ranges[i]
            cand: set[int] = set()
            for cx in range(cx0 - look_cells_radius, cx1 + look_cells_radius + 1):
                for cy in range(cy0 - look_cells_radius, cy1 + look_cells_radius + 1):
                    cand.update(self._grid.get((cx, cy), ()))

            for j in cand:
                if j == i:
                    continue
                box_j = self.boxes[j]

                if direction == "vertical":
                    overlap = one_d_overlap_len(box_i[0], box_i[2], box_j[0], box_j[2])
                    min_width = min(box_i[2] - box_i[0], box_j[2] - box_j[0])
                    if min_width <= 0 or overlap < align_frac * min_width:
                        continue
                    if box_j[1] >= box_i[3]:
                        gap = box_j[1] - box_i[3]
                    elif box_i[1] >= box_j[3]:
                        gap = box_i[1] - box_j[3]
                    else:
                        gap = 0.0
                else:
                    overlap = one_d_overlap_len(box_i[1], box_i[3], box_j[1], box_j[3])
                    min_height = min(box_i[3] - box_i[1], box_j[3] - box_j[1])
                    if min_height <= 0 or overlap < align_frac * min_height:
                        continue
                    if box_j[0] >= box_i[2]:
                        gap = box_j[0] - box_i[2]
                    elif box_i[0] >= box_j[2]:
                        gap = box_i[0] - box_j[2]
                    else:
                        gap = 0.0

                if gap <= cutoff:
                    union(i, j)

        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx in range(n):
            clusters[find(idx)].append(idx)
        return list(clusters.values())


__all__ = [
    "Box",
    "SpatialGrid",
    "box_area",
    "intersection_area",
    "one_d_overlap_len",
]
