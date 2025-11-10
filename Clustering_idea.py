from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import List, Tuple, Dict, Iterable, Optional, Literal

Box = Tuple[float, float, float, float]  # (x0, y0, x1, y1)

def box_area(b: Box) -> float:
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def intersection_area(a: Box, b: Box) -> float:
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:  # strict: edge-touch → zero area
        return 0.0
    return (x1-x0) * (y1-y0)

def one_d_overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    """Positive length of 1D interval intersection; 0 if disjoint or touching."""
    lo = max(a0, b0); hi = min(a1, b1)
    return max(0.0, hi - lo)

@dataclass
class SpatialGrid:
    boxes: List[Box]
    cell_w: Optional[float] = None
    cell_h: Optional[float] = None

    def __post_init__(self):
        if not self.boxes:
            # minimal scaffolding
            self._grid = {}
            self._minx = self._miny = 0.0
            self._cw = self._ch = 1.0
            return

        widths  = [max(1e-12, b[2]-b[0]) for b in self.boxes]
        heights = [max(1e-12, b[3]-b[1]) for b in self.boxes]
        self._cw = self.cell_w or median(widths)
        self._ch = self.cell_h or median(heights)
        self._minx = min(b[0] for b in self.boxes)
        self._miny = min(b[1] for b in self.boxes)

        self._grid: Dict[Tuple[int,int], List[int]] = defaultdict(list)
        self._cell_ranges: List[Tuple[int,int,int,int]] = []

        for i, b in enumerate(self.boxes):
            xr, yr = self._cells_for_box(b)
            self._cell_ranges.append((*xr, *yr))
            for cx in range(xr[0], xr[1]+1):
                for cy in range(yr[0], yr[1]+1):
                    self._grid[(cx, cy)].append(i)

        # Cache areas
        self._areas = [box_area(b) for b in self.boxes]

    # ---------- core grid helpers ----------
    def _cells_for_box(self, b: Box):
        cw, ch, minx, miny = self._cw, self._ch, self._minx, self._miny
        cx0 = int((b[0] - minx) // cw); cx1 = int((b[2] - minx) // cw)
        cy0 = int((b[1] - miny) // ch); cy1 = int((b[3] - miny) // ch)
        return (cx0, cx1), (cy0, cy1)

    def co_cell_candidates(self, idx: int) -> Iterable[int]:
        """All indices that share at least one grid cell with box idx (including itself)."""
        cx0, cx1, cy0, cy1 = self._cell_ranges[idx]
        seen = set()
        for cx in range(cx0, cx1+1):
            for cy in range(cy0, cy1+1):
                for j in self._grid.get((cx, cy), ()):
                    if j not in seen:
                        seen.add(j)
                        yield j

    # ---------- overlaps ----------
    def overlapping_pairs(self,
                          threshold: float = 0.25,
                          mode: Literal["min","iou","a","b"] = "min"):
        """
        Return (i, j, score) for all i<j with overlap score >= threshold.
        mode:
            'min' : inter / min(area(a), area(b))   # great for "≥25%" intent
            'iou' : inter / (Aa + Ab - inter)
            'a'   : inter / area(a)
            'b'   : inter / area(b)
        """
        results = []
        seen = set()
        for (cell, idxs) in self._grid.items():
            m = len(idxs)
            if m < 2: continue
            # Intra-cell pairs only; all-cells-touched mapping guarantees completeness.
            for p in range(m):
                i = idxs[p]
                bi = self.boxes[i]
                for q in range(p+1, m):
                    j = idxs[q]
                    if i < j:
                        key = (i, j)
                    else:
                        key = (j, i)
                    if key in seen:  # pair may co-occur in multiple cells
                        continue
                    seen.add(key)

                    bj = self.boxes[j]
                    # quick 1D reject (strict)
                    if bi[2] <= bj[0] or bj[2] <= bi[0]: continue
                    if bi[3] <= bj[1] or bj[3] <= bi[1]: continue

                    inter = intersection_area(bi, bj)
                    if inter <= 0.0: continue

                    if mode == "min":
                        denom = min(self._areas[i], self._areas[j])
                    elif mode == "iou":
                        denom = self._areas[i] + self._areas[j] - inter
                    elif mode == "a":
                        denom = self._areas[i]
                    elif mode == "b":
                        denom = self._areas[j]
                    else:
                        raise ValueError("unknown mode")

                    score = inter / denom if denom > 0.0 else 0.0
                    if score >= threshold:
                        results.append((key[0], key[1], score))
        return results

    # ---------- typical spacing estimation ----------
    def estimate_typical_spacing(self,
                                 direction: Literal["vertical","horizontal"] = "vertical",
                                 align_frac: float = 0.35,
                                 look_cells_radius: int = 1) -> float:
        """
        Robust estimate of typical inter-line (vertical) or inter-word (horizontal) spacing.
        - direction='vertical': measure vertical gaps between boxes whose horizontal
          projections overlap by at least align_frac fraction of the smaller width.
        - direction='horizontal': analogous with vertical projection alignment.

        Returns median of nearest-neighbor gaps (ignoring 0/negative).
        """
        gaps = []

        for i, bi in enumerate(self.boxes):
            best_gap = None

            # Candidate search: expand cell neighborhood slightly (radius)
            (cx0,cx1,cy0,cy1) = self._cell_ranges[i]
            cand = set()
            for cx in range(cx0 - look_cells_radius, cx1 + look_cells_radius + 1):
                for cy in range(cy0 - look_cells_radius, cy1 + look_cells_radius + 1):
                    cand.update(self._grid.get((cx,cy), []))

            for j in cand:
                if j == i: continue
                bj = self.boxes[j]

                if direction == "vertical":
                    # Horizontal alignment requirement
                    w_overlap = one_d_overlap_len(bi[0], bi[2], bj[0], bj[2])
                    min_w = min(bi[2]-bi[0], bj[2]-bj[0])
                    if min_w <= 0: continue
                    if w_overlap < align_frac * min_w:  # not sufficiently aligned in x
                        continue
                    # Vertical gap: positive distance between disjoint y-intervals
                    if bj[1] >= bi[3]:
                        gap = bj[1] - bi[3]
                    elif bi[1] >= bj[3]:
                        gap = bi[1] - bj[3]
                    else:
                        continue  # overlapping vertically; not a "gap"
                else:  # horizontal spacing
                    h_overlap = one_d_overlap_len(bi[1], bi[3], bj[1], bj[3])
                    min_h = min(bi[3]-bi[1], bj[3]-bj[1])
                    if min_h <= 0: continue
                    if h_overlap < align_frac * min_h:
                        continue
                    if bj[0] >= bi[2]:
                        gap = bj[0] - bi[2]
                    elif bi[0] >= bj[2]:
                        gap = bi[0] - bj[2]
                    else:
                        continue

                if gap <= 0: continue
                if (best_gap is None) or (gap < best_gap):
                    best_gap = gap

            if best_gap is not None:
                gaps.append(best_gap)

        return median(gaps) if gaps else 0.0

    # ---------- clustering (single-link via union-find) ----------
    def cluster_text(self,
                     direction: Literal["vertical","horizontal"] = "vertical",
                     cutoff: Optional[float] = None,
                     spacing_multiplier: float = 1.25,
                     align_frac: float = 0.35,
                     look_cells_radius: int = 1):
        """
        Single-link agglomerative clustering along a chosen direction.
        - If cutoff is None, uses cutoff = spacing_multiplier * typical_spacing(direction)
        - Two boxes connect if:
            * They are "neighbors" along direction with gap <= cutoff, and
            * Their projection overlap in the orthogonal axis ≥ align_frac⋅(smaller side).
        - Returns: List[List[int]] of index clusters (singletons preserved).
        """
        n = len(self.boxes)
        parent = list(range(n))
        rank = [0]*n

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a,b):
            ra, rb = find(a), find(b)
            if ra == rb: return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        # Decide cutoff
        if cutoff is None:
            typical = self.estimate_typical_spacing(direction=direction,
                                                    align_frac=align_frac,
                                                    look_cells_radius=look_cells_radius)
            cutoff = spacing_multiplier * typical if typical > 0 else float("inf")

        # Build edges using grid-limited candidates
        for i, bi in enumerate(self.boxes):
            (cx0,cx1,cy0,cy1) = self._cell_ranges[i]
            cand = set()
            for cx in range(cx0 - look_cells_radius, cx1 + look_cells_radius + 1):
                for cy in range(cy0 - look_cells_radius, cy1 + look_cells_radius + 1):
                    cand.update(self._grid.get((cx,cy), []))

            for j in cand:
                if j == i: continue
                bj = self.boxes[j]

                if direction == "vertical":
                    # horizontal alignment
                    w_overlap = one_d_overlap_len(bi[0], bi[2], bj[0], bj[2])
                    min_w = min(bi[2]-bi[0], bj[2]-bj[0])
                    if min_w <= 0 or w_overlap < align_frac * min_w:
                        continue
                    # vertical gap
                    gap = None
                    if bj[1] >= bi[3]:
                        gap = bj[1] - bi[3]
                    elif bi[1] >= bj[3]:
                        gap = bi[1] - bj[3]
                    else:
                        # overlapping vertically → essentially gap 0 → connect
                        gap = 0.0
                else:  # horizontal
                    h_overlap = one_d_overlap_len(bi[1], bi[3], bj[1], bj[3])
                    min_h = min(bi[3]-bi[1], bj[3]-bj[1])
                    if min_h <= 0 or h_overlap < align_frac * min_h:
                        continue
                    if bj[0] >= bi[2]:
                        gap = bj[0] - bi[2]
                    elif bi[0] >= bj[2]:
                        gap = bi[0] - bj[2]
                    else:
                        gap = 0.0

                if gap is not None and gap <= cutoff:
                    union(i, j)

        # Collect clusters (singletons preserved)
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        return list(groups.values())
