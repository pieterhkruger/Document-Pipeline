"""
Cell-Based Data Classes for Font Color Analysis
===============================================

Hierarchical data structure for cell-level (line-level) font color analysis:
- Docling cells (from parsed_pages)
- PyMuPDF spans (line-level)
- Pillow foreground/background color clustering
- Edge/gradient statistics for forgery detection
- Cell-to-span matching results

This is the NEW approach using cells instead of words.

Hierarchy:
    CellColorElement
        ├── docling_cell (line from parsed_pages)
        ├── pillow_colors (K-means foreground/background clustering)
        ├── edge_gradient_stats (Sobel gradients + ELA for forgery detection)
        └── pymupdf_match (matched PyMuPDF span)
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


# ============================================================================
# DOCLING CELL DATA CLASS
# ============================================================================

@dataclass
class DoclingCell:
    """
    Docling cell (line-level element from parsed_pages).

    Cells are the line-level elements extracted from Docling with generate_parsed_pages=True.
    Each cell represents a complete line of text with exact bbox and font metadata.
    """
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in BOTTOMLEFT coords
    font_name: Optional[str] = None
    font_key: Optional[str] = None
    rgba: Optional[Dict[str, int]] = None  # {"r": int, "g": int, "b": int, "a": int}
    confidence: Optional[float] = None
    from_ocr: Optional[bool] = None
    index: Optional[int] = None
    page_num: Optional[int] = None


# ============================================================================
# PYMUPDF SPAN DATA CLASS
# ============================================================================

@dataclass
class PyMuPDFSpan:
    """
    PyMuPDF span (line/phrase-level element).

    Spans are the line-level elements extracted from PyMuPDF get_text("dict").
    Each span represents a phrase/line with font and color metadata.
    """
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in BOTTOMLEFT coords
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    color_rgb: Optional[Tuple[int, int, int]] = None  # (r, g, b) 0-255
    flags: Optional[int] = None  # Font flags (bold, italic, etc.)


# ============================================================================
# PILLOW FOREGROUND/BACKGROUND DATA CLASS
# ============================================================================

@dataclass
class ForegroundBackgroundColors:
    """
    Foreground and background colors from K-means clustering.

    Uses 2-cluster K-means on all pixels in bbox to separate:
    - Foreground color (text)
    - Background color (page/highlighting)

    Cluster sizes determine text-to-background ratio.
    """
    foreground_color: Tuple[int, int, int]  # RGB (0-255) - text color
    background_color: Tuple[int, int, int]  # RGB (0-255) - background color
    foreground_pixels: int  # Number of pixels in foreground cluster
    background_pixels: int  # Number of pixels in background cluster
    text_to_background_ratio: float  # Ratio of foreground to background pixels
    total_pixels: int  # Total pixels analyzed
    bbox: Tuple[float, float, float, float]  # Bbox analyzed
    sample_method: str  # "kmeans_gpu" or "kmeans_cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "foreground_color": list(self.foreground_color),
            "background_color": list(self.background_color),
            "foreground_pixels": self.foreground_pixels,
            "background_pixels": self.background_pixels,
            "foreground_percentage": (self.foreground_pixels / self.total_pixels * 100) if self.total_pixels > 0 else 0.0,
            "background_percentage": (self.background_pixels / self.total_pixels * 100) if self.total_pixels > 0 else 0.0,
            "text_to_background_ratio": self.text_to_background_ratio,
            "total_pixels": self.total_pixels,
            "sample_method": self.sample_method
        }


# ============================================================================
# EDGE/GRADIENT STATISTICS DATA CLASSES
# ============================================================================

@dataclass
class EdgeGradientStats:
    """
    Edge and gradient statistics for a single bbox variant (full or foreground-masked).

    Computed using Sobel gradients to detect rendering inconsistencies that may
    indicate forgery (different anti-aliasing, compression, or paste artifacts).

    Statistics include:
    - Summary stats (mean, std, min, Q1, median, Q3, p95, p99, max, mode)
    - High-freq vs low-freq energy ratio
    - Laplacian (second-order edges for splicing detection)
    - Local variance (texture smoothness analysis)
    - Bilateral filter response (edge preservation analysis)

    Optimized for ML model feature input with OpenCV-based forgery features.
    """
    # Summary statistics (for ML features)
    mean_gradient: float  # Mean gradient magnitude
    std_gradient: float  # Standard deviation
    min_gradient: float  # Minimum gradient magnitude
    q1_gradient: float  # 25th percentile (Q1)
    median_gradient: float  # 50th percentile (Q2, robust to outliers)
    q3_gradient: float  # 75th percentile (Q3)
    p95_gradient: float  # 95th percentile
    p99_gradient: float  # 99th percentile
    max_gradient: float  # Maximum gradient magnitude
    mode_gradient: float  # Most frequent gradient value (binned)

    # High-freq vs low-freq energy
    high_freq_energy: float  # Sum of gradients above threshold
    low_freq_energy: float  # Sum of gradients below threshold
    high_to_low_ratio: float  # Ratio (high / low) - indicates sharpness
    threshold: float  # Threshold used to separate high/low freq

    # Laplacian statistics (second-order edges, splicing detection)
    laplacian_mean: float  # Mean of absolute Laplacian values
    laplacian_std: float  # Std deviation of Laplacian
    laplacian_max: float  # Maximum absolute Laplacian
    laplacian_p95: float  # 95th percentile of absolute Laplacian
    laplacian_zero_crossings: int  # Number of sign changes (edge indicators)

    # Local variance statistics (texture smoothness, artificial smoothing detection)
    local_variance_mean: float  # Mean variance in sliding windows
    local_variance_std: float  # Std deviation of variance (texture consistency)
    local_variance_max: float  # Maximum local variance
    low_variance_ratio: float  # Ratio of pixels with suspiciously low variance

    # Bilateral filter statistics (edge preservation, artificial sharpening detection)
    bilateral_difference_mean: float  # Mean |original - bilateral_filtered|
    bilateral_difference_std: float  # Std of difference
    bilateral_edge_preservation: float  # How well edges are preserved (0-1)

    # Metadata
    total_pixels: int  # Number of pixels analyzed
    computation_method: str  # "sobel_cv2" or "sobel_pillow"
    computation_time_ms: float  # Time taken in milliseconds
    mask_applied: bool  # True if foreground-only (using K-means mask)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary_stats": {
                "mean": round(self.mean_gradient, 4),
                "std": round(self.std_gradient, 4),
                "min": round(self.min_gradient, 4),
                "q1": round(self.q1_gradient, 4),
                "median": round(self.median_gradient, 4),
                "q3": round(self.q3_gradient, 4),
                "p95": round(self.p95_gradient, 4),
                "p99": round(self.p99_gradient, 4),
                "max": round(self.max_gradient, 4),
                "mode": round(self.mode_gradient, 4)
            },
            "frequency_energy": {
                "high_freq": round(self.high_freq_energy, 2),
                "low_freq": round(self.low_freq_energy, 2),
                "high_to_low_ratio": round(self.high_to_low_ratio, 4) if self.high_to_low_ratio != float('inf') else "Infinity",
                "threshold": round(self.threshold, 2)
            },
            "laplacian": {
                "mean": round(self.laplacian_mean, 4),
                "std": round(self.laplacian_std, 4),
                "max": round(self.laplacian_max, 4),
                "p95": round(self.laplacian_p95, 4),
                "zero_crossings": self.laplacian_zero_crossings
            },
            "local_variance": {
                "mean": round(self.local_variance_mean, 4),
                "std": round(self.local_variance_std, 4),
                "max": round(self.local_variance_max, 4),
                "low_variance_ratio": round(self.low_variance_ratio, 4)
            },
            "bilateral": {
                "difference_mean": round(self.bilateral_difference_mean, 4),
                "difference_std": round(self.bilateral_difference_std, 4),
                "edge_preservation": round(self.bilateral_edge_preservation, 4)
            },
            "metadata": {
                "total_pixels": self.total_pixels,
                "computation_method": self.computation_method,
                "computation_time_ms": round(self.computation_time_ms, 2),
                "mask_applied": self.mask_applied
            }
        }


@dataclass
class ELAStats:
    """
    Error Level Analysis (ELA) statistics for JPEG artifact detection.

    ELA detects compression artifacts by re-saving as JPEG and comparing.
    Forged regions often show different error levels than genuine regions.
    """
    ela_mean: float  # Mean error across RGB channels
    ela_p95: float  # 95th percentile of luminance error
    ela_max: float  # Maximum error
    quality: int  # JPEG quality used for re-compression (95)
    computation_time_ms: float  # Time taken in milliseconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ela_mean": round(self.ela_mean, 4),
            "ela_p95": round(self.ela_p95, 4),
            "ela_max": round(self.ela_max, 4),
            "quality": self.quality,
            "computation_time_ms": round(self.computation_time_ms, 2)
        }


@dataclass
class EdgeGradientAnalysis:
    """
    Complete edge/gradient analysis for a cell bbox.

    Includes:
    - Full bbox stats (text + background)
    - Foreground-only stats (using K-means mask)
    - ELA (Error Level Analysis)
    - Timing information

    This allows comparison between genuine and potentially forged text.
    """
    # Full bbox (text + background)
    full_bbox_stats: EdgeGradientStats

    # Foreground-only (using K-means mask from color clustering)
    foreground_only_stats: EdgeGradientStats

    # ELA (Error Level Analysis for JPEG artifacts)
    ela_stats: ELAStats

    # Total computation time
    total_computation_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "full_bbox": self.full_bbox_stats.to_dict(),
            "foreground_only": self.foreground_only_stats.to_dict(),
            "ela": self.ela_stats.to_dict(),
            "total_computation_time_ms": round(self.total_computation_time_ms, 2)
        }


# ============================================================================
# CELL-TO-SPAN MATCH DATA CLASS
# ============================================================================

@dataclass
class CellToSpanMatch:
    """
    Match between a Docling cell and a PyMuPDF span.

    Contains matching metadata (text similarity, bbox IoU, match method).
    """
    docling_cell: DoclingCell
    pymupdf_span: PyMuPDFSpan
    text_similarity: float  # Levenshtein ratio (0.0 - 1.0)
    bbox_iou: float  # Intersection over Union (0.0 - 1.0)
    match_method: str  # "text_based", "spatial_based", "combined"


# ============================================================================
# COMPOSITE CELL COLOR ELEMENT
# ============================================================================

@dataclass
class CellColorElement:
    """
    Complete cell-level element with all analysis data.

    This is the TOP-LEVEL class for cell-based font color analysis.

    Combines:
    - Docling cell (line from parsed_pages)
    - Pillow foreground/background color clustering
    - Edge/gradient statistics (Sobel + ELA for forgery detection)
    - PyMuPDF span match (with font and color metadata)
    - Match metadata (text similarity, bbox IoU, method)

    Hierarchy:
        CellColorElement
            ├── docling_cell (line from parsed_pages)
            ├── pillow_colors (K-means foreground/background)
            ├── edge_gradient_analysis (Sobel gradients + ELA)
            │   ├── full_bbox_stats
            │   ├── foreground_only_stats
            │   └── ela_stats
            └── pymupdf_match (matched PyMuPDF span)
                ├── text_similarity
                ├── bbox_iou
                └── match_method
    """
    # Cell data (from Docling parsed_pages)
    docling_cell: DoclingCell

    # Pillow foreground/background color analysis
    pillow_colors: ForegroundBackgroundColors

    # Edge/gradient analysis for forgery detection (optional)
    edge_gradient_analysis: Optional[EdgeGradientAnalysis] = None

    # PyMuPDF span match (can be None if unmatched)
    pymupdf_match: Optional[CellToSpanMatch] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Includes ALL data:
        - Cell data (text, bbox, font, rgba)
        - Pillow foreground/background colors and ratio
        - Edge/gradient statistics (Sobel + ELA)
        - PyMuPDF match (text, bbox, font, color, similarity, IoU)
        """
        result = {
            "cell_data": {
                "text": self.docling_cell.text,
                "bbox": list(self.docling_cell.bbox),
                "font_name": self.docling_cell.font_name,
                "font_key": self.docling_cell.font_key,
                "rgba": self.docling_cell.rgba,
                "confidence": self.docling_cell.confidence,
                "from_ocr": self.docling_cell.from_ocr,
                "index": self.docling_cell.index
            },
            "pillow_analysis": self.pillow_colors.to_dict()
        }

        # Add edge/gradient analysis if present
        if self.edge_gradient_analysis:
            result["edge_gradient_analysis"] = self.edge_gradient_analysis.to_dict()

        # Add PyMuPDF match if present
        if self.pymupdf_match:
            result["pymupdf_match"] = {
                "text": self.pymupdf_match.pymupdf_span.text,
                "bbox": list(self.pymupdf_match.pymupdf_span.bbox),
                "font_name": self.pymupdf_match.pymupdf_span.font_name,
                "font_size": self.pymupdf_match.pymupdf_span.font_size,
                "color_rgb": list(self.pymupdf_match.pymupdf_span.color_rgb) if self.pymupdf_match.pymupdf_span.color_rgb else None,
                "flags": self.pymupdf_match.pymupdf_span.flags
            }
            result["match_metadata"] = {
                "text_similarity": self.pymupdf_match.text_similarity,
                "bbox_iou": self.pymupdf_match.bbox_iou,
                "match_method": self.pymupdf_match.match_method
            }
        else:
            result["pymupdf_match"] = None
            result["match_metadata"] = {
                "match_method": "unmatched"
            }

        return result


# ============================================================================
# DOCUMENT-LEVEL CONTAINER
# ============================================================================

@dataclass
class CellColorAnalysisResult:
    """
    Complete cell-level color analysis for a document page.

    Contains:
    - Document metadata (filename, page number)
    - All cell color elements
    - Match statistics
    """
    document_name: str
    page_num: int
    total_cells: int
    total_spans: int
    matched_cells: int
    match_rate: float  # Percentage (0.0 - 100.0)
    elements: List[CellColorElement]
    match_statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document": self.document_name,
            "page": self.page_num,
            "total_cells": self.total_cells,
            "total_spans": self.total_spans,
            "matched_cells": self.matched_cells,
            "match_rate": f"{self.match_rate:.1f}%",
            "match_statistics": self.match_statistics,
            "elements": [elem.to_dict() for elem in self.elements]
        }
