"""
Edge/Gradient Analysis for Forgery Detection
============================================

Implements Sobel gradient analysis and ELA (Error Level Analysis) to detect
potential forgery in documents. Forged text often has:
- Different edge sharpness (anti-aliasing artifacts)
- Different gradient distributions (rendering inconsistencies)
- Different JPEG compression artifacts (paste from different sources)

This module provides:
1. Sobel gradient computation (cv2 and Pillow)
2. Statistical feature extraction (mean, std, min, Q1, median, Q3, p95, p99, max, mode)
3. High/low frequency energy analysis
4. ELA (Error Level Analysis) for JPEG artifacts
5. Both full bbox and foreground-masked analysis

Features optimized for ML features input (histograms removed to reduce noise).

Usage:
    from edge_gradient_analysis import analyze_edge_gradient

    result = analyze_edge_gradient(
        pdf_path="document.pdf",
        page_num=0,
        bbox=(x0, y0, x1, y1),
        foreground_mask=mask_array,  # Optional K-means mask from color clustering
        dpi=600,
        use_cv2=True
    )
"""

import time
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageChops, ImageStat, ImageFilter
import fitz  # PyMuPDF

from cell_data_classes import (
    EdgeGradientStats,
    ELAStats,
    EdgeGradientAnalysis
)

# Try to import cv2 (faster, more features)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] cv2 not available - will use Pillow for gradient computation (slower)")


# ============================================================================
# RENDERING UTILITIES
# ============================================================================

def extract_image_from_bbox(
    pdf_path: str,
    page_num: int,
    bbox: Tuple[float, float, float, float],
    dpi: int = 600
) -> Image.Image:
    """
    Extract PIL Image from PDF bbox at specified DPI.

    Uses same rendering as color clustering (600 DPI, no alpha) for consistency.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        bbox: (x0, y0, x1, y1) in BOTTOMLEFT coords
        dpi: Rendering DPI (default 600 for accuracy)

    Returns:
        PIL RGB Image of the bbox region
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # Calculate scale factor
    scale = dpi / 72
    mat = fitz.Matrix(scale, scale)

    # Render page to pixmap
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Crop to bbox (convert PDF coords to pixel coords)
    x0, y0, x1, y1 = bbox
    page_height = page.rect.height

    # Convert BOTTOMLEFT to pixel coords
    crop_x0 = int(x0 * scale)
    crop_y0 = int((page_height - y1) * scale)  # y1 is top edge in BOTTOMLEFT
    crop_x1 = int(x1 * scale)
    crop_y1 = int((page_height - y0) * scale)  # y0 is bottom edge in BOTTOMLEFT

    # Crop
    bbox_img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))

    doc.close()
    return bbox_img


# ============================================================================
# SOBEL GRADIENT COMPUTATION
# ============================================================================

def compute_sobel_gradients_cv2(gray_array: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gradient magnitudes using cv2 (fast, accurate).

    Args:
        gray_array: Grayscale image as numpy array (H, W) with values 0-255

    Returns:
        Gradient magnitude array (H, W) with float values
    """
    # Compute Sobel gradients in x and y directions
    gx = cv2.Sobel(gray_array, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_array, cv2.CV_32F, 0, 1, ksize=3)

    # Compute magnitude: M = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(gx * gx + gy * gy)

    return magnitude


def compute_sobel_gradients_pillow(img: Image.Image) -> np.ndarray:
    """
    Compute Sobel gradient magnitudes using Pillow (slower, pure Python).

    Args:
        img: PIL Image (RGB or L mode)

    Returns:
        Gradient magnitude array (H, W) with float values
    """
    # Convert to grayscale if needed
    if img.mode != 'L':
        gray_img = img.convert('L')
    else:
        gray_img = img

    # Sobel kernels (3x3)
    # Gx: horizontal edge detection
    sobel_x = ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1)
    # Gy: vertical edge detection
    sobel_y = ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1)

    # Apply Sobel filters
    gx_img = gray_img.filter(sobel_x)
    gy_img = gray_img.filter(sobel_y)

    # Convert to numpy arrays
    gx = np.array(gx_img, dtype=np.float32)
    gy = np.array(gy_img, dtype=np.float32)

    # Compute magnitude
    magnitude = np.sqrt(gx * gx + gy * gy)

    return magnitude


# ============================================================================
# STATISTICAL FEATURE EXTRACTION
# ============================================================================

def count_zero_crossings(laplacian: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
    """
    Count zero crossings in Laplacian (sign changes indicate edges).

    Args:
        laplacian: Laplacian array (H, W)
        mask: Optional binary mask (1 = include, 0 = exclude)

    Returns:
        Number of zero crossings
    """
    # Apply mask if provided
    if mask is not None:
        laplacian_masked = laplacian.copy()
        laplacian_masked[mask == 0] = 0
    else:
        laplacian_masked = laplacian

    # Count sign changes in horizontal direction
    horizontal_crossings = np.sum(
        (laplacian_masked[:, :-1] * laplacian_masked[:, 1:]) < 0
    )

    # Count sign changes in vertical direction
    vertical_crossings = np.sum(
        (laplacian_masked[:-1, :] * laplacian_masked[1:, :]) < 0
    )

    return int(horizontal_crossings + vertical_crossings)


def compute_gradient_statistics(
    magnitude: np.ndarray,
    gray_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "sobel_cv2"
) -> Tuple[EdgeGradientStats, float]:
    """
    Compute comprehensive gradient statistics including OpenCV-based forgery features.

    Args:
        magnitude: Gradient magnitude array (H, W)
        gray_image: Original grayscale image (H, W) for Laplacian/variance/bilateral
        mask: Optional binary mask (1 = include, 0 = exclude)
        method: "sobel_cv2" or "sobel_pillow"

    Returns:
        (EdgeGradientStats, computation_time_ms)
    """
    start_time = time.time()

    # Apply mask if provided
    if mask is not None:
        # Mask should be binary (0 or 1)
        values = magnitude[mask > 0].ravel()
        mask_applied = True
    else:
        values = magnitude.ravel()
        mask_applied = False

    # Handle empty case
    if values.size == 0:
        # Return zeros
        return EdgeGradientStats(
            mean_gradient=0.0,
            std_gradient=0.0,
            min_gradient=0.0,
            q1_gradient=0.0,
            median_gradient=0.0,
            q3_gradient=0.0,
            p95_gradient=0.0,
            p99_gradient=0.0,
            max_gradient=0.0,
            mode_gradient=0.0,
            high_freq_energy=0.0,
            low_freq_energy=0.0,
            high_to_low_ratio=0.0,
            threshold=0.0,
            total_pixels=0,
            computation_method=method,
            computation_time_ms=0.0,
            mask_applied=mask_applied,
            # OpenCV features (all zeros for empty case)
            laplacian_mean=0.0,
            laplacian_std=0.0,
            laplacian_max=0.0,
            laplacian_p95=0.0,
            laplacian_zero_crossings=0,
            local_variance_mean=0.0,
            local_variance_std=0.0,
            local_variance_max=0.0,
            low_variance_ratio=0.0,
            bilateral_difference_mean=0.0,
            bilateral_difference_std=0.0,
            bilateral_edge_preservation=0.0
        ), 0.0

    # Summary statistics (for ML features)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    min_val = float(np.min(values))
    q1_val = float(np.percentile(values, 25))
    median_val = float(np.median(values))  # Q2 (50th percentile)
    q3_val = float(np.percentile(values, 75))
    p95_val = float(np.percentile(values, 95))
    p99_val = float(np.percentile(values, 99))
    max_val = float(np.max(values))

    # Mode computation (binned to avoid continuous value issues)
    # Use 100 bins for mode estimation
    hist_mode, edges_mode = np.histogram(values, bins=100, range=(min_val, max_val))
    mode_bin_idx = np.argmax(hist_mode)
    # Mode is the center of the most frequent bin
    mode_val = float((edges_mode[mode_bin_idx] + edges_mode[mode_bin_idx + 1]) / 2.0)

    # High-freq vs low-freq energy
    # Threshold: use median as separator (could also use mean or fixed value)
    threshold = median_val
    high_freq_mask = values > threshold
    low_freq_mask = values <= threshold

    high_freq_energy = float(np.sum(values[high_freq_mask]))
    low_freq_energy = float(np.sum(values[low_freq_mask]))

    # Ratio (avoid division by zero)
    if low_freq_energy > 0:
        high_to_low_ratio = high_freq_energy / low_freq_energy
    else:
        high_to_low_ratio = float('inf') if high_freq_energy > 0 else 0.0

    # ========================================================================
    # NEW: OPENCV-BASED FORGERY DETECTION FEATURES
    # ========================================================================

    # PHASE 1: Laplacian (second-order edges - detects splicing/paste artifacts)
    if CV2_AVAILABLE and method == "sobel_cv2":
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        if mask is not None:
            laplacian_values = np.abs(laplacian[mask > 0].ravel())
        else:
            laplacian_values = np.abs(laplacian.ravel())

        laplacian_mean = float(np.mean(laplacian_values))
        laplacian_std = float(np.std(laplacian_values))
        laplacian_max = float(np.max(laplacian_values))
        laplacian_p95 = float(np.percentile(laplacian_values, 95))

        # Zero crossings (sign changes indicate edges)
        zero_crossings = count_zero_crossings(laplacian, mask)

        # PHASE 2: Local Variance (texture smoothness - detects artificial smoothing)
        # Use sqrBoxFilter approach: Var(X) = E[X^2] - E[X]^2
        gray_float = gray_image.astype(np.float32)
        mean_sq = cv2.boxFilter(gray_float**2, -1, (5, 5))
        mean = cv2.boxFilter(gray_float, -1, (5, 5))
        variance = mean_sq - mean**2

        if mask is not None:
            variance_values = variance[mask > 0].ravel()
        else:
            variance_values = variance.ravel()

        local_variance_mean = float(np.mean(variance_values))
        local_variance_std = float(np.std(variance_values))
        local_variance_max = float(np.max(variance_values))

        # Low variance ratio (suspiciously smooth areas - bottom 10%)
        if len(variance_values) > 0:
            threshold_low = np.percentile(variance_values, 10)
            low_variance_ratio = float(np.sum(variance_values < threshold_low) / len(variance_values))
        else:
            low_variance_ratio = 0.0

        # PHASE 3: Bilateral Filter (edge preservation - detects artificial sharpening)
        bilateral = cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)
        bilateral_diff = np.abs(gray_image.astype(float) - bilateral.astype(float))

        if mask is not None:
            bilateral_diff_values = bilateral_diff[mask > 0].ravel()
        else:
            bilateral_diff_values = bilateral_diff.ravel()

        bilateral_difference_mean = float(np.mean(bilateral_diff_values))
        bilateral_difference_std = float(np.std(bilateral_diff_values))

        # Edge preservation (how much edges change after bilateral filtering)
        # Higher values = edges preserved well (natural), lower = overly smooth (suspicious)
        edge_mask = magnitude > np.percentile(magnitude.ravel(), 75)
        if np.sum(edge_mask) > 0:
            bilateral_edge_preservation = float(np.mean(bilateral_diff[edge_mask]))
        else:
            bilateral_edge_preservation = 0.0
    else:
        # Pillow mode - set OpenCV features to 0 (not available without cv2)
        laplacian_mean = 0.0
        laplacian_std = 0.0
        laplacian_max = 0.0
        laplacian_p95 = 0.0
        zero_crossings = 0
        local_variance_mean = 0.0
        local_variance_std = 0.0
        local_variance_max = 0.0
        low_variance_ratio = 0.0
        bilateral_difference_mean = 0.0
        bilateral_difference_std = 0.0
        bilateral_edge_preservation = 0.0

    # Computation time
    computation_time_ms = (time.time() - start_time) * 1000

    # Create result
    stats = EdgeGradientStats(
        mean_gradient=mean_val,
        std_gradient=std_val,
        min_gradient=min_val,
        q1_gradient=q1_val,
        median_gradient=median_val,
        q3_gradient=q3_val,
        p95_gradient=p95_val,
        p99_gradient=p99_val,
        max_gradient=max_val,
        mode_gradient=mode_val,
        high_freq_energy=high_freq_energy,
        low_freq_energy=low_freq_energy,
        high_to_low_ratio=high_to_low_ratio,
        threshold=threshold,
        total_pixels=int(values.size),
        computation_method=method,
        computation_time_ms=computation_time_ms,
        mask_applied=mask_applied,
        # OpenCV features
        laplacian_mean=laplacian_mean,
        laplacian_std=laplacian_std,
        laplacian_max=laplacian_max,
        laplacian_p95=laplacian_p95,
        laplacian_zero_crossings=zero_crossings,
        local_variance_mean=local_variance_mean,
        local_variance_std=local_variance_std,
        local_variance_max=local_variance_max,
        low_variance_ratio=low_variance_ratio,
        bilateral_difference_mean=bilateral_difference_mean,
        bilateral_difference_std=bilateral_difference_std,
        bilateral_edge_preservation=bilateral_edge_preservation
    )

    return stats, computation_time_ms


# ============================================================================
# ELA (ERROR LEVEL ANALYSIS)
# ============================================================================

def compute_ela_statistics(
    img: Image.Image,
    quality: int = 95
) -> Tuple[ELAStats, float]:
    """
    Compute ELA (Error Level Analysis) for JPEG artifact detection.

    ELA works by re-saving the image as JPEG at specified quality and comparing
    the difference. Forged regions often show different error levels than
    genuine regions due to double JPEG compression or different source quality.

    Args:
        img: PIL Image (RGB mode)
        quality: JPEG quality for re-compression (default 95)

    Returns:
        (ELAStats, computation_time_ms)
    """
    import io

    start_time = time.time()

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Re-save as JPEG
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    # Compute difference
    diff = ImageChops.difference(img, resaved)

    # Get statistics
    stat = ImageStat.Stat(diff)

    # Mean error across RGB channels
    ela_mean = float(sum(stat.mean) / 3.0)

    # 95th percentile of luminance error
    diff_gray = diff.convert("L")
    diff_array = np.array(diff_gray).ravel()
    ela_p95 = float(np.percentile(diff_array, 95))

    # Maximum error
    ela_max = float(np.max(diff_array))

    # Computation time
    computation_time_ms = (time.time() - start_time) * 1000

    # Create result
    ela_stats = ELAStats(
        ela_mean=ela_mean,
        ela_p95=ela_p95,
        ela_max=ela_max,
        quality=quality,
        computation_time_ms=computation_time_ms
    )

    return ela_stats, computation_time_ms


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_edge_gradient(
    pdf_path: str,
    page_num: int,
    bbox: Tuple[float, float, float, float],
    foreground_mask: Optional[np.ndarray] = None,
    dpi: int = 600,
    use_cv2: Optional[bool] = None,
    ela_quality: int = 95
) -> EdgeGradientAnalysis:
    """
    Complete edge/gradient analysis for a cell bbox.

    Performs:
    1. Sobel gradient computation (full bbox)
    2. Sobel gradient computation (foreground-only using mask)
    3. Laplacian (second-order edges for splicing detection)
    4. Local variance (texture smoothness analysis)
    5. Bilateral filter (edge preservation analysis)
    6. ELA (Error Level Analysis)

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        bbox: (x0, y0, x1, y1) in BOTTOMLEFT coords
        foreground_mask: Optional binary mask for foreground pixels (from K-means)
        dpi: Rendering DPI (default 600)
        use_cv2: Use cv2 if True, Pillow if False, auto-detect if None
        ela_quality: JPEG quality for ELA (default 95)

    Returns:
        EdgeGradientAnalysis with all statistics
    """
    total_start_time = time.time()

    # Auto-detect cv2
    if use_cv2 is None:
        use_cv2 = CV2_AVAILABLE

    # Extract image from PDF
    img = extract_image_from_bbox(pdf_path, page_num, bbox, dpi)

    # Convert to grayscale for gradient computation
    gray_img = img.convert('L')
    gray_array = np.array(gray_img, dtype=np.uint8)

    # ========================================================================
    # 1. Compute Sobel gradients
    # ========================================================================
    if use_cv2 and CV2_AVAILABLE:
        magnitude = compute_sobel_gradients_cv2(gray_array)
        method = "sobel_cv2"
    else:
        magnitude = compute_sobel_gradients_pillow(img)
        method = "sobel_pillow"

    # ========================================================================
    # 2. Full bbox statistics
    # ========================================================================
    full_bbox_stats, _ = compute_gradient_statistics(
        magnitude=magnitude,
        gray_image=gray_array,
        mask=None,
        method=method
    )

    # ========================================================================
    # 3. Foreground-only statistics (using K-means mask)
    # ========================================================================
    if foreground_mask is not None:
        # Resize mask to match rendered image size if needed
        if foreground_mask.shape != magnitude.shape:
            # foreground_mask might be from K-means clustering at same DPI
            # Just verify shape matches
            if foreground_mask.shape[0] != magnitude.shape[0] or foreground_mask.shape[1] != magnitude.shape[1]:
                # Resize mask to match
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray((foreground_mask * 255).astype(np.uint8))
                mask_img_resized = mask_img.resize((magnitude.shape[1], magnitude.shape[0]), PILImage.NEAREST)
                foreground_mask = np.array(mask_img_resized) > 0

        foreground_only_stats, _ = compute_gradient_statistics(
            magnitude=magnitude,
            gray_image=gray_array,
            mask=foreground_mask.astype(np.uint8),
            method=method
        )
    else:
        # No mask provided - use same as full bbox
        foreground_only_stats = full_bbox_stats

    # ========================================================================
    # 4. ELA (Error Level Analysis)
    # ========================================================================
    ela_stats, _ = compute_ela_statistics(img, quality=ela_quality)

    # ========================================================================
    # 5. Total computation time
    # ========================================================================
    total_computation_time_ms = (time.time() - total_start_time) * 1000

    # Create result
    result = EdgeGradientAnalysis(
        full_bbox_stats=full_bbox_stats,
        foreground_only_stats=foreground_only_stats,
        ela_stats=ela_stats,
        total_computation_time_ms=total_computation_time_ms
    )

    return result


# ============================================================================
# MASK CREATION FROM COLOR CLUSTERING
# ============================================================================

def create_foreground_mask_from_clustering(
    pdf_path: str,
    page_num: int,
    bbox: Tuple[float, float, float, float],
    foreground_color: Tuple[int, int, int],
    background_color: Tuple[int, int, int],
    dpi: int = 600
) -> np.ndarray:
    """
    Create binary foreground mask by re-clustering with known colors.

    This is a helper function to create a foreground mask from the K-means
    foreground/background colors obtained from color clustering.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        bbox: (x0, y0, x1, y1) in BOTTOMLEFT coords
        foreground_color: RGB tuple from K-means (text color)
        background_color: RGB tuple from K-means (background color)
        dpi: Rendering DPI (must match color clustering DPI)

    Returns:
        Binary mask (H, W) where 1 = foreground, 0 = background
    """
    # Extract image
    img = extract_image_from_bbox(pdf_path, page_num, bbox, dpi)

    # Get all pixels
    pixels = np.array(img).reshape(-1, 3)

    # Compute distance to foreground and background
    fg_color = np.array(foreground_color, dtype=np.float32)
    bg_color = np.array(background_color, dtype=np.float32)

    dist_to_fg = np.linalg.norm(pixels - fg_color, axis=1)
    dist_to_bg = np.linalg.norm(pixels - bg_color, axis=1)

    # Assign to foreground if closer to foreground color
    labels = (dist_to_fg < dist_to_bg).astype(np.uint8)

    # Reshape to image dimensions
    mask = labels.reshape(img.height, img.width)

    return mask
