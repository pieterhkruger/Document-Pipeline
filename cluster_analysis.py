"""
Cluster Analysis for Enhanced Text Info
========================================

Performs K-means and DBSCAN clustering on enhanced text information features.

Features:
- RGB to single number conversion (luminance + hex decimal)
- Yeo-Johnson transformation for normality
- Feature standardization
- GPU-accelerated K-means with optimal K selection (Silhouette Score)
- DBSCAN with automatic eps determination (k-distance graph)
- Cluster statistics and comparisons

Usage:
    from cluster_analysis import perform_cluster_analysis

    result = perform_cluster_analysis(
        enhanced_text_info_path="path/to/enhanced_text_info.json",
        use_gpu=True
    )
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import torch

# Import PyTorch K-means
try:
    from pytorch_kmeans import kmeans_pytorch_simple, test_gpu_available
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


@dataclass
class ClusterResult:
    """Results from clustering analysis."""
    # Data
    feature_names: List[str]
    features_original: np.ndarray  # Original features
    features_transformed: np.ndarray  # After Yeo-Johnson
    features_standardized: np.ndarray  # After standardization
    texts: List[str]  # Text for each data point

    # Clustering tendency test
    hopkins_statistic: float  # Hopkins statistic (0-1, higher = more clustered)

    # K-means results
    kmeans_labels: np.ndarray
    kmeans_n_clusters: int
    kmeans_centroids: np.ndarray  # In standardized space
    kmeans_silhouette: float

    # DBSCAN results
    dbscan_labels: np.ndarray
    dbscan_eps: float
    dbscan_n_clusters: int
    dbscan_silhouette: Optional[float]

    # Anomaly detection results
    isolation_forest_scores: np.ndarray  # Anomaly scores from Isolation Forest
    lof_scores: np.ndarray  # Anomaly scores from LOF
    ensemble_anomaly_flags: np.ndarray  # Boolean array: True = anomaly
    ensemble_agreement_count: np.ndarray  # How many methods flagged each point (0-3)

    # Transformers (for inverse transform)
    yeo_johnson_transformer: PowerTransformer
    standard_scaler: StandardScaler

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'feature_names': self.feature_names,
            'n_samples': len(self.texts),
            'hopkins_statistic': float(self.hopkins_statistic),
            'kmeans': {
                'n_clusters': self.kmeans_n_clusters,
                'silhouette_score': float(self.kmeans_silhouette),
                'cluster_sizes': [int(np.sum(self.kmeans_labels == i)) for i in range(self.kmeans_n_clusters)]
            },
            'dbscan': {
                'n_clusters': self.dbscan_n_clusters,
                'eps': float(self.dbscan_eps),
                'silhouette_score': float(self.dbscan_silhouette) if self.dbscan_silhouette else None,
                'n_noise': int(np.sum(self.dbscan_labels == -1))
            },
            'anomaly_detection': {
                'n_anomalies': int(np.sum(self.ensemble_anomaly_flags)),
                'anomaly_rate': float(np.mean(self.ensemble_anomaly_flags)),
                'high_confidence_anomalies': int(np.sum(self.ensemble_agreement_count == 3)),  # All 3 methods agree
                'medium_confidence_anomalies': int(np.sum(self.ensemble_agreement_count == 2))  # 2/3 methods agree
            }
        }

    def get_cluster_centroids(self, labels: np.ndarray, cluster_id: int) -> Optional[np.ndarray]:
        """
        Get centroid for a specific cluster.

        Args:
            labels: Cluster labels
            cluster_id: Cluster to get centroid for

        Returns:
            Centroid in standardized space, or None if cluster is empty/invalid
        """
        # For K-means, use stored centroids
        if np.array_equal(labels, self.kmeans_labels) and cluster_id < self.kmeans_n_clusters:
            return self.kmeans_centroids[cluster_id]

        # For DBSCAN or other, calculate centroid
        mask = labels == cluster_id
        if not np.any(mask):
            return None

        return np.mean(self.features_standardized[mask], axis=0)

    def find_nearest_neighbor_cluster(self, labels: np.ndarray, cluster_id: int) -> Optional[int]:
        """
        Find the nearest neighboring cluster based on centroid distance.

        Args:
            labels: Cluster labels
            cluster_id: Cluster to find nearest neighbor for

        Returns:
            Nearest cluster ID, or None if no valid neighbor
        """
        centroid = self.get_cluster_centroids(labels, cluster_id)
        if centroid is None:
            return None

        # Get all unique clusters (excluding noise and current cluster)
        unique_clusters = set(labels)
        unique_clusters.discard(-1)  # Remove noise
        unique_clusters.discard(cluster_id)  # Remove current cluster

        if not unique_clusters:
            return None

        # Find nearest neighbor
        min_distance = float('inf')
        nearest_cluster = None

        for other_cluster in unique_clusters:
            other_centroid = self.get_cluster_centroids(labels, other_cluster)
            if other_centroid is None:
                continue

            distance = np.linalg.norm(centroid - other_centroid)
            if distance < min_distance:
                min_distance = distance
                nearest_cluster = other_cluster

        return nearest_cluster

    def get_feature_importance_for_separation(
        self,
        labels: np.ndarray,
        cluster_id: int,
        top_n: int = 3
    ) -> List[Tuple[str, float, float]]:
        """
        Get top features that contribute to separation from nearest neighbor cluster.

        For each feature, calculate the absolute standardized difference between
        the cluster centroid and its nearest neighbor centroid. Higher values
        indicate features that contribute more to cluster separation.

        Args:
            labels: Cluster labels (kmeans_labels or dbscan_labels)
            cluster_id: Cluster to analyze
            top_n: Number of top features to return (default: 3)

        Returns:
            List of tuples: (feature_name, absolute_difference, percentage_contribution)
            Sorted by contribution (highest first)
        """
        # Get current cluster centroid
        centroid = self.get_cluster_centroids(labels, cluster_id)
        if centroid is None:
            return []

        # Find nearest neighbor cluster
        nearest_cluster = self.find_nearest_neighbor_cluster(labels, cluster_id)
        if nearest_cluster is None:
            return []

        # Get nearest neighbor centroid
        neighbor_centroid = self.get_cluster_centroids(labels, nearest_cluster)
        if neighbor_centroid is None:
            return []

        # Calculate absolute differences for each feature
        differences = np.abs(centroid - neighbor_centroid)

        # Calculate percentage contribution of each feature
        # Using squared differences to match Euclidean distance calculation
        squared_differences = differences ** 2
        total_squared = np.sum(squared_differences)

        # Get feature importance
        feature_importance = []
        for i, feature_name in enumerate(self.feature_names):
            abs_diff = differences[i]
            # Percentage contribution: (feature_squared_diff / total_squared) * 100
            percentage = (squared_differences[i] / total_squared * 100) if total_squared > 0 else 0
            feature_importance.append((feature_name, abs_diff, percentage))

        # Sort by percentage contribution (descending)
        feature_importance.sort(key=lambda x: x[2], reverse=True)

        # Return top N
        return feature_importance[:top_n]


def get_human_explanation_for_feature(feature_name: str, cluster_value: float, neighbor_value: float) -> str:
    """
    Convert feature difference into human-readable explanation.

    Args:
        feature_name: Name of the feature
        cluster_value: Value in current cluster
        neighbor_value: Value in nearest neighbor cluster

    Returns:
        Human-readable explanation of what to look for
    """
    # Feature descriptions and what to look for
    explanations = {
        'fg_luminance': {
            'name': 'Text brightness',
            'higher': 'lighter, brighter text',
            'lower': 'darker text',
            'look_for': 'Compare the brightness/darkness of the text itself'
        },
        'fg_decimal': {
            'name': 'Text color value',
            'higher': 'different text color (numerically higher)',
            'lower': 'different text color (numerically lower)',
            'look_for': 'Look for differences in the actual color of the text'
        },
        'bg_luminance': {
            'name': 'Background brightness',
            'higher': 'lighter, brighter background behind the text',
            'lower': 'darker background behind the text',
            'look_for': 'Compare the brightness of the paper/background behind the text'
        },
        'bg_decimal': {
            'name': 'Background color value',
            'higher': 'different background color (numerically higher)',
            'lower': 'different background color (numerically lower)',
            'look_for': 'Look for differences in the background color behind the text'
        },
        'text_to_bg_ratio': {
            'name': 'Text-to-background contrast',
            'higher': 'higher contrast between text and background',
            'lower': 'lower contrast between text and background',
            'look_for': 'Notice how much the text stands out from its background'
        },
        'gradient_mean': {
            'name': 'Text edge sharpness',
            'higher': 'sharper, crisper text edges',
            'lower': 'softer, blurrier text edges',
            'look_for': 'Examine how sharp or blurry the edges of the letters appear'
        },
        'gradient_std': {
            'name': 'Edge sharpness variation',
            'higher': 'more variation in edge sharpness (mixed sharp and soft)',
            'lower': 'more uniform edge sharpness',
            'look_for': 'Check if text edges have consistent sharpness or vary'
        },
        'gradient_max': {
            'name': 'Maximum edge sharpness',
            'higher': 'some very sharp edges present',
            'lower': 'all edges are relatively soft',
            'look_for': 'Look for the sharpest edges in the text'
        },
        'high_freq_energy': {
            'name': 'Fine detail presence',
            'higher': 'more fine details and texture in the text',
            'lower': 'smoother, less detailed text',
            'look_for': 'Notice the level of fine detail and texture in the characters'
        },
        'low_freq_energy': {
            'name': 'Broad shape presence',
            'higher': 'stronger overall shapes and forms',
            'lower': 'weaker overall shapes',
            'look_for': 'Focus on the bold, overall shapes rather than fine details'
        },
        'high_to_low_ratio': {
            'name': 'Detail-to-shape ratio',
            'higher': 'more fine details relative to overall shape',
            'lower': 'more emphasis on overall shape than fine details',
            'look_for': 'Compare the balance between fine details and broad strokes'
        },
        'laplacian_mean': {
            'name': 'Edge intensity',
            'higher': 'stronger, more pronounced edges',
            'lower': 'weaker, subtler edges',
            'look_for': 'Notice how strong the boundaries between text and background are'
        },
        'laplacian_zero_crossings': {
            'name': 'Edge complexity',
            'higher': 'more complex edge structure',
            'lower': 'simpler edge structure',
            'look_for': 'Look at how intricate the letter edges and curves are'
        },
        'local_variance_mean': {
            'name': 'Texture smoothness',
            'higher': 'rougher, more textured appearance',
            'lower': 'smoother, more uniform appearance',
            'look_for': 'Check if the text looks rough/grainy or smooth/uniform'
        },
        'bilateral_difference_mean': {
            'name': 'Edge preservation quality',
            'higher': 'edges are well-preserved and distinct',
            'lower': 'edges are blended or smoothed',
            'look_for': 'Notice how well-defined the text boundaries are'
        }
    }

    # Get base explanation
    if feature_name not in explanations:
        return f"Look for differences in {feature_name.replace('_', ' ')}"

    info = explanations[feature_name]
    direction = 'higher' if cluster_value > neighbor_value else 'lower'

    # Build human-readable explanation
    explanation = f"**{info['name'].title()}**: This cluster has {info[direction]}. {info['look_for']}."

    return explanation


def calculate_hopkins_statistic(features: np.ndarray, sample_size: int = None) -> float:
    """
    Calculate Hopkins statistic to test for clustering tendency.

    Hopkins statistic measures spatial randomness:
    - H close to 0.5: Random data (no clustering structure)
    - H close to 1.0: Data has clustering tendency (clustering is appropriate)
    - H close to 0.0: Regularly spaced data (no clustering)

    Interpretation (with dynamic thresholds based on sample size):
    - Small datasets (n < 30): H > 0.75 needed for clustering
    - Medium datasets (30-100): H > 0.70 needed
    - Large datasets (100+): H > 0.65 needed

    Traditional thresholds:
    - H < 0.3: Regular/uniformly distributed (clustering not appropriate)
    - 0.3 ≤ H ≤ 0.7: Random data (clustering questionable)
    - H > 0.7: Clustered data (clustering appropriate)

    Args:
        features: Feature matrix (n_samples x n_features)
        sample_size: Number of samples to use (default: dynamic based on n)

    Returns:
        Hopkins statistic (0 to 1)
    """
    n_samples, n_features = features.shape

    # Early return for very small datasets
    if n_samples < 5:
        return 0.5  # Not enough data to assess clustering tendency

    # Dynamic sample size based on dataset size
    # Based on approach from Forgery_detection/analyze_edge_gradient_clustering.py
    if sample_size is None:
        # Formula: min(max(10, 0.1*n), 100)
        # - Minimum: 10 samples
        # - Default: 10% of dataset
        # - Maximum: 100 samples
        sample_size = int(min(max(10, 0.1 * n_samples), 100))

    # Cannot sample more than half the dataset (avoid sampling issues)
    sample_size = min(sample_size, n_samples // 2)

    # Randomly sample points from the data
    sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
    sampled_data = features[sample_indices]

    # Generate random points within the data space
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    random_points = np.random.uniform(min_vals, max_vals, size=(sample_size, n_features))

    # Calculate nearest neighbor distances for sampled real data points
    from scipy.spatial.distance import cdist

    # For each sampled point, find distance to nearest neighbor (excluding itself)
    remaining_data = np.delete(features, sample_indices, axis=0)
    distances_real = cdist(sampled_data, remaining_data, metric='euclidean')
    u = np.min(distances_real, axis=1)

    # For each random point, find distance to nearest real data point
    distances_random = cdist(random_points, features, metric='euclidean')
    w = np.min(distances_random, axis=1)

    # Calculate Hopkins statistic
    hopkins = np.sum(w) / (np.sum(u) + np.sum(w))

    return hopkins


def rgb_to_luminance(r: int, g: int, b: int) -> float:
    """Convert RGB to luminance (perceptually weighted)."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def rgb_to_decimal(r: int, g: int, b: int) -> int:
    """Convert RGB to decimal (hex-like: R*65536 + G*256 + B)."""
    return r * 65536 + g * 256 + b


def extract_features_from_enhanced_info(enhanced_text_info_path: Path) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract features from enhanced text info JSON.

    Returns:
        Tuple of (features_array, feature_names, texts)
    """
    with open(enhanced_text_info_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    elements = data.get('elements', [])

    if not elements:
        raise ValueError("No elements found in enhanced text info")

    features_list = []
    texts = []

    for elem in elements:
        # Extract text
        cell_text = elem.get('cell_data', {}).get('text', '')
        texts.append(cell_text)

        # Extract color features
        pillow = elem.get('pillow_analysis', {})
        fg_color = pillow.get('foreground_color', [0, 0, 0])
        bg_color = pillow.get('background_color', [255, 255, 255])

        fg_luminance = rgb_to_luminance(*fg_color)
        fg_decimal = rgb_to_decimal(*fg_color)
        bg_luminance = rgb_to_luminance(*bg_color)
        bg_decimal = rgb_to_decimal(*bg_color)
        text_to_bg_ratio = pillow.get('text_to_background_ratio', 0.0)

        # Extract edge/gradient features
        edge_analysis = elem.get('edge_gradient_analysis', {})

        # Full bbox summary stats
        full_bbox = edge_analysis.get('full_bbox', {})
        summary_stats = full_bbox.get('summary_stats', {})

        mean_grad = summary_stats.get('mean', 0.0)
        std_grad = summary_stats.get('std', 0.0)
        q1_grad = summary_stats.get('q1', 0.0)
        median_grad = summary_stats.get('median', 0.0)
        q3_grad = summary_stats.get('q3', 0.0)
        p95_grad = summary_stats.get('p95', 0.0)
        p99_grad = summary_stats.get('p99', 0.0)
        max_grad = summary_stats.get('max', 0.0)

        # Frequency energy
        freq_energy = full_bbox.get('frequency_energy', {})
        high_freq = freq_energy.get('high_freq', 0.0)
        low_freq = freq_energy.get('low_freq', 0.0)
        high_to_low = freq_energy.get('high_to_low_ratio', 0.0)
        # Handle infinity
        if high_to_low == float('inf') or high_to_low == "Infinity":
            high_to_low = 1000.0  # Cap at large value

        # Laplacian
        laplacian = full_bbox.get('laplacian', {})
        lap_mean = laplacian.get('mean', 0.0)
        lap_std = laplacian.get('std', 0.0)
        lap_max = laplacian.get('max', 0.0)
        lap_p95 = laplacian.get('p95', 0.0)
        lap_zero_crossings = laplacian.get('zero_crossings', 0)

        # Local variance
        local_var = full_bbox.get('local_variance', {})
        lv_mean = local_var.get('mean', 0.0)
        lv_std = local_var.get('std', 0.0)
        lv_max = local_var.get('max', 0.0)

        # Bilateral
        bilateral = full_bbox.get('bilateral', {})
        bil_diff_mean = bilateral.get('difference_mean', 0.0)
        bil_edge_pres = bilateral.get('edge_preservation', 0.0)

        # Compile features
        features = [
            fg_luminance, fg_decimal, bg_luminance, bg_decimal, text_to_bg_ratio,
            mean_grad, std_grad, q1_grad, median_grad, q3_grad, p95_grad, p99_grad, max_grad,
            high_freq, low_freq, high_to_low,
            lap_mean, lap_std, lap_max, lap_p95, lap_zero_crossings,
            lv_mean, lv_std, lv_max,
            bil_diff_mean, bil_edge_pres
        ]

        features_list.append(features)

    # Feature names
    feature_names = [
        'fg_luminance', 'fg_decimal', 'bg_luminance', 'bg_decimal', 'text_to_bg_ratio',
        'mean_grad', 'std_grad', 'q1_grad', 'median_grad', 'q3_grad', 'p95_grad', 'p99_grad', 'max_grad',
        'high_freq', 'low_freq', 'high_to_low',
        'lap_mean', 'lap_std', 'lap_max', 'lap_p95', 'lap_zero_crossings',
        'lv_mean', 'lv_std', 'lv_max',
        'bil_diff_mean', 'bil_edge_pres'
    ]

    features_array = np.array(features_list, dtype=np.float64)

    return features_array, feature_names, texts


def apply_yeo_johnson_transform(features: np.ndarray) -> Tuple[np.ndarray, PowerTransformer]:
    """
    Apply Yeo-Johnson transformation to make distributions more normal.

    Returns:
        Tuple of (transformed_features, transformer)
    """
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    transformed = transformer.fit_transform(features)
    return transformed, transformer


def standardize_features(features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize features (mean=0, std=1).

    Returns:
        Tuple of (standardized_features, scaler)
    """
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    return standardized, scaler


def find_optimal_k_kmeans(features: np.ndarray, k_range: range = range(2, 5), use_gpu: bool = True) -> Tuple[int, Dict[int, float]]:
    """
    Find optimal number of clusters using Silhouette Score.

    Args:
        features: Standardized features
        k_range: Range of K values to try (default: 2-4)
        use_gpu: Use GPU if available

    Returns:
        Tuple of (optimal_k, silhouette_scores_dict)
    """
    silhouette_scores = {}

    for k in k_range:
        # Run K-means
        if PYTORCH_AVAILABLE and use_gpu and test_gpu_available():
            centroids, labels, _ = kmeans_pytorch_simple(
                data=features,
                n_clusters=k,
                device='cuda',
                random_state=42
            )
            labels = labels.astype(int)
        else:
            # Fall back to sklearn
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)

        # Calculate silhouette score
        score = silhouette_score(features, labels)
        silhouette_scores[k] = score

    # Find optimal K (highest silhouette score)
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)

    return optimal_k, silhouette_scores


def perform_kmeans_clustering(features: np.ndarray, n_clusters: int, use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform K-means clustering.

    Returns:
        Tuple of (labels, centroids, silhouette_score)
    """
    if PYTORCH_AVAILABLE and use_gpu and test_gpu_available():
        centroids, labels, _ = kmeans_pytorch_simple(
            data=features,
            n_clusters=n_clusters,
            device='cuda',
            random_state=42
        )
        labels = labels.astype(int)
        # Convert centroids to numpy
        if isinstance(centroids, torch.Tensor):
            centroids = centroids.cpu().numpy()
    else:
        # Fall back to sklearn
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        centroids = kmeans.cluster_centers_

    # Calculate silhouette score
    sil_score = silhouette_score(features, labels)

    return labels, centroids, sil_score


def find_optimal_eps_dbscan(features: np.ndarray, k: int = 4) -> float:
    """
    Find optimal eps for DBSCAN using k-distance graph.

    Args:
        features: Standardized features
        k: Number of neighbors (default: 4, typically min_samples - 1)

    Returns:
        Optimal eps value
    """
    # Compute k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(features)
    distances, indices = neighbors.kneighbors(features)

    # Sort distances
    distances = np.sort(distances[:, k-1], axis=0)

    # Find elbow point (use 90th percentile as heuristic)
    eps = np.percentile(distances, 90)

    return eps


def perform_dbscan_clustering(features: np.ndarray, eps: float, min_samples: int = 5) -> Tuple[np.ndarray, int, Optional[float]]:
    """
    Perform DBSCAN clustering.

    Returns:
        Tuple of (labels, n_clusters, silhouette_score)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)

    # Count clusters (excluding noise label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Calculate silhouette score (if we have at least 2 clusters and not all noise)
    sil_score = None
    if n_clusters >= 2 and len(set(labels)) > 1:
        try:
            sil_score = silhouette_score(features, labels)
        except:
            pass

    return labels, n_clusters, sil_score


def perform_ensemble_anomaly_detection(
    features: np.ndarray,
    dbscan_labels: np.ndarray,
    contamination: float = 0.1,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform ensemble anomaly detection using three methods:
    1. Isolation Forest
    2. Local Outlier Factor (LOF)
    3. DBSCAN noise points

    Conservative approach: Flag as anomaly only if 2+ methods agree.

    Args:
        features: Standardized feature matrix
        dbscan_labels: Labels from DBSCAN (-1 = noise)
        contamination: Expected proportion of anomalies (default: 0.1 = 10%)
        verbose: Print progress messages

    Returns:
        Tuple of (isolation_forest_scores, lof_scores, ensemble_flags, agreement_count)
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    if verbose:
        print(f"\n[Anomaly Detection]")
        print(f"  Using ensemble approach: Isolation Forest + LOF + DBSCAN")

    # Method 1: Isolation Forest
    # Scores are negative - more negative = more anomalous
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    iso_predictions = iso_forest.fit_predict(features)  # -1 = anomaly, 1 = normal
    iso_scores = iso_forest.score_samples(features)  # Anomaly scores
    iso_anomalies = (iso_predictions == -1)

    if verbose:
        print(f"  Isolation Forest: {np.sum(iso_anomalies)} anomalies detected")

    # Method 2: Local Outlier Factor (LOF)
    # Scores < -1.5 are typically considered anomalies
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=False,  # Fit and predict on same data
        n_jobs=-1
    )
    lof_predictions = lof.fit_predict(features)  # -1 = anomaly, 1 = normal
    lof_scores = lof.negative_outlier_factor_  # More negative = more anomalous
    lof_anomalies = (lof_predictions == -1)

    if verbose:
        print(f"  LOF: {np.sum(lof_anomalies)} anomalies detected")

    # Method 3: DBSCAN noise points
    # Already computed, just extract
    dbscan_anomalies = (dbscan_labels == -1)

    if verbose:
        print(f"  DBSCAN noise: {np.sum(dbscan_anomalies)} anomalies detected")

    # Ensemble voting: Count how many methods flagged each point
    agreement_count = (
        iso_anomalies.astype(int) +
        lof_anomalies.astype(int) +
        dbscan_anomalies.astype(int)
    )

    # Conservative threshold: Require 2+ methods to agree
    ensemble_flags = (agreement_count >= 2)

    if verbose:
        print(f"\n  Ensemble Results:")
        print(f"    High confidence (3/3 agree): {np.sum(agreement_count == 3)} anomalies")
        print(f"    Medium confidence (2/3 agree): {np.sum(agreement_count == 2)} anomalies")
        print(f"    Total flagged: {np.sum(ensemble_flags)} anomalies ({np.mean(ensemble_flags)*100:.1f}%)")

    return iso_scores, lof_scores, ensemble_flags, agreement_count


def perform_cluster_analysis(
    enhanced_text_info_path: Path,
    use_gpu: bool = True,
    max_k: int = 4,
    verbose: bool = False
) -> ClusterResult:
    """
    Perform complete cluster analysis on enhanced text info.

    Args:
        enhanced_text_info_path: Path to enhanced text info JSON
        use_gpu: Use GPU for K-means if available
        max_k: Maximum number of clusters to try (default: 4)
        verbose: Print progress messages

    Returns:
        ClusterResult with all clustering results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"CLUSTER ANALYSIS")
        print(f"{'='*80}")
        print(f"File: {enhanced_text_info_path.name}")

    # Step 1: Extract features
    if verbose:
        print(f"\n[1/8] Extracting features...")

    features_original, feature_names, texts = extract_features_from_enhanced_info(enhanced_text_info_path)

    n_samples = len(texts)

    if verbose:
        print(f"  Extracted {n_samples} samples with {len(feature_names)} features")

    # Check if we have enough samples for clustering
    if n_samples < 2:
        if verbose:
            print(f"\n[WARNING] Only {n_samples} sample(s) found - clustering requires at least 2 samples")
            print(f"[WARNING] Skipping cluster analysis for this document")

        # Return a minimal result with default values
        # Create dummy transformers
        from sklearn.preprocessing import PowerTransformer, StandardScaler
        dummy_transformer = PowerTransformer()
        dummy_scaler = StandardScaler()

        # Fit dummy transformers on the single sample (will work but not meaningful)
        if n_samples == 1:
            dummy_data = features_original.reshape(1, -1)
            dummy_transformer.fit(dummy_data)
            dummy_scaler.fit(dummy_data)

        return ClusterResult(
            feature_names=feature_names,
            features_original=features_original,
            features_transformed=features_original,  # No transformation
            features_standardized=features_original,  # No standardization
            texts=texts,
            hopkins_statistic=0.5,  # Neutral value
            kmeans_labels=np.array([0] * n_samples),
            kmeans_n_clusters=1,
            kmeans_centroids=features_original,  # Use original features as centroid
            kmeans_silhouette=0.0,
            dbscan_labels=np.array([-1] * n_samples),  # All noise
            dbscan_eps=0.5,
            dbscan_n_clusters=0,
            dbscan_silhouette=None,
            isolation_forest_scores=np.zeros(n_samples),
            lof_scores=np.zeros(n_samples),
            ensemble_anomaly_flags=np.zeros(n_samples, dtype=bool),
            ensemble_agreement_count=np.zeros(n_samples, dtype=int),
            yeo_johnson_transformer=dummy_transformer,
            standard_scaler=dummy_scaler
        )

    # Step 2: Yeo-Johnson transformation
    if verbose:
        print(f"\n[2/8] Applying Yeo-Johnson transformation...")

    features_transformed, yj_transformer = apply_yeo_johnson_transform(features_original)

    # Step 3: Standardization
    if verbose:
        print(f"\n[3/8] Standardizing features...")

    features_standardized, scaler = standardize_features(features_transformed)

    # Step 4: Hopkins statistic (clustering tendency test)
    if verbose:
        print(f"\n[4/8] Calculating Hopkins statistic (clustering tendency test)...")

    hopkins_stat = calculate_hopkins_statistic(features_standardized)

    if verbose:
        print(f"  Hopkins statistic: {hopkins_stat:.4f}")
        if hopkins_stat < 0.3:
            print(f"  [WARNING]  WARNING: Low Hopkins score (<0.3) suggests data is uniformly distributed.")
            print(f"  [WARNING]  Clustering may not be appropriate for this dataset.")
        elif hopkins_stat < 0.7:
            print(f"  [WARNING]  WARNING: Moderate Hopkins score (0.3-0.7) suggests random data.")
            print(f"  [WARNING]  Clustering results may not be meaningful.")
        else:
            print(f"  [OK] Hopkins score (>0.7) indicates clustering tendency is present.")

    # Step 5: K-means with optimal K selection
    if verbose:
        print(f"\n[5/8] Running K-means clustering...")
        print(f"  Finding optimal K (range: 2-{max_k})...")

    optimal_k, sil_scores = find_optimal_k_kmeans(
        features_standardized,
        k_range=range(2, max_k + 1),
        use_gpu=use_gpu
    )

    if verbose:
        print(f"  Silhouette scores: {sil_scores}")
        print(f"  Optimal K: {optimal_k} (score: {sil_scores[optimal_k]:.4f})")
        print(f"  Running final K-means with K={optimal_k}...")

    kmeans_labels, kmeans_centroids, kmeans_sil = perform_kmeans_clustering(
        features_standardized,
        n_clusters=optimal_k,
        use_gpu=use_gpu
    )

    if verbose:
        print(f"  K-means complete. Silhouette score: {kmeans_sil:.4f}")

    # Step 6: DBSCAN
    if verbose:
        print(f"\n[6/8] Running DBSCAN clustering...")
        print(f"  Finding optimal eps...")

    dbscan_eps = find_optimal_eps_dbscan(features_standardized)

    if verbose:
        print(f"  Optimal eps: {dbscan_eps:.4f}")
        print(f"  Running DBSCAN...")

    dbscan_labels, dbscan_n_clusters, dbscan_sil = perform_dbscan_clustering(
        features_standardized,
        eps=dbscan_eps,
        min_samples=5
    )

    if verbose:
        print(f"  DBSCAN complete. Clusters: {dbscan_n_clusters}")
        if dbscan_sil:
            print(f"  Silhouette score: {dbscan_sil:.4f}")

    # Step 7: Ensemble Anomaly Detection
    if verbose:
        print(f"\n[7/8] Performing ensemble anomaly detection...")

    iso_scores, lof_scores, ensemble_flags, agreement_count = perform_ensemble_anomaly_detection(
        features=features_standardized,
        dbscan_labels=dbscan_labels,
        contamination=0.03,  # Expect only 3% anomalies (very conservative to reduce false positives)
        verbose=verbose
    )

    # Step 8: Create result
    if verbose:
        print(f"\n[8/8] Creating result...")

    result = ClusterResult(
        feature_names=feature_names,
        features_original=features_original,
        features_transformed=features_transformed,
        features_standardized=features_standardized,
        texts=texts,
        hopkins_statistic=hopkins_stat,
        kmeans_labels=kmeans_labels,
        kmeans_n_clusters=optimal_k,
        kmeans_centroids=kmeans_centroids,
        kmeans_silhouette=kmeans_sil,
        dbscan_labels=dbscan_labels,
        dbscan_eps=dbscan_eps,
        dbscan_n_clusters=dbscan_n_clusters,
        dbscan_silhouette=dbscan_sil,
        isolation_forest_scores=iso_scores,
        lof_scores=lof_scores,
        ensemble_anomaly_flags=ensemble_flags,
        ensemble_agreement_count=agreement_count,
        yeo_johnson_transformer=yj_transformer,
        standard_scaler=scaler
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"CLUSTER ANALYSIS COMPLETE")
        print(f"{'='*80}\n")

    return result
