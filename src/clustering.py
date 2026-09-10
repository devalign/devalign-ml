import logging
import numpy as np
import umap
import hdbscan
from sklearn.metrics import silhouette_score
from src import config

logger = logging.getLogger(__name__)

class ClusteringPipeline:
    def __init__(self):
        # Configure UMAP
        self.reducer = umap.UMAP(
            n_neighbors=config.UMAP_N_NEIGHBORS,
            n_components=config.UMAP_N_COMPONENTS,
            min_dist=config.UMAP_MIN_DIST,
            metric=config.UMAP_METRIC,
            random_state=config.UMAP_RANDOM_STATE
        )
        
        # Configure HDBSCAN
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=config.HDBSCAN_MIN_SAMPLES,
            metric=config.HDBSCAN_METRIC,
            prediction_data=True
        )

    def run(self, vectors):
        """Runs UMAP + HDBSCAN, computes metrics, and reassigns noise points."""
        logger.info(f"Applying UMAP reduction (1024d -> {config.UMAP_N_COMPONENTS}d)...")
        X_reduced = self.reducer.fit_transform(vectors)

        logger.info("Running HDBSCAN clustering...")
        labels = self.clusterer.fit_predict(X_reduced)

        unique_labels = set(labels)
        non_noise_labels = sorted([l for l in unique_labels if l != -1])
        n_clusters = len(non_noise_labels)
        noise_count = int(np.sum(labels == -1))
        noise_pct = (noise_count / len(labels)) * 100

        logger.info(f"HDBSCAN Results: Found {n_clusters} clusters. Noise points: {noise_count} ({noise_pct:.2f}%)")

        if n_clusters < 2:
            logger.error("HDBSCAN found less than 2 clusters. Unable to compute Silhouette Score.")
            return {
                "labels": labels,
                "final_labels": labels,
                "X_reduced": X_reduced,
                "silhouette_score": 0.0,
                "centroids_1024d": {}
            }

        # Calculate Silhouette Score on non-noise points
        non_noise_mask = labels != -1
        X_clustered = X_reduced[non_noise_mask]
        labels_clustered = labels[non_noise_mask]

        sil_score = 0.0
        try:
            sil_score = silhouette_score(X_clustered, labels_clustered, metric='euclidean')
            logger.info(f"Silhouette Score (excl. noise): {sil_score:.4f}")
        except Exception as e:
            logger.error(f"Failed to calculate silhouette score: {e}")

        # Preserve HDBSCAN noise points as -1 (unassigned outliers) to avoid cluster pollution
        final_labels = np.array(labels)
        noise_count = int(np.sum(final_labels == -1))
        logger.info(f"HDBSCAN: Preserving {noise_count} noise offers as unassigned (-1).")

        # Compute 1024d centroid vectors for each cluster (using core non-noise members)
        cluster_centroids_1024d = {}
        for k in non_noise_labels:
            core_indices = [idx for idx, l in enumerate(labels) if l == k]
            core_vectors = vectors[core_indices]
            cluster_centroids_1024d[k] = np.mean(core_vectors, axis=0).tolist()

        return {
            "labels": labels,
            "final_labels": final_labels,
            "X_reduced": X_reduced,
            "silhouette_score": sil_score,
            "centroids_1024d": cluster_centroids_1024d,
            "non_noise_labels": non_noise_labels
        }
