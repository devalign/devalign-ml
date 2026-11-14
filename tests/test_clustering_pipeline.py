"""Unit tests for ClusteringPipeline (UMAP + HDBSCAN)."""

import numpy as np

from src.clustering import ClusteringPipeline


def test_clustering_pipeline_synthetic_data():
    np.random.seed(42)
    # Generate 2 clear synthetic clusters in 1024d space
    c1 = np.random.normal(loc=1.0, scale=0.1, size=(30, 1024))
    c2 = np.random.normal(loc=-1.0, scale=0.1, size=(30, 1024))
    vectors = np.vstack([c1, c2])

    pipeline = ClusteringPipeline()
    # Temporarily tune min_cluster_size for small synthetic test dataset
    pipeline.clusterer.min_cluster_size = 10
    pipeline.clusterer.min_samples = 3

    results = pipeline.run(vectors)

    assert "labels" in results
    assert "final_labels" in results
    assert "X_reduced" in results
    assert "silhouette_score" in results
    assert "centroids_1024d" in results

    assert len(results["final_labels"]) == 60
    assert results["X_reduced"].shape == (60, 15)
    assert isinstance(results["silhouette_score"], float)


def test_clustering_pipeline_insufficient_clusters():
    # Only 5 random points: HDBSCAN won't find valid clusters
    vectors = np.random.normal(loc=0.0, scale=1.0, size=(5, 1024))

    pipeline = ClusteringPipeline()
    results = pipeline.run(vectors)

    assert results["silhouette_score"] == 0.0
    assert len(results["centroids_1024d"]) == 0
