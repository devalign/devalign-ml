import logging
import os
import requests
import matplotlib
# Use non-interactive backend for file generation (avoid Tkinter errors)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

logger = logging.getLogger(__name__)

def generate_cluster_names_batch(clusters_skills):
    """Generates unique names in Spanish for a batch of clusters using the configured naming strategy."""
    from src.services.naming_service import get_naming_strategy
    strategy = get_naming_strategy()
    results = []
    seen = {}
    for idx, item in enumerate(clusters_skills):
        skills = item.get("skills", []) if isinstance(item, dict) else item
        roles = item.get("roles", []) if isinstance(item, dict) else None
        freq_map = {s: 1.0 / (i + 1) for i, s in enumerate(skills)}
        name = strategy.name_cluster(freq_map, roles)
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        results.append(name)
    return results

def generate_cluster_name(top_skills, top_roles=None):
    """Generates a single cluster name using the configured naming strategy."""
    from src.services.naming_service import get_naming_strategy
    freq_map = {s: 1.0 / (i + 1) for i, s in enumerate(top_skills)}
    return get_naming_strategy().name_cluster(freq_map, top_roles)

def plot_and_save_clusters(X_reduced, labels, filename="hdbscan_clusters.png"):
    """Plots the UMAP-reduced clusters and saves the figure as an image."""
    try:
        plt.figure(figsize=(10, 8))
        
        # We project the first two dimensions of UMAP space for plotting
        sns.scatterplot(
            x=X_reduced[:, 0], 
            y=X_reduced[:, 1], 
            hue=labels, 
            palette='viridis', 
            legend='full', 
            alpha=0.8,
            edgecolor='w',
            linewidth=0.5
        )
        
        plt.title("Visualización de Clústeres UMAP + HDBSCAN (Habilidades Técnicas)")
        plt.xlabel("Dimensión UMAP 1")
        plt.ylabel("Dimensión UMAP 2")
        plt.legend(title="Etiqueta Clúster", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save output to models directory
        output_path = os.path.join(config.MODELS_DIR, filename)
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Scatter plot saved successfully to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate cluster plot: {e}")
        return None
