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
    """Generates unique names in Spanish for a batch of clusters using configured LLM API (OpenAI or Groq)."""
    if config.LLM_PROVIDER == "openai":
        if not config.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY is not defined. Skipping batch cluster naming.")
            return None
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
        model = config.LLM_MODEL or "gpt-4o-mini"
    else:
        if not config.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY is not defined. Skipping batch cluster naming.")
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
        model = config.LLM_MODEL or "llama-3.3-70b-versatile"

    try:
        prompt = (
            "You are a technical recruiter expert in IT. You are given a list of tech clusters, each characterized by its top skills. "
            "Your task is to assign a unique, concise, professional title (max 4 words) in Spanish for each IT specialty/cluster. "
            "For example: 'Desarrollador Java Backend', 'Ingeniero de Datos', 'Administrador de Bases de Datos', 'Desarrollador React Frontend'.\n\n"
            "CRITICAL RULES:\n"
            "1. Every single cluster name MUST be unique and distinct. Do NOT reuse names like 'Analista de Datos' or 'Desarrollador Full Stack' for different clusters. Differentiate them using their specific dominant skills.\n"
            "2. Be precise: if a cluster contains 'power bi' or 'tableau', prefer names like 'Analista de BI' or 'Especialista en Power BI'. If it contains only 'sql' and basic sheets, name it 'Analista de Datos Jr' or 'Analista de Soporte SQL'.\n"
            "3. Respond ONLY with the list of cluster names in the exact format shown below, without any intro or outro text.\n\n"
            "Input clusters:\n"
        )
        for idx, skills in enumerate(clusters_skills):
            prompt += f"Cluster {idx}: {', '.join(skills)}\n"

        prompt += "\nOutput format:\n"
        for idx in range(len(clusters_skills)):
            prompt += f"Cluster {idx}: <Name>\n"

        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
            },
            timeout=15
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            lines = content.strip().split("\n")
            names_map = {}
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    idx_str = parts[0].replace("Cluster", "").strip()
                    name = parts[1].strip().replace('"', '')
                    try:
                        idx = int(idx_str)
                        names_map[idx] = name
                    except ValueError:
                        pass

            results = []
            for idx in range(len(clusters_skills)):
                if idx in names_map:
                    results.append(names_map[idx])
                else:
                    logger.warning(f"Missing cluster name for index {idx} in batch response")
                    return None
            return results
    except Exception as e:
        logger.warning(f"LLM batch naming failed: {e}")
    return None

def generate_cluster_name(top_skills):
    """Generates a single cluster name as fallback."""
    if config.LLM_PROVIDER == "openai":
        if not config.OPENAI_API_KEY:
            return f"Especialidad: {' + '.join([n.title() for n in top_skills[:3]])}"
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
        model = config.LLM_MODEL or "gpt-4o-mini"
    else:
        if not config.GROQ_API_KEY:
            return f"Especialidad: {' + '.join([n.title() for n in top_skills[:3]])}"
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
        model = config.LLM_MODEL or "llama-3.3-70b-versatile"

    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a technical recruiter expert in IT. Give a concise, professional title (max 4 words) in Spanish for an IT specialty based on these top skills. Example: 'Desarrollador Java Backend', 'Ingeniero de Datos', 'Especialista DevOps Cloud'. Reply ONLY with the title.",
                    },
                    {"role": "user", "content": f"Skills: {', '.join(top_skills)}"},
                ],
                "temperature": 0.2,
                "max_tokens": 15,
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip().replace('"', "")
    except Exception as e:
        logger.warning(f"LLM naming failed: {e}")
    return f"Especialidad: {' + '.join([n.title() for n in top_skills[:3]])}"

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
