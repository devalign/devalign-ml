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
            "You are a principal technical recruiter and IT software taxonomy expert. You are given a list of tech clusters, each characterized by its top skills and dominant technologies.\n"
            "Your task is to assign a unique, concise, professional title (max 5 words) in Spanish representing the SPECIFIC technical specialization or engineering role of each cluster, highlighting its core stack or focus.\n"
            "Valid examples: 'Desarrollador Backend Python & FastAPI', 'Ingeniero Cloud DevOps & Terraform', 'Especialista QA Automation', 'Desarrollador Full Stack React & Node', 'Ingeniero de Datos & ETL', 'Científico de Datos & ML', 'Desarrollador PHP / Laravel', 'Desarrollador Java & Spring Boot', 'Diseñador UI/UX & Frontend', 'Administrador de Bases de Datos (DBA)'.\n\n"
            "STRICT RULES:\n"
            "1. BE SPECIFIC TO THE TECH STACK: Do NOT use overly generic or macro labels like just 'Desarrollador de Software' or 'Desarrollador Full Stack' if specific technologies (e.g. React, Python, Java, PHP, DevOps, QA, Data) define the cluster.\n"
            "2. NO SPREADSHEETS / ADMINISTRATIVE TITLES: Never use names like 'Analista de Hojas', 'Especialista en Excel', or generic administrative titles.\n"
            "3. STRICTLY UNIQUE: Every cluster title MUST be 100% unique across all clusters. Never repeat a title.\n"
            "4. Respond ONLY with the list of cluster names in the exact format shown below, without any intro or outro text.\n\n"
            "Input clusters:\n"
        )
        for idx, item in enumerate(clusters_skills):
            if isinstance(item, dict):
                skills_str = ", ".join(item.get("skills", []))
                roles = item.get("roles", [])
                roles_str = f" | Puestos frecuentes: {', '.join(roles[:2])}" if roles else ""
                prompt += f"Cluster {idx}: Habilidades: {skills_str}{roles_str}\n"
            else:
                prompt += f"Cluster {idx}: Habilidades: {', '.join(item)}\n"

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
                "max_tokens": 1500,
            },
            timeout=40
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
                if idx in names_map and len(names_map[idx].strip()) > 2:
                    results.append(names_map[idx].strip())
                else:
                    logger.warning(f"Missing or empty cluster name for index {idx} in batch response")
                    return None

            # Deduplicate any identical names
            seen = {}
            deduped_results = []
            for idx, name in enumerate(results):
                if name in seen:
                    seen[name] += 1
                    item = clusters_skills[idx]
                    sk_list = item.get("skills", []) if isinstance(item, dict) else item

                    suffix = sk_list[0].title() if sk_list else str(seen[name])
                    deduped_results.append(f"{name} ({suffix})")
                else:
                    seen[name] = 1
                    deduped_results.append(name)

            return deduped_results
    except Exception as e:
        logger.warning(f"LLM batch naming failed: {e}")
    return None

def generate_cluster_name(top_skills, top_roles=None):
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

    user_content = f"Skills: {', '.join(top_skills)}"
    if top_roles:
        user_content += f" | Puestos frecuentes: {', '.join(top_roles[:2])}"

    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a technical recruiter expert in IT. Give a concise, specific professional title (max 5 words) in Spanish for an IT engineering specialty based on these top skills, highlighting the primary stack or role (e.g., 'Desarrollador Backend Python', 'Ingeniero DevOps Cloud', 'Especialista QA Automation', 'Desarrollador Java & Spring Boot'). Reply ONLY with the title.",
                    },
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "max_tokens": 150,
            },
            timeout=10
        )
        if response.status_code == 200:
            raw_content = response.json()["choices"][0]["message"].get("content") or ""
            cleaned = raw_content.strip().replace('"', "")
            if cleaned and len(cleaned) > 2:
                return cleaned
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
