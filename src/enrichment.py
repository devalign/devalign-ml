import os
import json
import logging
import argparse
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# We assume config.py handles dotenv loading and setting these variables
from src import config
from supabase import create_client, Client
from groq import Groq, APIError

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Enrichment")

# --- 1. Definición de Enums ---
class ModalityEnum(str, Enum):
    remote = "Remoto"
    hybrid = "Híbrido"
    onsite = "Presencial"
    unknown = "No especificado"

class EducationEnum(str, Enum):
    universitaria = "Universitaria"
    tecnica = "Técnica"
    postgrado = "Postgrado"
    unknown = "No especificado"

# --- 2. Definición del Schema (Pydantic) ---
class ExtractedJobData(BaseModel):
    hard_skills: List[str] = Field(description="Lista de habilidades técnicas, lenguajes, frameworks, herramientas (ej. Python, AWS, Docker).")
    soft_skills: List[str] = Field(description="Lista de habilidades blandas, interpersonales o de gestión (ej. Liderazgo, Comunicación).")
    experience_years: Optional[str] = Field(None, description="Años de experiencia requeridos (ej. '3 years', 'No especificado').")
    salary: Optional[str] = Field(None, description="Rango salarial o monto ofrecido (ej. '$2000 - $3000 USD').")
    education_level: EducationEnum = Field(description="Nivel educativo requerido.")
    modality: ModalityEnum = Field(description="Modalidad de trabajo.")

    @model_validator(mode='before')
    @classmethod
    def normalize_fields(cls, values: dict) -> dict:
        if not isinstance(values, dict):
            return values

        # Normalize hard_skills to list
        hs = values.get("hard_skills")
        if isinstance(hs, str):
            values["hard_skills"] = [hs]
        elif hs is None or "hard_skills" not in values:
            values["hard_skills"] = []

        # Normalize soft_skills to list
        ss = values.get("soft_skills")
        if isinstance(ss, str):
            values["soft_skills"] = [ss]
        elif ss is None or "soft_skills" not in values:
            values["soft_skills"] = []

        # Normalize education_level
        edu = values.get("education_level")
        if edu is None:
            values["education_level"] = "No especificado"
        elif isinstance(edu, str):
            edu_lower = edu.lower().strip()
            if "univ" in edu_lower:
                values["education_level"] = "Universitaria"
            elif "tecn" in edu_lower or "técn" in edu_lower:
                values["education_level"] = "Técnica"
            elif "post" in edu_lower or "pos" in edu_lower or "maes" in edu_lower or "doc" in edu_lower:
                values["education_level"] = "Postgrado"
            else:
                values["education_level"] = "No especificado"

        # Normalize modality
        mod = values.get("modality")
        if mod is None:
            values["modality"] = "No especificado"
        elif isinstance(mod, str):
            mod_lower = mod.lower().strip()
            if "remot" in mod_lower or "home" in mod_lower or "tele" in mod_lower:
                values["modality"] = "Remoto"
            elif "hibr" in mod_lower or "híbr" in mod_lower or "mix" in mod_lower:
                values["modality"] = "Híbrido"
            elif "presen" in mod_lower or "site" in mod_lower or "oficina" in mod_lower:
                values["modality"] = "Presencial"
            else:
                values["modality"] = "No especificado"

        return values


# --- 3. Inicialización de Clientes ---
def get_supabase_client() -> Optional[Client]:
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        logger.error("Credenciales de Supabase no configuradas.")
        return None
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

def get_groq_client() -> Optional[Groq]:
    if not config.GROQ_API_KEY:
        logger.error("Credencial de Groq no configurada (GROQ_API_KEY).")
        return None
    return Groq(api_key=config.GROQ_API_KEY)


# --- 4. Lógica de Enriquecimiento (LLM) con Tenacity ---
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def enrich_description_with_groq(client: Groq, text: str) -> Optional[ExtractedJobData]:
    # Generamos una explicación amigable del formato esperado
    expected_format = {
        "hard_skills": ["Python", "AWS", "Docker"],
        "soft_skills": ["Liderazgo", "Trabajo en equipo"],
        "experience_years": "2 años",
        "salary": "S/. 3000 - S/. 4000",
        "education_level": "Universitaria",
        "modality": "Remoto"
    }

    prompt = (
        "Analiza la siguiente descripción de una oferta laboral de TI. "
        "Extrae los datos estructurados basándote ÚNICAMENTE en el texto proporcionado.\n\n"
        "DEBES devolver un JSON con esta estructura exacta (mismas llaves y tipos de datos):\n"
        f"{json.dumps(expected_format, indent=2)}\n\n"
        f"Los valores permitidos para 'education_level' son: {[e.value for e in EducationEnum]}\n"
        f"Los valores permitidos para 'modality' son: {[m.value for m in ModalityEnum]}\n\n"
        f"Descripción de la oferta:\n{text[:6000]}"
    )

    try:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto ingeniero de datos. Extraes datos en JSON exacto, respetando el schema solicitado."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"},
        )
        
        content = completion.choices[0].message.content
        if not content:
            return None
            
        data_dict = json.loads(content)
        
        try:
            return ExtractedJobData(**data_dict)
        except Exception as pydantic_err:
            logger.error(f"Error de validación Pydantic: {pydantic_err}. JSON Crudo: {data_dict}")
            return None
            
    except APIError as e:
        logger.warning(f"Error de API de Groq (Rate Limit u otro): {e}. Reintentando...")
        raise e
    except Exception as e:
        logger.error(f"Error inesperado al llamar a Groq: {e}")
        return None


# --- 5. Flujo Principal de Lotes ---
def run_enrichment(limit: int = 100):
    supabase = get_supabase_client()
    groq = get_groq_client()
    
    if not supabase or not groq:
        logger.error("No se pudo iniciar los clientes. Abortando.")
        return

    logger.info("Consultando Supabase por ofertas candidatas (ai_enriched = False)...")
    
    response = (
        supabase.table("job_offers")
        .select("job_offer_id, full_description, raw_hard_skills, raw_soft_skills")
        .eq("ai_enriched", False)
        .limit(limit)
        .execute()
    )
    candidates_to_process = response.data or []
    logger.info(f"Encontradas {len(candidates_to_process)} ofertas listas para enriquecer.")
    
    if not candidates_to_process:
        logger.info("No hay nada que procesar.")
        return

    updates_count = 0
    for offer in candidates_to_process:
        offer_id = str(offer["job_offer_id"])
        description = offer["full_description"]
        existing_hard_skills = offer.get("raw_hard_skills") or []
        existing_soft_skills = offer.get("raw_soft_skills") or []
        
        if not description:
            try:
                supabase.table("job_offers").update({"ai_enriched": True}).eq("job_offer_id", offer_id).execute()
            except Exception as e:
                logger.error(f"Error marcando oferta vacía ID {offer_id}: {e}")
            continue

        logger.info(f"Procesando oferta ID: {offer_id}...")
        
        enriched_data = enrich_description_with_groq(groq, description)
        
        if enriched_data:
            merged_hard_skills = list(set(existing_hard_skills + enriched_data.hard_skills))
            merged_soft_skills = list(set(existing_soft_skills + enriched_data.soft_skills))
            
            update_payload = {
                "raw_hard_skills": merged_hard_skills,
                "raw_soft_skills": merged_soft_skills,
                "is_normalized": False,
                "ai_enriched": True
            }
            
            if enriched_data.experience_years and "no especific" not in enriched_data.experience_years.lower():
                update_payload["experience_years"] = enriched_data.experience_years
                
            if enriched_data.salary and "no especific" not in enriched_data.salary.lower():
                update_payload["salary"] = enriched_data.salary
                
            if enriched_data.education_level != EducationEnum.unknown:
                update_payload["education_level"] = enriched_data.education_level.value
                
            if enriched_data.modality != ModalityEnum.unknown:
                update_payload["modality"] = enriched_data.modality.value

            try:
                supabase.table("offer_skills").delete().eq("job_offer_id", offer_id).execute()
                
                supabase.table("job_offers").update(update_payload).eq("job_offer_id", offer_id).execute()
                logger.info(f"[EXITO] ID {offer_id} enriquecido y reseteado para normalización.")
                updates_count += 1
            except Exception as e:
                logger.error(f"Error actualizando ID {offer_id} en Supabase: {e}")
        else:
            try:
                supabase.table("job_offers").update({"ai_enriched": True}).eq("job_offer_id", offer_id).execute()
                logger.warning(f"[OMITIDO] ID {offer_id} marcado como enriquecido (sin resultados válidos).")
            except Exception as e:
                logger.error(f"Error al omitir ID {offer_id} en Supabase: {e}")

    logger.info(f"Enriquecimiento finalizado. Se actualizaron {updates_count} ofertas en Supabase.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enriquecimiento de Job Offers mediante Groq LLM.")
    parser.add_argument("--limit", type=int, default=10, help="Límite máximo de ofertas a procesar en esta ejecución.")
    args = parser.parse_args()
    
    run_enrichment(limit=args.limit)
