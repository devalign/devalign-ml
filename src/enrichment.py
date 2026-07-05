import os
import json
import logging
import argparse
from typing import List, Optional
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# We assume config.py handles dotenv loading and setting these variables
from src import config
from supabase import create_client, Client
from groq import Groq, APIError

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Enrichment")

# --- 1. Definición del Schema (Pydantic) ---
class ExtractedJobData(BaseModel):
    hard_skills: List[str] = Field(description="Lista de habilidades técnicas, lenguajes, frameworks, herramientas (ej. Python, AWS, Docker).")
    soft_skills: List[str] = Field(description="Lista de habilidades blandas, interpersonales o de gestión (ej. Liderazgo, Comunicación).")
    experience_years: str = Field(description="Años de experiencia requeridos (ej. '3 years', 'No especificado').")
    salary: str = Field(description="Rango salarial o monto ofrecido (ej. '$2000 - $3000 USD').")
    education_level: str = Field(description="Nivel educativo requerido (ej. 'Universitaria', 'Técnica', 'Bachiller').")
    modality: str = Field(description="Modalidad de trabajo: 'Remoto', 'Presencial', o 'Híbrido'.")


# --- 2. Inicialización de Clientes ---
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


# --- 3. Lógica de Enriquecimiento (LLM) con Tenacity ---
# Reintenta si Groq devuelve error 429 o de conexión (Rate Limit).
# Espera 2, 4, 8, hasta 60 segundos entre intentos. Máximo 5 intentos.
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(APIError),
    reraise=True
)
def enrich_description_with_groq(client: Groq, text: str) -> Optional[ExtractedJobData]:
    # Instrucciones claras para el LLM
    prompt = (
        "Analiza la siguiente descripción de una oferta laboral de TI. "
        "Extrae los datos estructurados basándote ÚNICAMENTE en el texto proporcionado. "
        f"DEBES devolver un JSON que siga ESTRICTAMENTE este esquema:\n{json.dumps(ExtractedJobData.model_json_schema(), indent=2)}\n\n"
        f"Descripción:\n{text[:6000]}"  # Truncamos a 6k chars para evitar romper la ventana de contexto
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
            model="llama-3.1-8b-instant",  # Modelo rápido y barato
            response_format={"type": "json_object"},
            # Aquí inyectamos implícitamente qué keys debe tener. Groq es muy bueno siguiendo schemas.
            # Alternativamente en el prompt podíamos darle el JSON template. 
        )
        
        content = completion.choices[0].message.content
        if not content:
            return None
            
        # Parsea el string devuelto a Pydantic para validar
        data_dict = json.loads(content)
        
        # Validar y forzar default values si falla algo
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


# --- 4. Flujo Principal de Lotes ---
def run_enrichment(limit: int = 100):
    supabase = get_supabase_client()
    groq = get_groq_client()
    
    if not supabase or not groq:
        logger.error("No se pudo iniciar los clientes. Abortando.")
        return

    # Usaremos un archivo local temporal para trackear procesados si la DB no tiene columna 'ai_enriched'
    PROCESSED_FILE = os.path.join(config.DATA_DIR, "ai_processed_ids.json")
    processed_ids = set()
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, "r") as f:
            try:
                processed_ids = set(json.load(f))
            except:
                pass
                
    logger.info(f"Se han procesado {len(processed_ids)} ofertas históricamente.")

    # 1. Fetch Offers
    # Obtenemos ofertas. Asumimos que queremos las que no están en 'processed_ids'.
    logger.info("Consultando Supabase por ofertas candidatas...")
    
    # Hacemos una consulta grande, pero filtramos a nivel memoria local las ya procesadas.
    # Podría optimizarse si creamos un endpoint RPC o una vista.
    response = supabase.table("job_offers").select("job_offer_id, full_description").execute()
    all_offers = response.data or []
    
    candidates = [o for o in all_offers if o.get("full_description") and str(o["job_offer_id"]) not in processed_ids]
    logger.info(f"Encontradas {len(candidates)} ofertas listas para enriquecer.")
    
    candidates_to_process = candidates[:limit]
    
    if not candidates_to_process:
        logger.info("No hay nada que procesar.")
        return

    # 2. Procesamiento
    updates_count = 0
    for offer in candidates_to_process:
        offer_id = str(offer["job_offer_id"])
        description = offer["full_description"]
        
        logger.info(f"Procesando oferta ID: {offer_id}...")
        
        enriched_data = enrich_description_with_groq(groq, description)
        
        if enriched_data:
            # 3. Update Supabase
            update_payload = {
                "raw_hard_skills": enriched_data.hard_skills,
                "raw_soft_skills": enriched_data.soft_skills,
            }
            # Solo actualizamos los campos deterministas (salario, etc) si el LLM extrajo algo distinto de vacío
            # y podríamos decidir si sobreescribimos o no. Lo ideal es no sobreescribir si ya había un buen valor.
            # Por ahora, los mandamos a Supabase.
            if enriched_data.experience_years and "no especific" not in enriched_data.experience_years.lower():
                update_payload["experience_years"] = enriched_data.experience_years
                
            if enriched_data.salary and "no especific" not in enriched_data.salary.lower():
                update_payload["salary"] = enriched_data.salary
                
            if enriched_data.education_level and "no especific" not in enriched_data.education_level.lower():
                update_payload["education_level"] = enriched_data.education_level
                
            if enriched_data.modality and "no especific" not in enriched_data.modality.lower():
                update_payload["modality"] = enriched_data.modality

            try:
                supabase.table("job_offers").update(update_payload).eq("job_offer_id", offer_id).execute()
                logger.info(f"[EXITO] ID {offer_id} actualizado.")
                updates_count += 1
            except Exception as e:
                logger.error(f"Error actualizando ID {offer_id} en Supabase: {e}")
        
        # Guardar en archivo local como procesado incluso si falló el parseo,
        # para no entrar en loop infinito de procesar los mismos fallidos.
        processed_ids.add(offer_id)
        
        # Persistir checkpoint
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(PROCESSED_FILE, "w") as f:
            json.dump(list(processed_ids), f)

    logger.info(f"Enriquecimiento finalizado. Se actualizaron {updates_count} ofertas en Supabase.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enriquecimiento de Job Offers mediante Groq LLM.")
    parser.add_argument("--limit", type=int, default=10, help="Límite máximo de ofertas a procesar en esta ejecución.")
    args = parser.parse_args()
    
    run_enrichment(limit=args.limit)
