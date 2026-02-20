import os
import requests
import argparse
import json
import re
import base64
from datetime import datetime

def extract_extension(filepath:str):
    return filepath.split(".")[-1]

def prepare_text_message(patient_filepath:str, template:str):

    try:
        file_content = ""
        with open(patient_filepath, "r") as f:
            file_content = f.read()
    except Exception as e:
        raise e

    return {"role": "user", "content": f"""### DATA:
TEMPLATE:
{template}

FILE:
{file_content}

### FULLFILLED TEMPLATE:""" }


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_image_message(filepath):
    image_base64 = encode_image(filepath)
    # Detectamos el tipo mime básico según la extensión
    mime_type = "image/jpeg" if filepath.lower().endswith(('jpg', 'jpeg')) else "image/png"
    
    prompt_transcripcion = (
        "Act as a medical transcriber. Describe in detail all the clinical information, "
        "findings, and data present in this image. Use a professional medical tone."
    )

    return [{
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": f"{prompt_transcripcion}"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
            }
        ]
    }]

def call_medgemma(messages:list[dict]) -> str:

    lmstudio_url = "http://localhost:1234/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "medgemma-1.5-4b-it", # LMStudio identifier (Currently using Q4_0)
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": -1,    
        "top_p": 1.0, # let temperature rule the generation
        "seed": 314, 
        "repeat_penalty":1.15, #break infinite loops
        "top_k": 20,
    }

    try:
        response = requests.post(lmstudio_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Lanza error si la petición falla
        
        result = response.json()
        full_content = result['choices'][0]['message']['content']
        clean_content = re.sub(r'<unused94>.*?<unused95>', '', full_content, flags=re.DOTALL).strip()

        print("THINK CHANNEL:" + full_content.split("<unused95>")[0])
        return clean_content
    
    except Exception as e:
       raise e


def write_updated_template(updated_template:str, patient_folder: str):

    current_date = datetime.now().strftime("%Y%m%d")

    summary_filepath = os.path.join(patient_folder,f"{current_date}_summary.txt")

    with open(summary_filepath,"w",encoding="utf-8") as f:
        f.write(updated_template)

def preprocess_template(template_path):

    with open(template_path, "r", encoding="utf-8") as f:
        template_raw = f.read()

    # system_prompt = (
    #     "Act as a Template Architect. Your task is to transform any medical template into a "
    #     "structured list where every field ends with ': Not specified'.\n"
    #     "RULES:\n"
    #     "- Maintain the original hierarchy (1, a, b...).\n"
    #     "- Ensure every final field is followed by ': Not specified'.\n"
    #     "- GROUP fields by their natural sections.\n"
    #     "- Add EXACTLY TWO newlines (\\n\\n) between each major section.\n"
    #     "- Output ONLY the transformed template."
    # )

    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": f"Normalize this template:\n\n{template_raw}"}
    # ]

    # normalized_template = call_medgemma(messages)

    chunks = [chunk.strip() for chunk in template_raw.split("\n\n") if len(chunk.strip()) > 0]

    return chunks

def process_text_file(filepath, chunks):
    """
    Procesa el archivo médico completo contra cada chunk del template.
    """
    # 1. Leer el archivo médico completo (Contexto global)
    with open(filepath, "r", encoding="utf-8") as f:
        full_medical_file = f.read()

    # 2. Definir el System Prompt de extracción (Mano de Hierro)
    # Este prompt está optimizado para que el modelo solo se centre en el fragmento dado
    system_prompt_extractor = """You are a clinical data record keeper.

You will receive:
1) MEDICAL FILE (full text).
2) TEMPLATE FRAGMENT (part of a larger template).

RULES (MANDATORY):
- Only fill fields that ALREADY EXIST in the TEMPLATE FRAGMENT.
- NEVER create new fields, headings, bullets or comments.
- If the MEDICAL FILE clearly provides information for a field, you MUST UPDATE that field, even if the TEMPLATE FRAGMENT already contains a previous value.
- If the MEDICAL FILE does not clearly mention a field, KEEP the original text exactly as it appears in the TEMPLATE FRAGMENT.
- For fields that can contain multiple items (e.g., lists of medications, diagnoses, treatments), write ALL items in a SINGLE LINE after the colon, separated by commas.
- Keep exactly the same labels, numbering and order as in the TEMPLATE FRAGMENT.
- Output ONLY the rewritten TEMPLATE FRAGMENT, nothing else before or after.
- You MUST check EVERY field in the TEMPLATE FRAGMENT against the MEDICAL FILE, even if some fields seem unrelated to the most obvious findings.
- Do NOT focus only on the last field or the most prominent diagnosis.
"""

    processed_history = []

    # 3. Iterar sobre cada chunk (Procesamiento segmentado)
    for i, chunk in enumerate(chunks):
        print(f"Procesando bloque {i+1} de {len(chunks)}...")
        
        messages = [
            {"role": "system", "content": system_prompt_extractor},
            {"role": "user", "content": f"""MEDICAL FILE:
<<<
{full_medical_file}
>>>

TEMPLATE FRAGMENT:
<<<
{chunk}
>>>
Return ONLY the TEMPLATE FRAGMENT with updated values after the colon."""
}
        ]

        # Llamada al modelo para este fragmento específico
        # El modelo tiene el contexto completo pero la tarea es pequeña
        filled_fragment = call_medgemma(messages)

        filtered_fragment = filter_output(chunk,filled_fragment)
        
        # Limpieza básica para evitar duplicados de etiquetas si el modelo las repite
        processed_history.append(filtered_fragment.strip())

    # 4. Reconstrucción final
    # Unimos todo con el doble salto de línea para mantener la estructura original
    # return "\n\n".join(processed_history)
    return processed_history



def process_image_file(filepath, chunks):
    """
    1. Analiza una imagen con MedGemma para generar un reporte textual.
    2. Guarda el reporte con el formato de nombre: YYYYMMDD-auto-report-nombre.txt
    3. Llama a process_text_file para integrar la información en el template por chunks.
    """
    # 1. Preparar y llamar a MedGemma para el análisis de imagen
    # Se asume que prepare_image_message ya gestiona la conversión de la imagen
    img_report_messages = prepare_image_message(filepath)
    image_report = call_medgemma(img_report_messages)
    
    # 2. Gestión de nombres de archivo y fechas
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]
    patient_folder = os.path.dirname(filepath)
    
    if name_no_ext[:8].isdigit():
        date_prefix = name_no_ext[:8]
    else:
        date_prefix = datetime.now().strftime("%Y%m%d")
        
    new_report_name = f"{date_prefix}-auto-report-{name_no_ext}.txt"
    new_report_path = os.path.join(patient_folder, new_report_name)
    
    with open(new_report_path, "w", encoding="utf-8") as f:
        f.write(image_report)
    
    print(f"Reporte de imagen generado en: {new_report_name}")
    
    return process_text_file(new_report_path, chunks)


def filter_output(chunk: str, model_output: str) -> str:
    """
    Usa chunk como plantilla rígida.
    - Respeta títulos y etiquetas exactas del chunk.
    - Para líneas con "clave: valor", intenta sustituir solo el valor (lo que va tras ":").
    - Ignora cualquier etiqueta nueva que el modelo invente.
    """
    # Normaliza la salida del modelo, quita ``` y markdown básico
    text = (
        model_output
        .replace("```json", "")
        .replace("```", "")
        .replace("**", "")
        .strip()
    )
    resp_lines = [l.rstrip() for l in text.splitlines() if l.strip()]

    # Construimos un diccionario clave -> línea completa propuesta por el modelo
    # clave = parte antes de ":"
    resp_dict = {}
    for l in resp_lines:
        if ":" in l:
            key = l.split(":", 1)[0].strip()
            resp_dict[key] = l

    filtered_lines = []
    for orig_line in chunk.splitlines():
        line = orig_line.rstrip("\n")

        # Línea vacía: se respeta
        if not line.strip():
            filtered_lines.append(line)
            continue

        # Si no tiene ":", es título / encabezado / numeración → la dejamos igual
        if ":" not in line:
            filtered_lines.append(line)
            continue

        # Línea tipo "clave: valor"
        key, orig_value = line.split(":", 1)
        key_stripped = key.strip()

        if key_stripped in resp_dict:
            # Tenemos una propuesta del modelo para esta clave
            model_line = resp_dict[key_stripped]
            # Nos quedamos solo con lo que haya tras ":" en la línea propuesta
            _, model_value = model_line.split(":", 1)
            new_line = f"{key}: {model_value.strip()}"
            filtered_lines.append(new_line)
        else:
            # El modelo no ha propuesto nada para esta clave → conservamos el original
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def complete_template(patient_folder : str, template_path: str) -> str:

    if not os.path.exists(template_path):
        raise ValueError("Template is not found")
    
    if not os.path.exists(patient_folder):
        raise ValueError("Patient folder is not accesible")
    
    template = ""
    try:
        chunks = preprocess_template(template_path)
    except Exception as e:
        raise e
    
    list_patient_files = [f"{patient_folder}/{file}" for file in sorted(os.listdir(patient_folder))]

    for filepath in list_patient_files:

        extension = extract_extension(filepath)

        match(extension.lower()):
            case "txt" | "md" | "json" | "csv":
                # template = process_text_file(filepath,chunks)
                chunks = process_text_file(filepath,chunks)

            case "jpg" | "jpeg" | "png" | "tiff":
                # template = process_image_file(filepath,chunks)
                chunks = process_image_file(filepath,chunks)

        # chunks = [chunk.strip() for chunk in template.split("\n\n") if len(chunk.strip()) > 0]


    template = "\n\n".join(chunks)
    write_updated_template(template, patient_folder)
    return template
        
        




if __name__ == "__main__":
    complete_template("./patient_data/Beth Castro","./system_data/summary_template.txt")
    complete_template("./patient_data/Dean Espinosa","./system_data/summary_template.txt")
    complete_template("./patient_data/Fiona Graham","./system_data/summary_template.txt")