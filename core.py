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

    return {"role": "user", "content": f"""### TEMPLATE 
{template}


### FILE
{file_content}""" }


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
        "temperature": 0.0, # Baja para que sea preciso con los datos médicos
        "max_tokens": -1    # -1 permite que el modelo use lo que necesite
    }

    try:
        response = requests.post(lmstudio_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Lanza error si la petición falla
        
        result = response.json()
        full_content = result['choices'][0]['message']['content']
        clean_content = re.sub(r'<unused94>.*?<unused95>', '', full_content, flags=re.DOTALL).strip()
        return clean_content
    
    except Exception as e:
       raise e


def write_updated_template(updated_template:str, patient_folder: str):

    current_date = datetime.now().strftime("%Y%m%d")

    summary_filepath = os.path.join(patient_folder,f"{current_date}_summary.txt")

    with open(summary_filepath,"w") as f:
        f.write(updated_template)


def complete_template(patient_folder : str, template_path: str) -> str:

    if not os.path.exists(template_path):
        raise ValueError("Template is not found")
    
    if not os.path.exists(patient_folder):
        raise ValueError("Patient folder is not accesible")
    
    template = ""
    try:
        with open(template_path, "r",encoding="utf-8") as f:
            template = f.read()
    except Exception as e:
        raise e
    
    list_patient_files = [f"{patient_folder}/{file}" for file in sorted(os.listdir(patient_folder))]

    for filepath in list_patient_files:

        extension = extract_extension(filepath)

        system_prompt = """Act like an expert clinical assistant. Your goal is to synthesize medical records into a structured summary.

### MISSION:
You will receive a MEDICAL FILE and a TEMPLATE. You must analyze the file and populate the template, merging new findings with existing data.

### ACTION RULES:
1. DATA HARVESTING: Read every line of the FILE. Extract all available information requested in the templated. If a piece of data fits a field, INSERT IT.
2. CHRONOLOGICAL OVERWRITE: Treat the FILE as the latest truth. Update ages, dosages, or statuses if they differ from the provided TEMPLATE.
3. PRESERVATION: Never delete existing data from the template if the new record is silent about that section. Only replace "Not specified" with real data.
4. CLINICAL INFERENCE (MANDATORY): 
   - If the diagnosis is a severe neurodevelopmental disorder (e.g., Dravet, encephalopathy), do NOT leave Functional Status as "Not specified". 
   - Infer: "Dependent for ADLs" and "Lives with family/caregivers".
5. NO HALLUCINATION: If a field has no data in the template AND no mention in the record, use "Not specified".

### FORMAT RULES:
- Output ONLY the populated text. 
- NO JSON, NO coordinates, NO markdown code blocks (```), NO "Here is the summary".
- Use bullet points for lists (medications, conditions).
- Ensure every field ends with ': ' followed by the data.

"""

        messages = [{"role": "system", "content": system_prompt}]

        match(extension.lower()):
            case "txt" | "md" | "json" | "csv":
                messages.append(prepare_text_message(filepath,template))
            case "jpg" | "jpeg" | "png" | "tiff":
                img_report_messages = prepare_image_message(filepath)
                image_report = call_medgemma(img_report_messages)
                filename = os.path.basename(filepath)
                name_no_ext = os.path.splitext(filename)[0]
                date_prefix = name_no_ext[:8] if name_no_ext[:8].isdigit() else datetime.now().strftime("%Y%m%d")
                new_report_name = f"{date_prefix}_auto-report_{name_no_ext}.txt"
                new_report_path = os.path.join(patient_folder, new_report_name)
                with open(new_report_path, "w", encoding="utf-8") as f:
                    f.write(image_report)
                messages.append(prepare_text_message(new_report_path, template))
            case "dicom":
                pass

        template = call_medgemma(messages)

    write_updated_template(template, patient_folder)
        
        




if __name__ == "__main__":
    # complete_template("./patient_data/Beth Castro","./system_data/summary_template.txt")
    complete_template("./patient_data/Fiona Graham","./system_data/summary_template.txt")