import os
import requests
import argparse
import json
import re

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

    summary_filepath = os.path.join(patient_folder,"summary.txt")

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
    
    list_patient_files = [f"{patient_folder}/{file}" for file in os.listdir(patient_folder)]

    for filepath in list_patient_files:

        extension = extract_extension(filepath)

        system_prompt = """Act like an expert clinician specializing in clinical reasoning and data synthesis.
Your goal is to update a medical template using information from a new file.

### CORE OPERATING RULES:
1. THINK STEP-BY-STEP: Analyze each section of the template against the file content.
2. PERSISTENCE: If the file has NO information for a section, you MUST keep the exact original text from the template. DO NOT delete or summarize existing data.
3. INTEGRATION: If new data is found, merge it logically with the existing text.

### CLINICAL INFERENCE RULES:
- DO NOT be strictly literal. If the clinical data describes severe conditions (e.g., severe cognitive delay, inability to speak, or multiple daily seizures), INFER the functional impact for the "Baseline Functional Status" section.
- Use medical judgment: Instead of "Not specified", use "Likely dependent/limited due to [Specific Condition Found]".
- Connect the dots: Use neurological or psychomotor delays to determine the level of independence.

### MEDICATION & DATA INTEGRITY RULES:
- STRICT EVIDENCE: DO NOT list any medication unless it is explicitly named in the file or is already present in the template. 
- NO PROBABILISTIC GUESSING: Even if a treatment is standard for a condition (e.g., Levetiracetam for Epilepsy), if it is not in the text, DO NOT add it.
- PLACEHOLDERS: If no medications are mentioned in the new file and the template is empty, write "No medication listed in the analyzed records".
- AUDIT TRAIL: If you add a medication, ensure it comes from the text or is already present in the template.

### OUTPUT FORMAT:
- Output ONLY the updated template. 
- NO markdown (no ```), NO code blocks, NO introductions, NO "Here is the update".
- Start directly with the first section of the template.
"""

        messages = [{"role": "system", "content": system_prompt}]

        match(extension):
            case "txt" | "md" | "json" | "csv":
                messages.append(prepare_text_message(filepath,template))
            case "jpg" | "jpeg" | "png" | "tiff":
                pass
            case "dicom":
                pass

        template = call_medgemma(messages)

    write_updated_template(template, patient_folder)
        
        




if __name__ == "__main__":
    complete_template("./data/Master First","./data/summary_template.txt")