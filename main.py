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

        system_prompt = """Analyze the content of the provided file and the provided template.

Update the template with the information of the file. 

Everytime rewrite ONLY the template with the new information if any."""

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