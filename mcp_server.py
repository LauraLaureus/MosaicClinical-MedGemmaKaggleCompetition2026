from mcp.server.fastmcp import FastMCP
import os
import httpx

# Creamos el servidor MCP
mcp = FastMCP("MedicalFileManager")


@mcp.tool()
async def list_medical_files(folder_path: str) -> str:
    """List all the files in the provided folder_path and the list is flatten into a string with newline character as separator"""
    return "\n- ".join(os.listdir(folder_path))

@mcp.tool()
async def read_file(filepath: str) -> str:
    """Read the content of the file specified in the filepath."""
    if not os.path.exists(filepath):
        return f"Error: The file at {filepath} does not exist."
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@mcp.tool()
async def write_file(filepath: str, content: str) -> str:
    """Create or override the content of the provided filepath"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Archivo {filepath} guardado con éxito."

@mcp.tool()
async def append_to_file(filepath: str, content: str) -> str:
    """Append content at the end of the provided filepath"""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n" + content)
    return f"Contenido añadido a {filepath}."

@mcp.tool()
async def call_medgemma_expert(system_prompt: str, file_content: str) -> str:
    """
    Send the content of a text file to MedGemma along with the system_prompt to perform an action. 
    """
    url = "http://localhost:1234/v1/chat/completions"
    
    payload = {
        "model": "medgemma-1.5-4b-it", # Verifica el ID exacto en tu LM Studio
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"File content:\n\n{file_content}"}
        ],
        "temperature": 0.1 # Baja temperatura para mayor precisión médica
    }
    
    # Timeout largo porque JIT puede tardar en cargar el modelo
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            return f"Error llamando a MedGemma: {str(e)}"

if __name__ == "__main__":
    mcp.run()