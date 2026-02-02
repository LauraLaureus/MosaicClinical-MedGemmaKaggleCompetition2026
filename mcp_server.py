from mcp.server.fastmcp import FastMCP
import os

# Creamos el servidor MCP
mcp = FastMCP("MedicalFileManager")

# Carpeta de trabajo para los registros médicos
BASE_DIR = "./pacientes_data"
os.makedirs(BASE_DIR, exist_ok=True)

@mcp.tool()
def list_medical_files() -> list:
    """Lista todos los documentos en la carpeta de datos médicos."""
    return os.listdir(BASE_DIR)

@mcp.tool()
def read_medical_report(filename: str) -> str:
    """Lee el contenido de un informe médico específico."""
    path = os.path.join(BASE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@mcp.tool()
def save_analysis(filename: str, content: str):
    """Guarda el análisis final o formulario en un fichero nuevo."""
    path = os.path.join(BASE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Fichero {filename} guardado con éxito."

@mcp.tool()
def ask_medgemma_to_fill_form(report_text: str, image_path: str = None) -> str:
    """
    Envía la información a MedGemma para rellenar un formulario clínico.
    Nota: Esta función actúa como puente hacia tu servidor de LM Studio.
    """
    # Aquí iría la llamada a la API local de LM Studio que configuraremos después
    return f"SIMULACIÓN: MedGemma analizando el reporte: {report_text[:30]}..."

if __name__ == "__main__":
    mcp.run()