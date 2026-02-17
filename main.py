import gradio as gr
import core
import os
import tempfile

def complete_template(paciente, template):

    patient_folder = os.path.join("./patient_data/",paciente)
    
    template_path = os.path.join(tempfile.gettempdir(),"template.txt")
    with open(template_path,"w",encoding="utf-8") as f:
        f.write(template)

    updated_template = core.complete_template(patient_folder=patient_folder,template_path=template_path)
    return updated_template

with gr.Blocks(title="MedGemma Clinician Assistant") as demo:
    gr.Markdown("# 🏥 Dashboard for Dr. Anna Doe")
    gr.Markdown("Summarize clinical history using reasoning over records.")
    
    # --- ZONA SUPERIOR: ENTRADA ---
    with gr.Row():
        # Columna 1: Agenda
        with gr.Column(scale=1):
            gr.Markdown("### 📅 Agenda")
            
            if not os.path.exists("./patient_data"):
                raise ValueError("Patient Folder not found.")
            
            patients = os.listdir("./patient_data")
            lista_pacientes = gr.Radio(
                patients, 
                label="Pick patient (Folder)",
                value=patients[0] if len(patients) > 0 else None,
                interactive=True
            )
            gr.Info("Select the patient to generate the summary.")

        # Columna 2: Custom Template
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Custom Template")
            template_input = gr.Textbox(
                label="Edit your summary template here",
                placeholder="1) Demographic Data...",
                lines=10,
                value="MODEL CLINICAL HISTORY SUMMARY\n\n1) Demographic Data..." 
            )
            btn_submit = gr.Button("🚀 Generate", variant="primary")

    gr.Markdown("---")

    # --- ZONA INFERIOR: OUTPUT ---
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Output: Updated template")
            output_text = gr.Textbox(
                label="Output",
                lines=15,
                # show_copy_button=True # Muy útil para el doctor
            )

    # Lógica del botón
    btn_submit.click(
        fn=complete_template, 
        inputs=[lista_pacientes, template_input], 
        outputs=output_text
    )

# Lanzar la app
if __name__ == "__main__":
    demo.launch()