# %%
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import sys
import re
import anyio
import time

torch.set_num_threads(min(4, os.cpu_count()))
device = "cpu"

# 1. Configuración del Modelo (Cerebro)
# model_id = "HuggingFaceTB/SmolLM3-3B"
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.float32, 
    device_map={"": device},
    low_cpu_mem_usage=True,
    attn_implementation="sdpa" if sys.platform != "win32" else "eager"
)

server_params = StdioServerParameters(
    command=sys.executable,
    args=["mcp_server.py"], 
)


async def run_agent():
    with open("mcp_debug.log", 'w') as fnull:
        async with stdio_client(server_params, errlog=fnull) as (read, write):
            async with ClientSession(read, write) as session:
                session.default_timeout = 300 # 5 minutes
                # Inicializar conexión con el servidor MCP
                await session.initialize()
                
                # List MCP tools
                tools = await session.list_tools()
                tools_description = []
                for t in tools.tools:
                    # t.inputSchema contiene el JSON Schema de los argumentos
                    schema_info = json.dumps(t.inputSchema.get("properties", {}), indent=2)
                    tools_description.append(f"- {t.name}: {t.description}\n  Arguments schema: {schema_info}")

                tools_formatted_list = "\n".join(tools_description)


                summary_template = ""
                with open("./data/summary_template.txt","r",encoding="utf-8") as f:
                    summary_template = f.read()

                patient_folder = "./data/Master First"

                SYSTEM_PROMPT = f"""You are a helpful medical assistant.

To help you with this task you have access to the following tools which will DO grant you access to the file system:\n{tools_formatted_list}
Remind that medgemma is a medical model that can help you with medical functions. 

When you need to use a tool, provide a tool request.
Tool Request are in the following format "<tool_call>{{"name": "tool_name", "arguments": {{}} }}</tool_call>"
The tool request should be the last thing in the message. 
Use one tool at the time. 

Attend to the user's request and make a plan.
Write the plan as a checkboxes list for example [ ] - <STEP Number> - Task
As first step ALWAYS write the plan into './to-do.txt' file.

""" 

                messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Fill the MODEL CLINICAL HISTORY SUMMARY with the information of the patient that is available in '{patient_folder}'\n{summary_template}"}
            ]


                for step in range(1):
                    
                    # print(f"\n--- Thinking (Paso {step+1}) ---", flush=True)

                    # --- Bucle de Razonamiento del Agente ---

                    prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )

                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        outputs = model.generate(**inputs,
                                                max_new_tokens=1024,
                                                temperature=0.01,
                                                do_sample=False,
                                                repetition_penalty=1.1,
                                                pad_token_id=tokenizer.eos_token_id,
                                                eos_token_id=tokenizer.eos_token_id
                                                )
                        
                    end_time = time.perf_counter()
                    duration = end_time - start_time

                    full_generation = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                    new_tokens_count = len(outputs[0]) - inputs.input_ids.shape[-1]
                    tokens_per_sec = new_tokens_count / duration if duration > 0 else 0

                    print(f"\n⏱️  Generation time: {duration:.2f} s")
                    print(f"🚀 Speed: {tokens_per_sec:.2f} t/s")
                    print(f"📊 Number of generated tokens: {new_tokens_count}")


                    print(f"\nSmolLM3 generated: {full_generation}", flush=True)


                    if "</think>" in full_generation:
                        channels = full_generation.split("</think>")
                        think_channel = channels[0].replace("<think>","")
                        response_channel = channels[-1]

                    else:
                        response_channel = full_generation


                    match = re.search(r"<tool_call>(.*?)</tool_call>", response_channel, re.DOTALL)

                    if match:

                        # torch.set_num_threads(1) 

                        tool_data = json.loads(match.group(1))
                        tool_name = tool_data["name"]
                        tool_args = tool_data.get("arguments", {})

                        print(f"--- Ejecutando herramienta: {tool_name} ---")
                        
                        try:
                            print(f"⚙️ Enviando petición a la herramienta: {tool_name}...", flush=True)
                            result = await session.call_tool(tool_name, arguments=tool_args)
                            print("✅ Respuesta recibida del servidor.", flush=True)
                        except anyio.ClosedResourceError:
                            print("❌ ERROR: El servidor MCP se cerró inesperadamente.")
                            # Mira el log para saber por qué
                            if os.path.exists("mcp_debug.log"):
                                with open("mcp_debug.log", "r") as f:
                                    print(f"LOG DEL SERVIDOR: {f.read()}")
                            break
                        
                        # Añadimos el resultado al historial para que el modelo lo vea
                        messages.append({"role": "assistant", "content": full_generation})
                        messages.append({"role": "user", "content": f"Tool result: {result.content}"})
                    else:
                        break


if __name__ == "__main__":

    asyncio.run(run_agent())

    # await run_agent()

