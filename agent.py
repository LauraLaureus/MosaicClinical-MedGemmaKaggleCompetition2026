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

torch.set_num_threads(min(4, os.cpu_count()))
device = "cpu"

# 1. Configuración del Modelo (Cerebro)
model_id = "HuggingFaceTB/SmolLM3-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=torch.float32, 
    device_map={"": device},
    low_cpu_mem_usage=True,
    attn_implementation="sdpa" if sys.platform != "win32" else "eager"
)

# 2. Configuración del Servidor MCP (Manos)
server_params = StdioServerParameters(
    command=sys.executable,
    args=["mcp_server.py"], 
)

async def run_agent():
    with open("mcp_debug.log", 'w') as fnull:
        async with stdio_client(server_params, errlog=fnull) as (read, write):
            async with ClientSession(read, write) as session:
                session.default_timeout = 300 # 5 minutos
                # Inicializar conexión con el servidor MCP
                await session.initialize()
                
                # Listar herramientas disponibles en el servidor
                tools = await session.list_tools()
                print(f"Detected tools: {[t.name for t in tools.tools]}", flush=True)

                tools_description = []

                for t in tools.tools:
                    # t.inputSchema contiene el JSON Schema de los argumentos
                    schema_info = json.dumps(t.inputSchema.get("properties", {}), indent=2)
                    tools_description.append(f"- {t.name}: {t.description}\n  Arguments schema: {schema_info}")

                tools_formatted_list = "\n".join(tools_description)

                SYSTEM_PROMPT = f"""You are a medical assistant. You have access to the following tools {tools_formatted_list}

Attend to the user's request and make a plan. You will execute one step at the time. 
When you need to use a tool, provide a tool request.
Tool Request are in the following format "<tool_call>{{"name": "tool_name", "arguments": {{}} }}</tool_call>"
The tool request should be the last thing in the message. 
Use one tool at the time. 
""" 

                messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "List the files and print the list."}
            ]


                for step in range(3):
                    
                    print(f"\n--- Pensando (Paso {step+1}) ---", flush=True)

                    # --- Bucle de Razonamiento del Agente ---

                    prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )

                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs,
                                                max_new_tokens=512,
                                                temperature=0.01,
                                                do_sample=False,
                                                repetition_penalty=1.1,
                                                pad_token_id=tokenizer.eos_token_id,
                                                eos_token_id=tokenizer.eos_token_id
                                                )
                    full_generation = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

                    print(f"\nSmolLM3 generated: {full_generation}", flush=True)


                    if "</think>" in full_generation:
                        channels = full_generation.split("</think>")
                        think_channel = channels[0].replace("<think>","")
                        response_channel = channels[-1]

                    else:
                        response_channel = full_generation


                    match = re.search(r"<tool_call>(.*?)</tool_call>", response_channel, re.DOTALL)

                    if match:

                        torch.set_num_threads(1) 

                        tool_data = json.loads(match.group(1))
                        tool_name = tool_data["name"]
                        tool_args = tool_data.get("arguments", {})

                        print(f"--- Ejecutando herramienta: {tool_name} ---")
                        
                        try:
                            print(f"⚙️ Enviando petición a la herramienta: {tool_name}...", flush=True)
                            # Algunos servidores necesitan un pequeño respiro si la CPU está al 100%
                            await asyncio.sleep(0.5) 
                            
                            result = await session.call_tool(tool_name, arguments=tool_args)
                            torch.set_num_threads(4)
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


