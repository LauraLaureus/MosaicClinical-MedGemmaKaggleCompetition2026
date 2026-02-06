# %%
import asyncio
from typing import Optional
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

def agent_generation(messages: list[dict]) -> tuple[Optional[str],Optional[str]]:
    """
    Ask the agent model to generate
    
    :param messages: list of all messages to send to the agent brain in OpenAI format. 
    :type messages: list[Dict]
    :return: think channel(string), output channel(string)
    :rtype: string
    """
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


    print(f"\nBrain generated: {full_generation}", flush=True)


    think_channel = None
    response_channel = None

    if "</think>" in full_generation:
        channels = full_generation.split("</think>")
        think_channel = channels[0].replace("<think>","")
        response_channel = channels[-1]

    else:
        response_channel = full_generation

    return think_channel, response_channel.strip()

def parse_tool_call(response_channel: str) -> tuple[Optional[str], Optional[dict]]:
    tool_name = None
    tool_args = None

    # 1. Extraer lo que hay ENTRE los tags (o desde el tag de apertura hasta el final)
    # El regex busca <tool_call>, captura todo lo que sigue, y para en </tool_call> si existe
    match = re.search(r"<tool_call>(.*?)(?:</tool_call>|$)", response_channel, re.DOTALL)

    if match:
        raw_content = match.group(1).strip()
        
        # 2. Reparación de emergencia para la "Patata"
        # Si el modelo cortó el JSON antes de tiempo:
        if raw_content.count('{') > raw_content.count('}'):
            raw_content += "}"
        if raw_content.count('[') > raw_content.count(']'):
            raw_content += "]"

            
            
        raw_content = raw_content.replace("<tool_call>","")
        raw_content = raw_content.replace("</tool_call>", "")            


        try:
            # 3. Intentar parsear el JSON limpio
            tool_data = json.loads(raw_content)
            tool_name = tool_data.get("name")
            tool_args = tool_data.get("arguments", {})
        except json.JSONDecodeError as e:
            print(f"⚠️ Error de formato JSON: {e} en el contenido: {raw_content}")
            # Aquí podrías intentar una limpieza manual más agresiva si fuera necesario
    
    return tool_name, tool_args
        
def get_next_todo_point(current_plan:str) -> str:

    tasks = current_plan.splitlines()
    for task in tasks:
        if task.startswith("TODO"):
            return task
    return None

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


                async def call_tool(tool_name, tool_args):
                    try:
                        print(f"⚙️ Enviando petición a la herramienta: {tool_name}...", flush=True)
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        print("✅ Respuesta recibida del servidor.", flush=True)

                        return result
                    except anyio.ClosedResourceError as e :
                        print("❌ ERROR: MCP server closed abruptly.")

                        fnull.write(str(e))


                summary_template = ""
                with open("./data/summary_template.txt","r",encoding="utf-8") as f:
                    summary_template = f.read()

                patient_folder = "./data/Master First"

                INITIAL_PLAN_SYSTEM_PROMPT = f"""
You are a medical agent with tools to access file system.

TASK:
The user needs to fill the MODEL CLINICAL HISTORY SUMMARY form with medical data. 
The medical data is available in the patient folder.
The user will provide the patient's folder.
You can use the following tools to access the filesystem:
{tools_formatted_list}


Prepare a plan following the following format. 

PLAN FORMAT:
TODO - <step number> - <list idea> 
TODO - 1 - List files in ./data/<patient_name>
TODO - 2 - Read patient_report.txt
TODO - 3 - Use MedGemma to fill summary
TODO - 4 - Save final result to summary.txt

{summary_template}

Your current task is writting the plan using the PLAN FORMAT, without introductions or explanations. 
Write only the tasks you are 100% sure you can complete right now. You will be able to update the plan later. 
""" 
                print(f"System prompt: {INITIAL_PLAN_SYSTEM_PROMPT}")

                messages = [
                {"role": "system", "content": INITIAL_PLAN_SYSTEM_PROMPT},
                {"role": "user", "content": f"Patient folder'{patient_folder}'"}
            ]

                think_channel, output_channel = agent_generation(messages)
                output_channel = output_channel.strip()

                blackboard = {"planning_step": patient_folder,
                              "summary_template_filepath": "./data/summary_template.txt"}


                _ = await call_tool("write_file",{"filepath":"./to-do.txt","content":output_channel.strip()})
                _ = await call_tool("write_file",{"filepath":"./blackboard.txt", "content":json.dumps(blackboard)})

                current_plan = output_channel

### region execute plan loop
                CURRENT_STEP_EXECUTION_SYSTEM_PROMPT = f"""You are a helpful agent that can use the following tools:
### TOOLS
{tools_formatted_list}

### CURRENT DATA
{json.dumps(blackboard)}
"""
                next_step = get_next_todo_point(current_plan)
                if next_step:

                    CURRENT_STEP_USER_PROMPT =f"""
The current step is '{next_step}'
Provide a tool call request in the format <tool_call>{{"name": "tool_name", "arguments": {{"arg_name": "value"}} }}</tool_call>

RULES:
- ALWAYS respect the format.
- ALWAYS start with the "<tool_call>" tag. 
- ALWAYS provide a Valid JSON format. Check that every curly parenthesis has its closing one. Use only double quote symbol (").
- ALWAYS end with the "</tool_call>" tag.  
"""

                    messages = [    
                        {"role":"system", "content":CURRENT_STEP_EXECUTION_SYSTEM_PROMPT},
                        {"role":"user", "content":CURRENT_STEP_USER_PROMPT}
                    ]
                    think_channel, output_channel = agent_generation(messages)
                    print(f"AGENT Think channel: {think_channel}")
                    print("-"*50)
                    print(f"AGENT GENERATION: {output_channel}")
                    output_channel = output_channel.strip()

                    if "FINISH" not in output_channel.upper():
                        tool_name, tool_args = parse_tool_call(output_channel)

                        if tool_name:
                            tool_results = await call_tool(tool_name=tool_name, tool_args=tool_args)
                            if tool_results:
                                blackboard["tool_result"] = tool_results.content[0].text

                    print(f"blackboard: {json.dumps(blackboard)}")

                    


                    


if __name__ == "__main__":

    asyncio.run(run_agent())

    # await run_agent()
