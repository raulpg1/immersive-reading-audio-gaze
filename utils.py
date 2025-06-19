import os
import sys
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
import re
import json
import torch
import scipy
import subprocess
import google.generativeai as genai
from diffusers import AudioLDMPipeline

from dotenv import load_dotenv
load_dotenv(".env")

MODEL_NAME = os.getenv("MODEL_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AUDIO_OUTPUT_PATH = os.getenv("AUDIO_OUTPUT_PATH")
AUDIO_LDM_MODEL = os.getenv("AUDIO_LDM_MODEL")

def gemini_api_llm(user_prompt: str, retries: int = 3) -> dict | None:
    """
    Sends a prompt to the Gemini API and returns the parsed JSON response.

    Args:
        user_prompt (str): Prompt to send to Gemini.
        retries (int): Number of retry attempts on failure.

    Returns:
        dict | None: Parsed response or None on failure.
    """
    
    genai.configure(api_key=GOOGLE_API_KEY)

    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"[ERROR] Error al cargar el modelo '{MODEL_NAME}': {e}")
        return None

    try:
        response = model.generate_content(user_prompt)
        content = response.text
        # print(content,"**"*100)
        match = re.search(r"```(?:json)?\n(.*?)```", content, re.DOTALL)
        json_str = match.group(1) if match else content
        return json.loads(json_str.replace('```json\n', '').replace('\n', '').replace('`', ''))
    except Exception as e:
        print(f"[ERROR] Fallo al generar/parsing respuesta: {e}")
        if retries > 0:
            print("[INFO] Reintentando...")
            return gemini_api_llm(user_prompt, retries - 1)
        return None
    
def generate_audioldm(prompt_struct):
    pipe = AudioLDMPipeline.from_pretrained(AUDIO_LDM_MODEL, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    linea_act = 0
    for i in prompt_struct.keys():
        for j in range(len(prompt_struct[i])-1):
            prompt = prompt_struct[i][j]["prompt"]
            linea = prompt_struct[i][j]['linea']
            length = prompt_struct[i][j]["length"]
            print(f"[AUDIOLDM] Generando audio del parrafo {i}: linea: {linea_act+linea} Prompt: {prompt}")
            audio = pipe(prompt, num_inference_steps=100, audio_length_in_s=length).audios[0]
            audio_output_file_path = os.path.join(AUDIO_OUTPUT_PATH,f"{linea_act+linea}.wav")
            scipy.io.wavfile.write(audio_output_file_path, rate=16000, data=audio)
        linea_act += prompt_struct[i][-1]

def read_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as archivo:
            file = archivo.read()
    except FileNotFoundError:
        print("Error: No se encontró el archivo {file_path}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    return file

def play_audio_with_xdg_open(num_audio):
    """
    Reproduce un archivo de audio usando xdg-open (abrir con la aplicación predeterminada).
    No bloquea la ejecución del script Python.
    """
    file_path = os.path.join(AUDIO_OUTPUT_PATH,f"{num_audio}.wav")
    if not os.path.exists(file_path):
        # print(f"Error: El archivo '{file_path}' no se encontró.")
        return
    try:
        # print(f"Intentando reproducir '{file_path}' con xdg-open...")
        subprocess.Popen(['xdg-open', file_path])
    except FileNotFoundError:
        print("Error: 'xdg-open' no se encontró. Asegúrate de que tu sistema Linux tenga un entorno de escritorio.")
    except Exception as e:
        print(f"Ocurrió un error al intentar abrir el audio: {e}")