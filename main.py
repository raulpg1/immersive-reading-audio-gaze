import os
import json
from gaze_tracking import run_gaze_tracking
from utils import generate_audioldm, read_text
from chains.audio_chain import create_audio_chain
from config import GOOGLE_API_KEY, DATA_DIR

def main():
    prompt_file_name = "audio_prompt_encoder.txt"
    chain = create_audio_chain(api_key=GOOGLE_API_KEY, prompt_path=os.path.join(DATA_DIR,prompt_file_name))

    page_file_name = "full_page.txt"
    page_file_path = read_text(os.path.join(DATA_DIR,page_file_name))
    
    procesado = {}
    for i,parrafo in enumerate(page_file_path.split("\n\n")):
        
        total_lines_paragraph = len([ x for x in parrafo.split("\n") if x != ""])
        procesado[i+1] = json.loads(chain.invoke({"paragraph": parrafo}))
        procesado[i+1].append(total_lines_paragraph)

    generate_audioldm(procesado)
    
    run_gaze_tracking(page_file_name)

if __name__ == "__main__":
    main()