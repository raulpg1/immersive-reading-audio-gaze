from utils import read_text
from models.gemini_llm import GeminiLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

def create_audio_chain(api_key: str, prompt_path: str) -> Runnable:
    base_prompt = read_text(prompt_path)
    full_prompt = base_prompt + "\n{paragraph}"
    prompt = PromptTemplate.from_template(full_prompt)
    llm = GeminiLLM(api_key=api_key)
    return prompt | llm