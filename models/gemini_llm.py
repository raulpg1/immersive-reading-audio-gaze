import re
import json
import google.generativeai as genai
from config import MODEL_NAME
from langchain_core.language_models import LLM


class GeminiLLM(LLM):
    model_name: str = MODEL_NAME
    api_key: str = None

    def _call(self, prompt: str, stop=None) -> str:
        genai.configure(api_key=self.api_key)
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            content = response.text
            match = re.search(r"```(?:json)?\n(.*?)```", content, re.DOTALL)
            json_str = match.group(1) if match else content
            parsed = json.loads(json_str.replace('\n', '').replace('`', ''))
            return json.dumps(parsed, indent=2)
        except Exception as e:
            print(f"[ERROR] {e}")
            return "Error al procesar la solicitud."

    @property
    def _llm_type(self) -> str:
        return "custom-gemini"