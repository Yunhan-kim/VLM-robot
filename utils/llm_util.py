import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from utils.io_util import load_json
import google.generativeai as genai

gemini_keys_folder = Path(__file__).resolve().parent.parent / "gemini_keys"
GEMINI_KEYS = load_json(gemini_keys_folder / "gemini_keys.json")

class LLMBase:
    def __init__(self):        

        genai.configure(api_key = GEMINI_KEYS["key"])
        generation_config = genai.GenerationConfig(temperature=2)
        self.model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    def prompt_llm(self, prompt: str):
        # logger.info("\n" + "#" * 50)
        # logger.info(f"Prompt:\n{prompt}")       

        llm_output = self.model.generate_content(prompt).text        

        # logger.info("\n" + "#" * 50)
        # logger.info(f"LLM output:\n{llm_output}")

        return llm_output