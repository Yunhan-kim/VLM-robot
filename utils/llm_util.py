import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from utils.io_util import load_json
import google.generativeai as genai
import openai

api_keys_folder = Path(__file__).resolve().parent.parent / "api_keys"
GEMINI_KEYS = load_json(api_keys_folder / "gemini_keys.json")
OPENAI_KEYS = load_json(api_keys_folder / "openai_keys.json")
GROQ_KEYS = load_json(api_keys_folder / "groq_keys.json")

# # GEMINI
# class LLMBase:
#     def __init__(self):
#         genai.configure(api_key = GEMINI_KEYS["key"])
#         generation_config = genai.GenerationConfig(temperature=2)
#         self.model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

#     def prompt_llm(self, prompt: str):
#         # logger.info("\n" + "#" * 50)
#         # logger.info(f"Prompt:\n{prompt}")       

#         llm_output = self.model.generate_content(prompt).text        

#         # logger.info("\n" + "#" * 50)
#         # logger.info(f"LLM output:\n{llm_output}")

#         return llm_output

# # OpenAI
# class LLMBase:
#     def __init__(self):        
#         self.client = openai.OpenAI(api_key=OPENAI_KEYS["key"])         
#         self.model_name = "gpt-3.5-turbo"        

#     def prompt_llm(self, prompt: str):
#         response = self.client.chat.completions.create(
#             model = self.model_name,
#             messages = [{"role": "user", "content": prompt}],
#             temperature = 0.7
#         )
#         return response.choices[0].message.content

# Groq
class LLMBase:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key = GROQ_KEYS["key"],
            base_url = "https://api.groq.com/openai/v1"
        )
        self.model_name = "llama-3.3-70b-versatile"

    def prompt_llm(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content