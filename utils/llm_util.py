import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import numpy as np
import openai
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from utils.io_util import load_json

api_keys_folder = Path(__file__).resolve().parent.parent / "api_keys"
OPENAI_KEYS = load_json(api_keys_folder / "openai_keys.json")
GROQ_KEYS = load_json(api_keys_folder / "groq_keys.json")

class LLMBase:
    BACKENDS = {
        "openai": dict(base_url=None,
                       api_key=lambda: OPENAI_KEYS["key"],
                       model="gpt-3.5-turbo"),
        "groq":   dict(base_url="https://api.groq.com/openai/v1",
                       api_key=lambda: GROQ_KEYS["key"],
                       model="deepseek-r1-distill-llama-70b"),
        "ollama": dict(base_url="http://localhost:11434/v1",
                       api_key="ollama",
                       model="gemma3:27b"),
    }

    def __init__(self, backend="openai"):
        self.backend = backend
        cfg = self.BACKENDS[backend]
        self.model_name = cfg["model"]
        api_key = cfg["api_key"]
        self.client = openai.OpenAI(
            api_key=api_key() if callable(api_key) else api_key,
            base_url=cfg["base_url"],
        )

    def prompt_llm(self, prompt: str):        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content