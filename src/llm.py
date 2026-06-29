import json
import requests
import re
from google import genai
from google.genai import types
from src import config

OLLAMA_TASK_MODELS = {
    "fast": "gemma3:1b",
    "validation": "gemma3:4b",
    "reasoning": "gemma3:4b",
    "vision": "moondream:latest"
}

GEMINI_TASK_MODELS = {
    "fast": "gemini-2.5-flash",
    "validation": "gemini-2.5-flash",
    "reasoning": "gemini-2.5-flash",
    "vision": "gemini-2.5-flash"
}

# NOTE: For Ollama provider, gemma3:4b handles all text tasks (fits in 4GB VRAM).
# moondream:latest handles vision-only tasks (image uploads).


def get_active_config():
    """Retrieve active LLM configurations (either from streamlit session state or config file)"""
    try:
        import streamlit as st
        if st.runtime.exists():
            provider = st.session_state.get("llm_provider", config.DEFAULT_PROVIDER)
            ollama_url = st.session_state.get("ollama_url", config.OLLAMA_BASE_URL)
            ollama_model = st.session_state.get("ollama_model", config.OLLAMA_MODEL)
            ollama_embed = config.OLLAMA_EMBED_MODEL
            gemini_key = st.session_state.get("gemini_key", config.GEMINI_API_KEY)
            gemini_model = st.session_state.get("gemini_model", config.GEMINI_MODEL)
            gemini_embed = config.GEMINI_EMBED_MODEL
            return {
                "provider": provider,
                "ollama_url": ollama_url,
                "ollama_model": ollama_model,
                "ollama_embed": ollama_embed,
                "gemini_key": gemini_key,
                "gemini_model": gemini_model,
                "gemini_embed": gemini_embed,
            }
    except Exception:
        pass
    
    return {
        "provider": config.DEFAULT_PROVIDER,
        "ollama_url": config.OLLAMA_BASE_URL,
        "ollama_model": config.OLLAMA_MODEL,
        "ollama_embed": config.OLLAMA_EMBED_MODEL,
        "gemini_key": config.GEMINI_API_KEY,
        "gemini_model": config.GEMINI_MODEL,
        "gemini_embed": config.GEMINI_EMBED_MODEL,
    }

def test_connection() -> tuple[bool, str]:
    """Test connection to the configured LLM provider. Returns (success, message)."""
    cfg = get_active_config()
    if cfg["provider"] == "gemini":
        if not cfg["gemini_key"]:
            return False, "Gemini API key is not configured."
        model_name = GEMINI_TASK_MODELS.get("fast", "gemini-2.5-flash")
        try:
            client = genai.Client(api_key=cfg["gemini_key"])
            response = client.models.generate_content(
                model=model_name,
                contents="Ping",
                config=types.GenerateContentConfig(max_output_tokens=10)
            )
            return True, f"Successfully connected to Gemini using model: {model_name}"
        except Exception as e:
            return False, f"Failed to connect to Gemini: {str(e)}"
    else:
        url = f"{cfg['ollama_url']}/api/generate"
        model_name = OLLAMA_TASK_MODELS.get("fast", "gemma2:2b")
        try:
            # We check if Ollama is running by asking for model generation
            response = requests.post(
                url,
                json={"model": model_name, "prompt": "Ping", "stream": False, "options": {"num_predict": 10}},
                timeout=45
            )
            if response.status_code == 200:
                return True, f"Successfully connected to Ollama using model: {model_name}"
            else:
                return False, f"Ollama returned status code {response.status_code}. Response: {response.text}"
        except requests.exceptions.RequestException as e:
            return False, f"Failed to connect to Ollama at {cfg['ollama_url']}: {str(e)}"

def generate_text(prompt: str, system_instruction: str = None, task: str = "fast") -> str:
    """Generate text from prompt using active provider and task routing."""
    cfg = get_active_config()
    if cfg["provider"] == "gemini":
        if not cfg["gemini_key"]:
            raise ValueError("Gemini API key is not configured.")
        client = genai.Client(api_key=cfg["gemini_key"])
        model_name = GEMINI_TASK_MODELS.get(task, "gemini-2.5-flash")
        
        gen_config = None
        if system_instruction:
            gen_config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0)
        else:
            gen_config = types.GenerateContentConfig(temperature=0.0)
            
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=gen_config
        )
        return response.text
    else:
        url = f"{cfg['ollama_url']}/api/generate"
        model_name = OLLAMA_TASK_MODELS.get(task, "gemma2:2b")
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        if system_instruction:
            payload["system"] = system_instruction
            
        try:
            response = requests.post(url, json=payload, timeout=90)
            if response.status_code == 404:
                raise ValueError(f"Model '{model_name}' was not found in your Ollama server. Please run 'ollama pull {model_name}' or wait for background download.")
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
                raise ValueError(f"Model '{model_name}' was not found in your Ollama server. Please run 'ollama pull {model_name}' or wait for background download.")
            raise e

def clean_json_string(text: str) -> str:
    """Helper to strip markdown backticks and clean a string so it can be parsed as JSON"""
    # Remove markdown code blocks if present
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def generate_json(prompt: str, system_instruction: str = None, task: str = "fast") -> dict | list:
    """Generate and parse JSON from the LLM. Retries once with clean prompt on JSON parse failure."""
    # Try once
    try:
        response_text = generate_text(prompt, system_instruction, task=task)
        cleaned = clean_json_string(response_text)
        return json.loads(cleaned)
    except Exception as e:
        # If it fails, retry once with an explicit request for clean JSON
        retry_prompt = f"{prompt}\n\nCRITICAL: The previous output failed to parse as valid JSON. Ensure you return ONLY a raw JSON format. No markdown blocks, no extra text."
        response_text = generate_text(retry_prompt, system_instruction, task=task)
        cleaned = clean_json_string(response_text)
        return json.loads(cleaned)

def generate_vision(image_bytes: bytes, mime_type: str, prompt: str) -> str:
    """Generate text description for an image using the active provider's vision model."""
    cfg = get_active_config()
    if cfg["provider"] == "gemini":
        if not cfg["gemini_key"]:
            raise ValueError("Gemini API key is not configured.")
        client = genai.Client(api_key=cfg["gemini_key"])
        model_name = GEMINI_TASK_MODELS.get("vision", "gemini-2.5-flash")
        
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            ]
        )
        return response.text
    else:
        import base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        url = f"{cfg['ollama_url']}/api/generate"
        model_name = OLLAMA_TASK_MODELS.get("vision", "moondream:latest")
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {"temperature": 0.0}
        }
        try:
            response = requests.post(url, json=payload, timeout=90)
            if response.status_code == 404:
                raise ValueError(f"Model '{model_name}' was not found in your Ollama server. Please run 'ollama pull {model_name}' or wait for background download.")
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
                raise ValueError(f"Model '{model_name}' was not found in your Ollama server. Please run 'ollama pull {model_name}' or wait for background download.")
            raise e

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using the active provider."""
    if not texts:
        return []
        
    cfg = get_active_config()
    if cfg["provider"] == "gemini":
        if not cfg["gemini_key"]:
            raise ValueError("Gemini API key is not configured.")
        client = genai.Client(api_key=cfg["gemini_key"])
        
        # We fetch embeddings in batch or individually
        embeddings = []
        for text in texts:
            result = client.models.embed_content(
                model=cfg["gemini_embed"],
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings
    else:
        url = f"{cfg['ollama_url']}/api/embed"
        payload = {
            "model": cfg["ollama_embed"],
            "input": texts
        }
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        return response.json()["embeddings"]
