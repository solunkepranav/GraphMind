import pytest
import requests
from unittest.mock import patch
from src import llm

def test_ollama_offline_connection_error_generate():
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        with patch("src.llm.get_active_config") as mock_cfg:
            mock_cfg.return_value = {
                "provider": "ollama",
                "ollama_url": "http://localhost:11434",
                "ollama_model": "gemma3:4b"
            }
            
            with pytest.raises(ConnectionError) as exc_info:
                llm.generate_text("Test prompt")
            
            assert "Cannot connect to local Ollama service" in str(exc_info.value)
            assert "ollama serve" in str(exc_info.value)

def test_ollama_offline_connection_error_embeddings():
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        with patch("src.llm.get_active_config") as mock_cfg:
            mock_cfg.return_value = {
                "provider": "ollama",
                "ollama_url": "http://localhost:11434",
                "ollama_embed": "nomic-embed-text"
            }
            
            with pytest.raises(ConnectionError) as exc_info:
                llm.get_embeddings(["Test text"])
            
            assert "Cannot connect to local Ollama service" in str(exc_info.value)
