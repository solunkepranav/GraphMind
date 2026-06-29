import os
import json
import datetime
from src import config

class SourceRegistry:
    def __init__(self):
        self.registry_path = config.SOURCE_REGISTRY_PATH
        self.load()

    def load(self):
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                if not isinstance(self.data, dict) or "sources" not in self.data:
                    self.data = {"sources": []}
            except Exception as e:
                print(f"Error loading source registry, starting fresh: {e}")
                self.data = {"sources": []}
        else:
            self.data = {"sources": []}

    def save(self):
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving source registry: {e}")

    def register_source(self, filename: str, file_type: str, chunk_count: int, page_count: int, size_bytes: int):
        self.load()
        # Check if already exists, update or add
        existing = next((s for s in self.data["sources"] if s["filename"] == filename), None)
        if existing:
            existing["chunk_count"] = chunk_count
            existing["page_count"] = page_count
            existing["size_bytes"] = size_bytes
            existing["ingested_at"] = datetime.datetime.now().isoformat()
        else:
            source_id = os.urandom(4).hex()  # Simple short unique ID
            self.data["sources"].append({
                "id": source_id,
                "filename": filename,
                "file_type": file_type,
                "ingested_at": datetime.datetime.now().isoformat(),
                "chunk_count": chunk_count,
                "page_count": page_count,
                "size_bytes": size_bytes
            })
        self.save()

    def list_sources(self) -> list[dict]:
        self.load()
        return self.data["sources"]

    def delete_source(self, source_id: str) -> str:
        self.load()
        source = next((s for s in self.data["sources"] if s["id"] == source_id), None)
        if source:
            filename = source["filename"]
            self.data["sources"] = [s for s in self.data["sources"] if s["id"] != source_id]
            self.save()
            return filename
        return None

    def reset(self):
        """Clears the registry database."""
        self.data = {"sources": []}
        self.save()
