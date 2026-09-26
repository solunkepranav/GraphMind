import math

class FakeLLM:
    """Model-free fake LLM returning deterministic or scripted responses."""
    def __init__(self, default_response: str = "Fake LLM answer with citation [Doc: test.pdf, Page: 1]"):
        self.default_response = default_response
        self.call_history = []

    def generate_text(self, prompt: str, system_instruction: str = None, task: str = "fast") -> str:
        self.call_history.append({"prompt": prompt, "system_instruction": system_instruction, "task": task})
        return self.default_response

    def generate_json(self, prompt: str, system_instruction: str = None, task: str = "fast") -> dict | list:
        self.call_history.append({"prompt": prompt, "system_instruction": system_instruction, "task": task})
        if "category" in prompt or "SIMPLE" in prompt:
            return {"category": "HYBRID", "reasoning": "Test classification"}
        if "triples" in prompt.lower() or "subject" in prompt.lower():
            return [{"subject": "Alpha", "relation": "connects_to", "object": "Beta"}]
        return {"result": "ok"}


class FakeEmbedding:
    """Deterministic, model-free embeddings based on character frequency / hash."""
    def __init__(self, dim: int = 64):
        self.dim = dim

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            vec = [0.0] * self.dim
            for i, char in enumerate(text):
                vec[i % self.dim] += ord(char)
            # Normalize vector
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            embeddings.append([v / norm for v in vec])
        return embeddings


class FakeVectorStore:
    """In-memory mock for vector store with exact contract."""
    def __init__(self):
        self.documents = []

    def add_documents(self, documents: list[dict]):
        self.documents.extend(documents)

    def search(self, query: str, top_k: int = 5, n_results: int = None, source_filter: list[str] = None) -> list[dict]:
        k = top_k if n_results is None else n_results
        results = []
        for doc in self.documents:
            if source_filter is not None and doc.get("source") not in source_filter:
                continue
            # Simple keyword overlap simulation for fake scoring
            overlap = sum(1 for w in query.lower().split() if w in doc.get("text", "").lower())
            res = dict(doc)
            res["score"] = float(overlap)
            results.append(res)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]


class FakeGraphStore:
    """In-memory mock for knowledge graph store with exact contract."""
    def __init__(self):
        self.edges = []

    def add_triples(self, triples: list[dict], source: str, page: int, chunk_id: str = None, text: str = None):
        for t in triples:
            self.edges.append({
                "source_node": t["subject"],
                "target_node": t["object"],
                "relation": t["relation"],
                "sources": [source],
                "pages": [page],
                "evidence": [{"source": source, "page": page, "chunk_id": chunk_id or "", "text": text or ""}]
            })

    def find_seeds_in_query(self, query: str) -> list[str]:
        seeds = []
        q_lower = query.lower()
        for e in self.edges:
            if e["source_node"].lower() in q_lower:
                seeds.append(e["source_node"])
            if e["target_node"].lower() in q_lower:
                seeds.append(e["target_node"])
        return list(set(seeds))

    def traverse_subgraph(self, seeds: str | list[str], max_depth: int = 2, depth: int = None, source_filter: list[str] = None) -> list[dict]:
        matching = []
        seed_list = [seeds] if isinstance(seeds, str) else list(seeds)
        seed_lowers = [s.lower() for s in seed_list]
        for edge in self.edges:
            if source_filter is not None and not any(s in source_filter for s in edge.get("sources", [])):
                continue
            if any(s in edge["source_node"].lower() or s in edge["target_node"].lower() for s in seed_lowers):
                matching.append({
                    "u": edge["source_node"],
                    "v": edge["target_node"],
                    "subject": edge["source_node"],
                    "object": edge["target_node"],
                    "relation": edge["relation"],
                    "sources": edge.get("sources", []),
                    "pages": edge.get("pages", []),
                    "evidence": edge.get("evidence", [])
                })
        return matching

    def get_contradictions(self) -> list[dict]:
        return []

    def get_global_summary(self) -> dict:
        return {
            "top_nodes": ["Alpha", "Beta", "Gamma"],
            "total_nodes": 3,
            "total_edges": len(self.edges),
            "communities": [{"id": 1, "size": 3, "nodes": ["Alpha", "Beta", "Gamma"]}],
            "bridges": []
        }
