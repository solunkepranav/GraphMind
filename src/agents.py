import json
from src import config
from src import llm
from src.vector_store import VectorStore
from src.graph_store import GraphStore
from src.retrieval import RetrievalOrchestrator, RetrievalResult
from src.mistake_ledger import MistakeLedger, calibrate_confidence
from src.security import PromptInjectionDetector
from src.cache import LRUCache

class QueryAgent:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        bm25_store=None,
        orchestrator: RetrievalOrchestrator = None,
        cache_ttl_seconds: float = 3600
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.bm25_store = bm25_store
        if orchestrator is not None:
            self.orchestrator = orchestrator
        else:
            self.orchestrator = RetrievalOrchestrator(
                vector_store=vector_store,
                bm25_store=bm25_store,
                graph_store=graph_store
            )
        self.ledger = MistakeLedger()
        self.security_detector = PromptInjectionDetector()
        self.cache = LRUCache(maxsize=300, ttl_seconds=cache_ttl_seconds)

    def clear_cache(self):
        """Clears the query response cache."""
        self.cache.clear()

    def route_query(self, query: str) -> dict:
        """Classifies the query into SIMPLE, COMPLEX, GLOBAL, or HYBRID."""
        prompt = config.ROUTER_PROMPT.format(query=query)
        try:
            result = llm.generate_json(prompt, task="fast")
            if isinstance(result, dict) and "category" in result:
                category = str(result["category"]).upper().strip()
                if category in ["SIMPLE", "COMPLEX", "GLOBAL", "HYBRID"]:
                    return result
            # Fallback
            return {"category": "HYBRID", "reasoning": "Fallback classification to hybrid retrieval."}
        except Exception as e:
            print(f"Error routing query: {e}")
            return {"category": "HYBRID", "reasoning": f"Routing failed due to error: {e}"}

    def answer_query(self, query: str, source_filter: list[str] = None) -> dict:
        """Processes the query using agentic routing, retrieves context, and synthesizes an answer."""
        # 0. Check cache
        cache_key = f"{query.strip().lower()}|{','.join(sorted(source_filter or []))}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            res = dict(cached_result)
            res["cached"] = True
            return res

        # 1. Route the query
        route = self.route_query(query)
        category = route.get("category", "HYBRID")
        reasoning = route.get("reasoning", "")

        top_k_map = {
            "SIMPLE": 5,
            "COMPLEX": 4,
            "GLOBAL": 8,
            "HYBRID": 5
        }
        top_k = top_k_map.get(category, 5)

        # 2. Retrieve Context based on Category via unified RetrievalOrchestrator
        retrieval_result = self.orchestrator.retrieve(
            query=query,
            query_type=category,
            top_k=top_k,
            source_filter=source_filter
        )
        vector_chunks = retrieval_result.chunks
        graph_relations = retrieval_result.graph_edges

        # 3. Format Context using PromptInjectionDetector for security isolation
        vector_context, graph_context = self.security_detector.format_safe_context(
            vector_chunks, graph_relations
        )
        if not vector_chunks:
            vector_context = "No relevant text chunks retrieved."
        if not graph_relations:
            graph_context = "No relevant knowledge graph relations retrieved."

        # 4. Generate Answer and Self-Reflected Confidence (with Critic Guard)
        qa_system_instruction = (
            "You are GraphMind, an advanced RAG question answering system. Synthesize your final answer "
            "along with an estimated confidence score (0 to 100) representing how fully the context answers the query. "
            "Output your response as a JSON object with two fields: 'answer' (markdown text with citations) "
            "and 'confidence' (integer between 0 and 100)."
        )
        
        qa_prompt = f"""Context Chunks:
{vector_context}

Knowledge Graph Relations:
{graph_context}

Question: {query}

Instructions:
1. Rely ONLY on the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."
2. Cite your sources.
   - For text chunks, cite them using document name and page: [Doc: DocumentName, Page: PageNum] or bracket index.
   - For graph relations, cite them like: (Subject -> relation -> Object).
3. Synthesize a coherent, professional answer in markdown.
4. Output your response ONLY as a JSON object with 'answer' and 'confidence' fields. Do not use markdown wrappers.
"""
        # Append historical OKF failures to prompt to prevent regression
        recent_mistakes = self.ledger.get_recent_mistakes()
        if recent_mistakes:
            qa_prompt += f"\n\n{config.OKF_VALIDATION_RULES.format(historical_failures=recent_mistakes)}"

        max_retries = 2
        answer = "I cannot find the answer in the provided documents."
        confidence = 0

        for attempt in range(max_retries):
            try:
                task_type = "reasoning" if attempt == 0 else "validation"
                response_json = llm.generate_json(qa_prompt, system_instruction=qa_system_instruction, task=task_type)
                if isinstance(response_json, dict) and "answer" in response_json:
                    answer = response_json["answer"]
                    confidence = response_json.get("confidence", 80)
                else:
                    answer = str(response_json)
                    confidence = 70
            except Exception as e:
                # Fallback if JSON parsing fails
                print(f"Failed to generate JSON answer, falling back to standard text: {e}")
                fallback_prompt = f"{config.QA_PROMPT.format(vector_context=vector_context, graph_context=graph_context, question=query)}\nOutput raw markdown response directly."
                task_type = "reasoning" if attempt == 0 else "validation"
                answer = llm.generate_text(fallback_prompt, task=task_type)
                confidence = 65

            # CRITIC GUARD: Validate the output
            if "[[ " in answer or " ]]" in answer or not answer.strip():
                print(f"Validation failed on attempt {attempt + 1}. Retrying...")
                self.ledger.log_mistake(query, answer, "Failed formatting or empty response.")
                # Give feedback in the prompt for the next try
                qa_prompt += f"\n\nPrevious attempt failed validation. Please ensure proper markdown and citation formatting, and do not use unclosed brackets."
            else:
                break # Passed validation

        # Compute calibrated confidence from normalized observable feature signals
        calibration = calibrate_confidence(
            raw_llm_score=confidence,
            answer=answer,
            retrieved_chunks=vector_chunks,
            graph_relations=graph_relations
        )

        final_result = {
            "answer": answer,
            "confidence": calibration["calibrated_score"],
            "raw_confidence": confidence,
            "confidence_signals": calibration["signals"],
            "category": category,
            "reasoning": reasoning,
            "vector_chunks": vector_chunks,
            "graph_relations": graph_relations,
            "retrieval_metadata": retrieval_result.metadata,
            "cached": False
        }
        self.cache.set(cache_key, final_result)
        return final_result
