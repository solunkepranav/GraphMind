import json
from src import config
from src import llm
from src.vector_store import VectorStore
from src.graph_store import GraphStore
from src.mistake_ledger import MistakeLedger

class QueryAgent:
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.ledger = MistakeLedger()

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
        # 1. Route the query
        route = self.route_query(query)
        category = route.get("category", "HYBRID")
        reasoning = route.get("reasoning", "")

        vector_chunks = []
        graph_relations = []

        # 2. Retrieve Context based on Category
        if category == "SIMPLE":
            vector_chunks = self.vector_store.search(query, top_k=5, source_filter=source_filter)
        elif category == "COMPLEX":
            # Search graph seeds
            seeds = self.graph_store.find_seeds_in_query(query)
            # If no seeds directly in query, search vector store first to extract seed nodes
            if not seeds:
                top_chunks = self.vector_store.search(query, top_k=3, source_filter=source_filter)
                for chunk in top_chunks:
                    chunk_seeds = self.graph_store.find_seeds_in_query(chunk["text"])
                    seeds.extend(chunk_seeds)
                seeds = list(set(seeds))
                
            graph_relations = self.graph_store.traverse_subgraph(seeds, max_depth=2, source_filter=source_filter)
        elif category == "GLOBAL":
            # Global queries get both vector search chunks (broad semantic overview)
            # and high-degree hub relations from the graph
            vector_chunks = self.vector_store.search(query, top_k=8, source_filter=source_filter)
            # Find top high-degree nodes in the graph to get core relations
            high_deg_nodes = [node for node, deg in sorted(self.graph_store.graph.degree(), key=lambda x: x[1], reverse=True)[:5]]
            graph_relations = self.graph_store.traverse_subgraph(high_deg_nodes, max_depth=1, source_filter=source_filter)
        else:  # HYBRID
            vector_chunks = self.vector_store.search(query, top_k=4, source_filter=source_filter)
            seeds = self.graph_store.find_seeds_in_query(query)
            if not seeds:
                for chunk in vector_chunks[:2]:
                    chunk_seeds = self.graph_store.find_seeds_in_query(chunk["text"])
                    seeds.extend(chunk_seeds)
                seeds = list(set(seeds))
            graph_relations = self.graph_store.traverse_subgraph(seeds, max_depth=2, source_filter=source_filter)

        # 3. Format Context
        vector_context = ""
        for idx, chunk in enumerate(vector_chunks):
            vector_context += f"[{idx+1}] Source: {chunk['source']} (Page: {chunk['page']})\nContent: {chunk['text']}\n\n"

        graph_context = ""
        for idx, rel in enumerate(graph_relations):
            sources = ", ".join(rel['sources'])
            pages = ", ".join(rel['pages'])
            graph_context += f"- Relation: ({rel['subject']}) --[{rel['relation']}]--> ({rel['object']}) [Sources: {sources}, Pages: {pages}]\n"

        if not vector_context.strip():
            vector_context = "No relevant text chunks retrieved."
        if not graph_context.strip():
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
   - For text chunks, cite them using bracket numbers matching their index like [1] or [2] (which map to Doc: DocumentName, Page: PageNum).
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

        return {
            "answer": answer,
            "confidence": confidence,
            "category": category,
            "reasoning": reasoning,
            "vector_chunks": vector_chunks,
            "graph_relations": graph_relations
        }
