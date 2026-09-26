import re

INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"developer\s+mode\s+(?:enabled|on|activate)",
    r"jailbreak",
    r"override\s+(?:security|system|rules)"
]

class PromptInjectionDetector:
    """Detects indirect and direct prompt injection patterns.
    Enforces the security invariant: detect and isolate without mutating or stripping source text.
    """

    def __init__(self, patterns: list[str] = None):
        self.patterns = patterns or INJECTION_PATTERNS
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def detect(self, text: str) -> dict:
        """Analyzes text for injection signatures."""
        if not text:
            return {"is_injection": False, "matched_patterns": [], "risk_level": "LOW"}

        matches = []
        for regex in self._compiled:
            found = regex.findall(text)
            if found:
                matches.extend(found)

        is_injection = len(matches) > 0
        risk_level = "HIGH" if len(matches) >= 2 else ("MEDIUM" if is_injection else "LOW")

        return {
            "is_injection": is_injection,
            "matched_patterns": list(set(matches)),
            "risk_level": risk_level
        }

    def isolate_chunk(self, chunk: dict) -> str:
        """Wraps chunk in explicit isolation tags.
        Preserves original text verbatim so citations and facts remain intact.
        """
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        page = chunk.get("page", 1)

        detection = self.detect(text)
        if detection["is_injection"]:
            return (
                f'<untrusted_document source="{source}" page="{page}" '
                f'injection_risk="{detection["risk_level"]}" security_alert="true">\n'
                f'{text}\n'
                f'</untrusted_document>'
            )
        else:
            return (
                f'<document source="{source}" page="{page}">\n'
                f'{text}\n'
                f'</document>'
            )

    def format_safe_context(self, chunks: list[dict], graph_relations: list[dict]) -> tuple[str, str]:
        """Formats context with XML boundary delimiters separating untrusted document data
        from prompt instructions.
        """
        chunk_lines = [self.isolate_chunk(c) for c in chunks]
        vector_context = (
            "<retrieved_documents>\n"
            + "\n".join(chunk_lines)
            + "\n</retrieved_documents>"
        )

        relation_lines = []
        for rel in graph_relations:
            subj = rel.get("subject", rel.get("u", ""))
            obj = rel.get("object", rel.get("v", ""))
            relation_label = rel.get("relation", "connected_to")
            sources = ", ".join(str(s) for s in rel.get("sources", []))
            pages = ", ".join(str(p) for p in rel.get("pages", []))
            relation_lines.append(
                f'<relation subject="{subj}" predicate="{relation_label}" object="{obj}" sources="{sources}" pages="{pages}" />'
            )

        graph_context = (
            "<knowledge_graph_relations>\n"
            + "\n".join(relation_lines)
            + "\n</knowledge_graph_relations>"
        )

        return vector_context, graph_context
