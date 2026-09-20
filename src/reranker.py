import logging
from src import config

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Local Cross-Encoder reranker using sentence-transformers on CPU.
    Reranks candidate chunks from RRF pool using a cross-encoder model to optimize precision.
    """

    def __init__(
        self,
        model_name: str = config.RERANKER_MODEL,
        device: str = config.RERANKER_DEVICE,
        top_n: int = config.RERANKER_TOP_N,
        enabled: bool = config.ENABLE_RERANKER
    ):
        self.model_name = model_name
        self.device = device
        self.top_n = top_n
        self.enabled = enabled
        self._model = None
        self._load_attempted = False

    def _load_model(self):
        """Lazy-loads the CrossEncoder model upon first rerank request."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded CrossEncoder model '{self.model_name}' on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder '{self.model_name}': {e}. Falling back to RRF rankings.")
            self._model = None
            self.enabled = False

    def rerank(self, query: str, candidates: list[dict], top_n: int = None) -> list[dict]:
        """Scores and re-ranks candidate text chunks against query.
        Returns top_n chunks sorted descending by cross-encoder score.
        """
        n = top_n if top_n is not None else self.top_n

        if not candidates:
            return []

        if not self.enabled or len(candidates) <= 1:
            return candidates[:n]

        self._load_model()
        if not self._model:
            return candidates[:n]

        try:
            pairs = [[query, c.get("text", "")] for c in candidates]
            scores = self._model.predict(pairs)

            ranked = []
            for candidate, score in zip(candidates, scores):
                c_copy = dict(candidate)
                c_copy["reranker_score"] = float(score)
                ranked.append(c_copy)

            ranked.sort(key=lambda x: x["reranker_score"], reverse=True)
            return ranked[:n]
        except Exception as e:
            logger.warning(f"Error during cross-encoder prediction: {e}. Preserving candidate order.")
            return candidates[:n]
