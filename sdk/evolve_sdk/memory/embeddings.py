"""
Embeddings for evolution memory.

Uses sentence-transformers for generating embeddings from code and text.
Supports lazy loading to avoid heavy imports when not needed.
"""

import hashlib
from typing import Any

# Lazy imports to avoid loading torch/transformers when not needed
_model = None
_model_name = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy load the sentence-transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
    return _model


class CodeEmbedder:
    """
    Generates embeddings for code and text.

    Uses sentence-transformers with all-MiniLM-L6-v2 by default.
    Provides methods for embedding code, diffs, and problem descriptions.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder.

        Args:
            model_name: Name of the sentence-transformer model to use.
        """
        global _model_name
        _model_name = model_name
        self._model = None  # Lazy load

    @property
    def model(self):
        """Lazy load and return the model."""
        if self._model is None:
            self._model = _get_model()
        return self._model

    def embed_code(self, code: str) -> list[float]:
        """
        Generate embedding for code.

        Preprocesses code to improve embedding quality:
        - Removes comments
        - Normalizes whitespace
        - Keeps structure identifiers

        Args:
            code: Python code string

        Returns:
            List of floats representing the embedding
        """
        # Preprocess code for better embeddings
        processed = self._preprocess_code(code)
        embedding = self.model.encode(processed, convert_to_numpy=True)
        return embedding.tolist()

    def embed_diff(self, diff: str) -> list[float]:
        """
        Generate embedding for a code diff.

        Focuses on the changed lines for better pattern matching.

        Args:
            diff: Git-style diff string

        Returns:
            List of floats representing the embedding
        """
        # Extract just the changed lines for focused embedding
        changed_lines = []
        for line in diff.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                changed_lines.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                changed_lines.append(line[1:])

        if changed_lines:
            text = "\n".join(changed_lines)
        else:
            text = diff

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for plain text (e.g., problem descriptions).

        Args:
            text: Text string

        Returns:
            List of floats representing the embedding
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of text strings

        Returns:
            List of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (0 to 1)
        """
        import numpy as np
        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess code for better embedding quality.

        - Removes docstrings and comments
        - Normalizes whitespace
        - Keeps identifiers and structure
        """
        import re

        # Remove docstrings
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        code = re.sub(r"'''[\s\S]*?'''", "", code)

        # Remove comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)

        # Normalize whitespace but keep structure
        lines = [line.rstrip() for line in code.split("\n") if line.strip()]
        code = "\n".join(lines)

        return code

    @staticmethod
    def hash_code(code: str) -> str:
        """
        Generate a hash of code for quick comparison.

        Args:
            code: Code string

        Returns:
            SHA256 hash of normalized code
        """
        # Normalize before hashing
        normalized = "\n".join(line.strip() for line in code.split("\n") if line.strip())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# Singleton instance for convenience
_embedder: CodeEmbedder | None = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> CodeEmbedder:
    """Get or create a singleton CodeEmbedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = CodeEmbedder(model_name)
    return _embedder
