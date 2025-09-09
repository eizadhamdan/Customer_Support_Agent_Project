# utils/embeddings.py

import os
from sentence_transformers import SentenceTransformer

# Directory to save/load embedding models locally
MODEL_DIR = os.path.join("models_cache", "embeddings")
os.makedirs(MODEL_DIR, exist_ok=True)


class EmbeddingGenerator:
    """
    Utility class for generating sentence embeddings using SentenceTransformers.
    Downloads model once and caches it locally for reuse.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.

        Args:
            model_name (str): The Hugging Face model name for embeddings.
                              Default is all-MiniLM-L6-v2 (lightweight, 384-dim).
        """
        self.model_name = model_name
        self.model_path = os.path.join(MODEL_DIR, model_name.replace("/", "_"))

        if not os.path.exists(self.model_path):
            print(f"Downloading embedding model {model_name}...")
            self.model = SentenceTransformer(model_name)
            self.model.save(self.model_path)
        else:
            print(f"Loading embedding model from local cache: {self.model_path}")
            self.model = SentenceTransformer(self.model_path)

    def encode(self, texts, normalize: bool = True):
        """
        Generate embeddings for a single string or list of strings.

        Args:
            texts (str or List[str]): Input text(s) to embed.
            normalize (bool): Whether to normalize embeddings (unit length). Default True.

        Returns:
            np.ndarray: Embedding vector(s).
        """
        return self.model.encode(texts, normalize_embeddings=normalize)
