from sentence_transformers import SentenceTransformer
import numpy as np
import os


class Vectorizer:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=self.normalize)
