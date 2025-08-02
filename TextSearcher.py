from vectorizer import Vectorizer
import numpy as np
import faiss

class TextSearcher:
    def __init__(self, model_name="all-MiniLM-L6-v2", normalize=True):
        self.vectorizer = Vectorizer(model_name=model_name, normalize=normalize)
        self.texts = []
        self.embeddings = None
        self.index = None

    def build_index(self, texts):
        self.texts = texts
        PROMPT_PREFIX = "Represent this sentence for retrieval: "
        vector_texts = []
        for t in texts:
            text = t.get("text", "").strip()
            filename = t.get("filename", "")
            modified_time = t.get("modified_time", "")
            created_time = t.get("created_time", "")
            vector_text = PROMPT_PREFIX +(
                f"[filename: {filename}]\n"
                f"[modified_time: {modified_time}]\n"
                f"[created_time: {created_time}]\n"
                f"{text}"
            )
            vector_texts.append(vector_text)
        self.embeddings = self.vectorizer.encode(vector_texts)
        self.embeddings = np.array(self.embeddings).astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, query_text, k):
        if self.index is None:
            raise ValueError("Please build the index first with build_index(texts)")
        PROMPT_PREFIX = "Represent this sentence for retrieval: "
        query_vector = self.vectorizer.encode(PROMPT_PREFIX + query_text)
        query_vector = np.array(query_vector).reshape(1, -1).astype("float32")

        D, I = self.index.search(query_vector, k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = dict(self.texts[idx])
            meta["distance"] = float(dist)
            meta["index"] = idx
            results.append(meta)
        return results
