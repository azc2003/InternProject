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
        self.embeddings = self.vectorizer.encode(texts)
        # faiss需要是float32 type
        self.embeddings = np.array(self.embeddings).astype("float32")

        # 得到每个向量的大小 应该是1024
        dim = self.embeddings.shape[1]
        # 让faiss知道向量dimension是1024 L2就是点间直线距离 这种搜索方法只适合小规模 flat是无压缩
        self.index = faiss.IndexFlatL2(dim)
        # 将我们的向量加入到faiss
        self.index.add(self.embeddings)

    def search(self, query_text, k=2):
        if self.index is None:
            raise ValueError("先用 build_index(texts) 构建索引")

        # 这是我们要查询的向量
        query_vector = self.vectorizer.encode(query_text)
        # 要求搜索是二维向量 （1，dim）
        query_vector = np.array(query_vector).reshape(1, -1).astype("float32")

        # k=2 是找最相似的两个向量， Distance 二维数组L2距离 越小越相似 I 对应索引的Index
        D, I = self.index.search(query_vector, k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            results.append({
                "index": idx,
                "distance": float(dist),
                "text": self.texts[idx]
            })

        print("Query:", query_text)
        print("Top Results:")
        for res in results:
            print(f"- Index: {res['index']}, Distance: {res['distance']:.4f}, Text: {res['text']}")
        return results
