import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
import json

class SemanticCache:
    def __init__(self, embedding_model_path: str, cache_path: str):
        self.model = SentenceTransformer(embedding_model_path)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # annoy 用 angular（余弦距离），配合 normalize_embeddings=True 等价于内积搜索
        self.index = AnnoyIndex(self.embedding_dim, metric="angular")
        self.vector_id_map = {}
        self.id_counter = 0

        # 使用 sqlite 作为缓存
        self.cache_db_path = os.path.join(cache_path, "semantic_cache.db")
        os.makedirs(cache_path, exist_ok=True)
        self.conn = sqlite3.connect(self.cache_db_path, check_same_thread=False)
        self._init_db()
        self._load_cache()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                query TEXT PRIMARY KEY,
                response TEXT,
                plan TEXT,
                intent TEXT
            )
        ''')
        self.conn.commit()

    def _load_cache(self):
        print("加载历史语义缓存中...")
        cursor = self.conn.cursor()
        cursor.execute("SELECT query FROM cache")
        all_queries = cursor.fetchall()
        for (query,) in all_queries:
            vector = self.model.encode(query, normalize_embeddings=True).astype(np.float32)
            self.index.add_item(self.id_counter, vector)
            self.vector_id_map[self.id_counter] = query
            self.id_counter += 1

        if self.id_counter > 0:
            self.index.build(10)  # 建 10 棵树（速度和精度的折中）
        print(f"已恢复 {self.id_counter} 条语义问答缓存\n")

    def get_embedding(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def search_similar_query(self, query_vector: np.ndarray):
        threshold = 0.0
        top_k = 1
        if self.id_counter == 0:
            return None, 0, None

        ids, distances = self.index.get_nns_by_vector(query_vector, top_k, include_distances=True)
        if ids:
            idx = ids[0]
            # Annoy 返回的是距离 (0 = 最近)，余弦距离 -> 相似度 = 1 - d
            score = 1 - distances[0]
            if score >= threshold:
                matched_query = self.vector_id_map[idx]
                cursor = self.conn.cursor()
                cursor.execute("SELECT response, plan, intent FROM cache WHERE query=?", (matched_query,))
                row = cursor.fetchone()
                cached_data = {"response": row[0], "plan": row[1], "intent": row[2]} if row else {}
                return matched_query, score, cached_data
        return None, 0, None

    def save_to_cache(self, query: str, response: str = None, plan: str = None, intent: dict = None):
        if response is None and plan is None and intent is None:
            print(f"[警告] 未传入 response、plan 或 intent，跳过缓存保存：{query}")
            return

        cursor = self.conn.cursor()
        intent_str = json.dumps(intent, ensure_ascii=False) if intent is not None else None

        if response is not None and plan is not None and intent is not None:
            cursor.execute('''
                INSERT OR REPLACE INTO cache (query, response, plan, intent)
                VALUES (?, ?, ?, ?)
            ''', (query, response, plan, intent_str))
        else:
            fields = ['query']
            values = [query]
            updates = []

            if response is not None:
                fields.append('response')
                values.append(response)
                updates.append('response=excluded.response')
            if plan is not None:
                fields.append('plan')
                values.append(plan)
                updates.append('plan=excluded.plan')
            if intent is not None:
                fields.append('intent')
                values.append(intent_str)
                updates.append('intent=excluded.intent')

            sql = f'''
                INSERT INTO cache ({', '.join(fields)})
                VALUES ({', '.join(['?'] * len(values))})
                ON CONFLICT(query) DO UPDATE SET {', '.join(updates)}
            '''
            cursor.execute(sql, values)

        self.conn.commit()

        # 添加向量索引
        vector = self.get_embedding(query)
        self.index.add_item(self.id_counter, vector)
        self.vector_id_map[self.id_counter] = query
        self.id_counter += 1

        # 每次新增后重新 build（简单写法，频繁插入可以批量重建以提高效率）
        self.index.build(10)

        cursor.execute('SELECT * FROM cache WHERE query=?', (query,))
        row = cursor.fetchone()
        print(f"[DEBUG] 数据库中的条目: {row}")

        if row and row[-1]:
            try:
                parsed_intent = json.loads(row[-1])
                print(f"[DEBUG] intent 字典解析结果: {parsed_intent}")
            except Exception as e:
                print(f"[警告] intent 字段解析失败: {e}")

    def extract_plan(self, response_text: str) -> str:
        return response_text.split("。")[0] + "。" if "。" in response_text else response_text

    def close(self):
        self.conn.close()


# 用法示例
if __name__ == "__main__":
    cache = SemanticCache("./m3e-small", "./cache_sqlite")
    cache.save_to_cache("你好吗", response="我很好。", plan="打招呼")
    query = "你最近怎么样"
    vec = cache.get_embedding(query)
    result = cache.search_similar_query(vec)
    print(result)
