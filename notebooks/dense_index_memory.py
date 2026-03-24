import os
import os.path
import numpy as np
import faiss
from typing import List, Tuple
import text_chunk
import json

class DenseIndexMemory:
    def __init__(self, model, documents):
        self.model = model # from FlagEmbedding import BGEM3FlagModel
        self.parent_id_to_citation = {}
        self.id_to_parent_id = {}
        embeddings, ids = self._load_embedding(embeddings_path)

        print("DenseIndex.embeddings: ", embeddings.shape)
        
        dim = embeddings.shape[1]

        # =========================
        # 3. 构建 FAISS 索引
        # =========================
        # 因为做了 normalize，所以用 Inner Product 等价于 cosine
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.documents = documents
        self.parent_indices = parent_indices

    def info(self):
        print("[dense_index] documents.len:",len(self.documents), "parent_idx.len:", len(self.parent_indices))

    def _load_embedding(self, jsonl_path):

        null_fp = open(os.devnull, 'w')
        sys.stderr = null_fp

        child_id = 0
        embedding_l = []
        with open(jsonl_path, "r") as inf:
            for parent_id, line in enumerate(inf):
                d = json.loads(line.strip())
                chunks = chunk_with_sliding_window(d['text'], chunk_size=384, overlap=128)
                embedding_l.append(self.model.encode(chunks)['dense_vecs'])
                self.id_to_citation[id] = citation
        sys.stderr = sys.__stderr__
        null_fp.close()
        
        return np.vstack(embedding_l), ids_l

    def search(self, q, top_k):
        '''
        return: list of index of embeddings
        '''
        # =========================
        # 4. 查询
        # =========================
        query_encoded_result = self.model.encode(
            [q]
        )

        # query_embedding = np.array(query_embedding)
        query_embedding = query_encoded_result['dense_vecs']
        # print("query_embedding.shape:", query_embedding.shape)

        scores, indices = self.index.search(query_embedding, top_k)

        parent_indics = [self.parent_indices[idx] for idx in indices[0]]

        seen_parent_indics = set()
        parent_indics2 = []
        for parent_idx in parent_indics:
            if parent_idx in seen_parent_indics:
                pass
            else:
                seen_parent_indics.add(parent_idx)
                parent_indics2.append(parent_idx)
        
        ret = []
        for idx in parent_indics2:
            ret.append(self.documents[idx])
            
        return ret
    