import faiss
from tqdm import tqdm
import text_chunk
import numpy as np

def __check_dim(model):
    doc_embeddings = model.encode(
        ['abc'],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )
    dim = doc_embeddings.shape[1] 
    return dim


def query_by_dense_navie(model, query: str, docs: list, top_k=20):
    documents = [doc['text'] for doc in docs]
    citations = [doc['citation'] for doc in docs]

    dim = __check_dim(model)

    # =========================
    # 3. 构建 FAISS 索引
    # =========================
    # 因为做了 normalize，所以用 Inner Product 等价于 cosine
    index = faiss.IndexFlatIP(dim)

    for doc in tqdm(documents, total=len(documents)):
        doc_embeddings = model.encode(
            [doc],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False
        )
        index.add(np.array(doc_embeddings))

    # =========================
    # 4. 查询
    # =========================
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(np.array(query_embedding), top_k)

    rets = []
    for score, idx in zip(scores[0], indices[0]):
        rets.append({'text':documents[idx], 'citation':citations[idx]})
    return rets

def query_chunks_return_max_score(model, query: str, chunks: list, dim: int):
    documents = [doc for doc in chunks]

    # =========================
    # 3. 构建 FAISS 索引
    # =========================
    # 因为做了 normalize，所以用 Inner Product 等价于 cosine
    index = faiss.IndexFlatIP(dim)

    doc_embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False
    )
    index.add(np.array(doc_embeddings))

    # =========================
    # 4. 查询
    # =========================
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(np.array(query_embedding), 1)
    return scores[0][0]

def query_multi_vector(model, 
                       query: str, 
                       doc: str, 
                       chunk_size: int, 
                       overlap_size: int, 
                       dim: int):
    chunks = text_chunk.chunk_with_sliding_window(doc, chunk_size, overlap_size)
    return query_chunks_return_max_score(model, query, chunks, dim)

def query_by_dense_chunk_multi_vector(model, 
                                      query: str, 
                                      docs: list, 
                                      top_k: int, 
                                      chunk_size: int, 
                                      overlap_size: int,
                                      dim: int):
    for doc in tqdm(docs, total=len(docs), desc="query_by_dense_chunk_multi_vector"):
        score = query_multi_vector(model, query, doc['text'], chunk_size, overlap_size, dim)
        doc['score'] = score

    return sorted([doc for doc in docs], key=lambda x: x['score'], reverse=True)[:top_k]

def query_by_dense(model, 
                   query: str, 
                   docs: list, 
                   top_k: int,
                   chunk_size: int | None=None, 
                   overlap_size: int | None=None):
    if chunk_size is not None:
        if overlap_size is None:
            raise ValueError("overlap_size should be integer")
        dim = __check_dim(model)
        return query_by_dense_chunk_multi_vector(model, query, docs, top_k, chunk_size, overlap_size, dim)
    else:
        return query_by_dense_navie(model, query, docs, top_k)
    