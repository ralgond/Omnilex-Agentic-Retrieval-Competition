import os
import os.path
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import pickle
import re
import bm25s
import Stemmer
import text_chunk
from process_de import DEProcessor

class BM25Index:
    """BM25 index for keyword search over legal documents.

    Supports Swiss federal laws (SR) and court decisions (BGE).
    """

    def __init__(
        self,
        documents: list[dict] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
    ):
        """Initialize BM25 index.

        Args:
            documents: List of document dictionaries
            text_field: Key for document text in dict
            citation_field: Key for citation string in dict
        """
        self.text_field = text_field
        self.citation_field = citation_field

        self.documents: list[dict] = []
        self.index = bm25s.BM25()
        self.stemmer = Stemmer.Stemmer("german")
        self.de_processor = DEProcessor()
        self._tokenized_corpus: list[list[str]] = []

        if documents:
            self.build(documents)

    def preprocess(self, documents):
        ret = []
        for doc in tqdm(documents, total=len(documents), desc="bm25index.preprocess"):
            text = doc.get(self.text_field, "")
            new_text = self.de_processor.lowercase_splitcompound(text)
            doc[self.text_field] = new_text
            ret.append(doc)
        return ret

    def build(self, documents: list[dict]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of document dictionaries
        """
        self.documents = self.preprocess(documents)

        corpus_texts = [doc.get(self.text_field, "") for doc in self.documents]
        corpus_tokens = bm25s.tokenize(
                            corpus_texts,
                            stopwords='de',
                            stemmer=self.stemmer,
                            lower=True
                        )

        # Build BM25 index
        self.index.index(corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]:
        """Search the index with a query.

        Args:
            query: Search query string
            top_k: Number of results to return
            return_scores: Whether to include BM25 scores in results

        Returns:
            List of matching documents (with optional scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
            
        query = self.de_processor.lowercase_splitcompound(query)
        
        queries = [query]
        
        query_tokens = bm25s.tokenize(
            queries,
            stopwords='de',
            stemmer=self.stemmer,
            lower=True
        )

        if not query_tokens:
            return []

        # retrieve 返回索引，再映射回原始 corpus
        results, scores = self.index.retrieve(
            query_tokens,
            k=top_k,
        )
        
        output = []
        for q_idx, query in enumerate(queries):
            hits = []
            for rank in range(results.shape[1]):
                doc_idx = results[q_idx, rank]
                hits.append({
                    "rank": rank + 1,
                    "score": round(float(scores[q_idx, rank]), 4),
                    "text": self.documents[doc_idx]["text"],
                    "citation": self.documents[doc_idx]["citation"],
                })
            output.append({"query": query, "hits": hits})
        return output

    def save(self, path: Path | str) -> None:
        """Save index to disk.

        Args:
            path: Path to save index (creates .pkl file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.index.save(path)

        doc_path = str(path) + '.doc'

        data = {}
        data['text_field'] = self.text_field
        data['citation_field'] = self.citation_field
        data["documents"] = self.documents
        
        with open(doc_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        """Load index from disk.

        Args:
            path: Path to saved index

        Returns:
            Loaded BM25Index instance
        """
        path = Path(path)

        doc_path = str(path) + ".doc"
        with open(doc_path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            text_field=data["text_field"],
            citation_field=data.get("citation_field", "citation"),
        )
        instance.documents = data["documents"]
        instance.index = bm25s.BM25.load(str(path))

        return instance

def load_csv_corpus(
    csv_path: Path,
    chunk_size: int = 100_000,
    max_rows: int | None = None
) -> list[dict]:
    """Load CSV corpus into list of dicts with progress bar.
    
    Args:
        csv_path: Path to CSV file with 'citation' and 'text' columns
        chunk_size: Rows to process per chunk (for memory efficiency)
        max_rows: Optional limit on rows (for testing with smaller corpus)
    
    Returns:
        List of {"citation": str, "text": str} dicts
    """
    documents = []
    
    # Count rows for progress bar (fast line count)
    print(f"Counting rows in {csv_path.name}...")
    with open(csv_path, encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # minus header
    
    if max_rows:
        total_rows = min(total_rows, max_rows)
    print(f"Total rows to load: {total_rows:,}")
    
    rows_loaded = 0
    with tqdm_notebook(total=total_rows, desc=f"Loading {csv_path.name}") as pbar:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                if max_rows and rows_loaded >= max_rows:
                    break
                documents.append({
                    "citation": str(row["citation"]),
                    "text": str(row["text"]) if pd.notna(row["text"]) else ""
                })
                rows_loaded += 1
            pbar.update(min(len(chunk), total_rows - pbar.n))
            if max_rows and rows_loaded >= max_rows:
                break
    
    return documents

    
def get_or_build_index(
    name: str,
    csv_path: Path,
    index_path: Path,
    force_rebuild: bool = False,
    max_rows: int | None = None,
    chunk_size: int | None = None,
    overlap_size: int | None = None
) -> BM25Index:
    """Load cached index or build from CSV.
    
    Args:
        name: Index name for logging
        csv_path: Path to corpus CSV
        index_path: Path to cache index pickle
        force_rebuild: If True, rebuild even if cache exists
        max_rows: Optional row limit (for testing with smaller corpus)
    
    Returns:
        BM25Index instance
    """
        
    
    # Use cached index if available and not forcing rebuild
    if index_path.exists() and not force_rebuild:
        print(f"Loading cached {name} index from {index_path}")
        index = BM25Index.load(index_path)
        print(f"  Loaded {len(index.documents):,} documents")
        return index
    
    # Check CSV exists
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Creating empty index.")
        return BM25Index(documents=[])
    
    # Load corpus from CSV
    print(f"\n{'='*50}")
    print(f"Building {name} index from {csv_path}")
    print(f"{'='*50}")
    documents = load_csv_corpus(csv_path, max_rows=max_rows)

    # chunk the documents
    if chunk_size is not None and overlap_size is not None:
        documents = text_chunk.batch_chunk_with_sliding_window(documents, chunk_size, overlap_size)
        print("chunk the documents, count:", len(documents))
    
    if not documents:
        print(f"Warning: No documents loaded. Creating empty index.")
        return BM25Index(documents=[])
    
    # Build BM25 index
    print(f"\nBuilding BM25 index for {len(documents):,} documents...")
    index = BM25Index(
        documents=documents,
        text_field="text",
        citation_field="citation"
    )
    print(f"Index built successfully!")
    
    # Cache index for future runs
    print(f"Saving index to {index_path}...")
    index.save(index_path)
    print(f"Index cached.")
    
    return index

if __name__ == "__main__":
    # text = 'A '* 40 + 'B '*40
    # chunks = text_chunk.chunk_with_sliding_window(text, chunk_size=30, overlap=10)
    # for chunk in chunks:
    #     print(chunk)

    splitter = Splitter()
    print(splitter.split_compound("Autobahnraststätte"))

    print(splitter.split_compound("Ist"))
    print(splitter.split_compound("Das"))