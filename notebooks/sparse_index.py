import os
import os.path
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import re
import Stemmer
import text_chunk
from sparse_engine import SparseSearchEngine

class SparseIndex:
    def __init__(self, model, parent_documents: list[dict], work_path: str):
        self.model = model
        self.parent_documents = parent_documents
        self.parent_indices = []
        self.work_path = work_path
        self.matrix_path = os.path.join(self.work_path, 'matrix.npz')
        self.vocab_path = os.path.join(self.work_path, 'vocab.txt')
        self.doc_ids_path = os.path.join(self.work_dir, "doc_ids.txt")
        self.engine = SparseSearchEngine(self.work_path)

    def __load_sparse_dict(self, sparse_dict_path: str):
        ret = []
        i = 0
        while True:
            fn = os.path.join(sparse_dict_path, f"{i}.pkl")
            if not os.path.exists(fn):
                break
            with open(fn, 'rb') as inf:
                ret.extend(pickle.load(inf))
            i += 1
        return ret
            

    def load(self):
        with open(os.path.join(self.work_path, 'parent.txt'), 'r') as inf:
            for line in inf:
                self.parent_indices.append(int(line.strip()))
        
        if not os.path.exists(self.matrix_path) or not os.path.exists(self.vocab_path) or os.path.exists(self.doc_ids_path):
            sparse_dict_l = self.__load_sparse_dict(self.work_path)
            print("loaded, sparse_dict_l.len:", len(sparse_dict_l))
            self.engine.build_index_by_dict_list(sparse_dict_l)
            self.engine.save()

        self.engine.load()

    def search(self, query: str, top_k=10):
        q = self.model.encode(texts, 
                              batch_size=10, 
                              return_dense=False, 
                              return_sparse=True, 
                              return_colbert_vecs=False)['lexical_weights']

        res_l = self.engine.search(q, top_k) # [(ids, score)]

        ret_doc_l = []
        seen_parent_idx_set = set([self.parent_indices[res] for res in res_l])

        for parent_idx in seen_parent_idx_set:
            ret_doc_l.append(parent_documents[parent_idx])

        return ret_doc_l

    
    
    
        
        
        