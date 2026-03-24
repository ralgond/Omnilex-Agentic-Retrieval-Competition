import re
from collections import defaultdict

def extract_citations_from_text(text: str) -> list[str]:
    """Extract citations from any text (tool output or final answer)."""
    citations = []
    
    # SR pattern: SR followed by number (optionally with article)
    sr_matches = re.findall(
        r"SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(sr_matches)
    
    # BGE pattern: BGE volume section page
    bge_matches = re.findall(
        r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(bge_matches)
    
    # Art. pattern: Art. X LAW (e.g., Art. 1 ZGB, Art. 41 OR)
    art_matches = re.findall(
        r"Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)?[A-Z]{2,}",
        text,
        re.IGNORECASE
    )
    citations.extend(art_matches)
    
    return list(set(citations))

def extract_citations_from_text_repeat(text: str) -> list[str]:
    """Extract citations from any text (tool output or final answer)."""
    citations = []
    
    # SR pattern: SR followed by number (optionally with article)
    sr_matches = re.findall(
        r"SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(sr_matches)
    
    # BGE pattern: BGE volume section page
    bge_matches = re.findall(
        r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(bge_matches)
    
    # Art. pattern: Art. X LAW (e.g., Art. 1 ZGB, Art. 41 OR)
    art_matches = re.findall(
        r"Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)?[A-Z]{2,}",
        text,
        re.IGNORECASE
    )
    citations.extend(art_matches)

    return citations


def BFS_citation(court_consideration_d, law_d, first_layer_citation, max_level=2):
    queue = []
    seen_citation = set()
    idx = 0
    for citation in first_layer_citation:
        queue.append((citation,0))

    ret = []
    
    while idx < len(queue):
        citation,level = queue[idx]
        if level >= max_level:
            idx += 1
            break
            
        if citation in seen_citation:
            idx += 1
            continue

        raw_text = court_consideration_d.get(citation, None)
        if raw_text is None:
            raw_text = law_d.get(citation, None)
            if raw_text is None:
                idx += 1
                continue

        ret.append({'citation':citation, 'text':raw_text}) # 没见过这个hits
        seen_citation.add(citation) # 现在看见了

        for c in extract_citations_from_text(raw_text):
            if c in seen_citation:
                continue
            
            if c in court_consideration_d:
                queue.append((c, level+1))

            if c in law_d:
                queue.append((c, level+1))

        idx += 1

    return ret

def __get_source(court_consideration_d, law_d, citation):
    if citation in court_consideration_d:
        return court_consideration_d[citation], True
    elif citation in law_d:
        return law_d[citation], False
    else:
        return None, False

def second_layer_citation_with_score(court_consideration_d, law_d, first_layer_citation_with_score):
    ignore_citation_set = set()
    for citation, score in first_layer_citation_with_score:
        ignore_citation_set.add(citation)

    d = {}
    for citation, score in first_layer_citation_with_score:
        text, is_court = __get_source(court_consideration_d, law_d, citation)
        if text is None:
            continue

        extracted_citations = extract_citations_from_text(text)
        for _c in extracted_citations:
            if _c in ignore_citation_set:
                continue
            else:
                if _c not in d:
                    d[_c] = score
                else:
                    d[_c] += score

    return sorted([(citation, score) for citation, score in d.items()], key=lambda x: x[1], reverse=True)

from collections import defaultdict

def second_layer_citation_with_score_ge2(court_consideration_d, law_d, first_layer_citation_with_score):
    ignore_citation_set = set()
    for citation, score in first_layer_citation_with_score:
        ignore_citation_set.add(citation)

    d = defaultdict(int)
    for citation, score in first_layer_citation_with_score:
        text, is_court = __get_source(court_consideration_d, law_d, citation)
        if text is None:
            continue

        __d = defaultdict(int)
        extracted_citations = extract_citations_from_text_repeat(text)
        for _c in extracted_citations:
            if _c in ignore_citation_set:
                continue
            __d[_c] += 1

        __l = [c for c, count in __d.items() if count >= 2]

        for c in __l:
            d[c] += 1

    return sorted([(citation, score) for citation, score in d.items()], key=lambda x: x[1], reverse=True)
    

def second_layer_citation_with_tfidf(court_consideration_d, law_d, first_layer_citation_with_score, citation_idf):
    ignore_citation_set = set()
    for citation, score in first_layer_citation_with_score:
        ignore_citation_set.add(citation)

    d = {}
    for citation, score in first_layer_citation_with_score:
        text, is_court = __get_source(court_consideration_d, law_d, citation)
        if text is None:
            continue

        extracted_citations = extract_citations_from_text_repeat(text)
        for _c in extracted_citations:
            if _c in ignore_citation_set:
                continue
            else:
                if _c not in d:
                    d[_c] = score
                else:
                    d[_c] += score

    d2 = {}
    for citation, tf in d.items():
        if citation not in citation_idf:
            print(f'{citation} not in citation_idf')
        else:
            d2[citation] = tf * citation_idf[citation]

    return sorted([(citation, score) for citation, score in d2.items()], key=lambda x: x[1], reverse=True)


import re
import math



import re

def split_sentences_with_citations(text: str) -> list[str]:
    """先保护citation占位符，断句后还原。"""
    
    # 定义所有citation模式（顺序重要：长模式先匹配）
    citation_patterns = [
        r"SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?",
        r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?",
        r"Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)?[A-Z]{2,}",
    ]
    combined = "|".join(f"(?:{p})" for p in citation_patterns)
    
    # Step 1: 替换为占位符
    placeholders = {}
    counter = [0]
    
    def replacer(m):
        key = f"__CITATION_{counter[0]}__"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key
    
    protected_text = re.sub(combined, replacer, text, flags=re.IGNORECASE)
    
    # Step 2: 断句（此时句号已安全）
    # 简单规则：句号/问号/感叹号 + 空格 + 大写字母
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', protected_text)
    
    # Step 3: 还原占位符
    restored = []
    for sent in sentences:
        for key, value in placeholders.items():
            sent = sent.replace(key, value)
        restored.append(sent.strip())
    
    return [s for s in restored if s]

import re

def __split_sentences(text: str) -> list[str]:
    """
    正确断句：先将citation中的句号保护起来，断句后再还原。
    """
    citations = extract_citations_from_text(text)
    
    # 1. 用占位符替换所有citation，避免其中的句号干扰断句
    placeholder_map = {}
    protected = text
    for i, citation in enumerate(citations):
        placeholder = f"__CITATION_{i}__"
        placeholder_map[placeholder] = citation
        # 替换文本中所有该citation的出现
        protected = protected.replace(citation, placeholder)
    
    # 2. 在保护后的文本上断句
    # 匹配句末标点：.  !  ? 后跟空白或结尾
    raw_sentences = re.split(r'(?<=[.!?])\s+', protected.strip())
    
    # 3. 还原每个句子中的citation占位符
    sentences = []
    for s in raw_sentences:
        for placeholder, original in placeholder_map.items():
            s = s.replace(placeholder, original)
        s = s.strip()
        if s:
            sentences.append(s)
    
    return sentences
    
def compute_citation_score_with_sentence_pos(candidates_with_scores, decay="reciprocal"):
    """
    candidates_with_scores: [(consideration_text, reranker_score), ...]
    返回: {law_citation: aggregated_score}
    """
    law_scores = {}
    
    decay_fn = {
        "reciprocal": lambda p: 1 / (p + 1),
        "log":        lambda p: 1 / math.log(p + 2),
        "exp":        lambda p: math.exp(-0.3 * p),
    }[decay]
    
    for doc, reranker_score in candidates_with_scores:
        text = doc['text']
        cited_laws = extract_citations_from_text(text)  # 你的citation抽取函数
        # sentences = re.split(r'(?<=[.!?])\s+', text)

        sentences = __split_sentences(text)
        
        # 建立每个法条首次出现的句子位置
        law_first_pos = {}
        for i, sent in enumerate(sentences):
            for law in cited_laws:
                if law in sent and law not in law_first_pos:
                    law_first_pos[law] = i
        
        for law, pos in law_first_pos.items():
            position_weight = decay_fn(pos)
            law_scores[law] = law_scores.get(law, 0) + reranker_score * position_weight
    
    return sorted(law_scores.items(), key=lambda x: -x[1])
