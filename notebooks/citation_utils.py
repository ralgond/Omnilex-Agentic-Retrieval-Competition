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

def BFS_citation_multivalue(court_consideration_d, law_d, first_layer_citation, max_level=2):
    if not isinstance(court_consideration_d, defaultdict):
        raise ValueError("court_consideration_d should be a defaultdict")

    if not isinstance(law_d, defaultdict):
        raise ValueError("law_d should be a defaultdict")
    
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

        raw_text = court_consideration_d.get(citation)
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