import pandas as pd
from collections import defaultdict
import math
import json
import re
from tqdm import tqdm

court_df = pd.read_csv("./data/court_considerations.csv")

print("data loaded")

citation_d = defaultdict(int)

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
    
for _, row in tqdm(court_df.iterrows(), total=len(court_df)):
    citations = extract_citations_from_text(row['text'])
    for c in citations:
        citation_d[c] += 1

total_court_count = len(court_df)

idf_d = {}

for c, df in citation_d.items():
    idf_d[c] = math.log((total_court_count+1)*1.0/(df+1))

with open("./data/citation_idf.jsonl", "w+", encoding='utf-8') as of:
    for citation, idf in idf_d.items():
        of.write(json.dumps({"citation":citation, 'idf':idf})+"\n")