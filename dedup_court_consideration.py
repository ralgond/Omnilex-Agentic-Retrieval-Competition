import pandas as pd

court_consideration_df = pd.read_csv("./data/court_considerations.csv")
court_consideration_d = {}
for citation, text in zip(court_consideration_df['citation'].tolist(), court_consideration_df['text'].tolist()):
    if citation in court_consideration_d:
        continue
    court_consideration_d[citation] = text

print("court_consideration_d.len:", len(court_consideration_d))

court_citation_l = [] 
court_text_l = []

for citation, text in court_consideration_d.items():
    court_citation_l.append(citation)
    court_text_l.append(text)
result = pd.DataFrame({'citation':court_citation_l, 'text':court_text_l})
result.to_csv("./data/court_considerations_dedup.csv", index=False)