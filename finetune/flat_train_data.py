import json
import random
d_l = []

with open("../ft_data/train.jsonl", 'r', encoding='utf-8') as inf:
    for line in inf:
        d = json.loads(line.strip())
        d_l.append(d)

random.seed(42)

ret_l = []
for d in d_l:
    for pos in d['pos']:
        ret = {'query':d['query'], 'passage':pos, 'label':1}
        ret_l.append(ret)
    for neg in d['neg']:
        ret = {'query':d['query'], 'passage':neg, 'label':0}
        ret_l.append(ret)

random.shuffle(ret_l)

with open("../ft_data/train_flat_train.jsonl", "w", encoding='utf-8') as of:
    for ret in ret_l[:20000]:
        of.write(json.dumps(ret, ensure_ascii=False)+'\n')

with open("../ft_data/train_flat_valid.jsonl", "w", encoding='utf-8') as of:
    for ret in ret_l[20000:]:
        of.write(json.dumps(ret, ensure_ascii=False)+'\n')