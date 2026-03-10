
def compute(ranked_l_l : list[list[str]]):
    count = len(ranked_l_l)
    freq_d = {}
    rank_d = {}
    for l in ranked_l_l:
        for term in l:
            if term in freq_d:
                freq_d[term] += 1
            else:
                freq_d[term] = 1

    for term, freq in freq_d.items():
        if freq != count:
            raise ValueError(f'{term}, {freq} != {count}')


    for idx, l in enumerate(ranked_l_l):
        for rank, term in enumerate(l, start=1):
            if term in rank_d:
                rank_d[term].append(rank)
            else:
                rank_d[term] = [rank]
                
    # print(rank_d)
    
    term_socre_l = []
    for term, rank_l in rank_d.items():
        score = 0.
        for rank in rank_l:
            score += 1/(60.+rank)
        term_socre_l.append((term, score))

    return sorted(term_socre_l, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    print(compute([['A', 'B', 'C'],['C','B','A']]))
        


    
        