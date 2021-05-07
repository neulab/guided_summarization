# A script to get the oracle-selected relations from pruned Stanford OpenIE relations
import sys
import re
import numpy as np

import nltk
from nltk.corpus import stopwords
src_oie = open(sys.argv[1]).readlines()
tgt_sum = open(sys.argv[2]).readlines()

def _get_ngrams(n, text):
    """Calcualtes n-grams.
    Args:
      n: which n-grams to calculate
      text: An array of tokens
    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])
    return _get_ngrams(n, words)

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def _rouge_clean(s):
    return re.sub(r'[^a-zA-Z0-9 ]', '', s)


out=open(sys.argv[3], 'w')

cnt=0
tot=0
for src, tgt in zip(src_oie, tgt_sum):
    tgt_sents = tgt.strip().split('<q>')
    tgt_sents_rouge = [_rouge_clean(s_t).split() for s_t in tgt_sents]
    tgt_sents_1grams = set.union(*map(set, [(_get_word_ngrams(1, [sent])) for sent in tgt_sents_rouge]))
    tgt_sents_2grams = set.union(*map(set, [(_get_word_ngrams(2, [sent])) for sent in tgt_sents_rouge]))

    src = src.rstrip()
    if src == '':
        out.write('\n')
        continue
    new_rels = []
    new_scores = []
    rels = src.split('\t')
    rel_1grams = []
    rel_2grams = []
    rels_text=[]
    for rel in rels:
        s, v, o = rel.split('|||')
        s = s.split('#')[0]
        v = v.split('#')[0]
        o = o.split('#')[0]
        rel_sent = _rouge_clean(' qqqq '.join([s, v, o])).split()
        rel_1gram = _get_word_ngrams(1, [rel_sent])
        rel_2gram = _get_word_ngrams(2, [rel_sent])
        rel_1grams.append(rel_1gram)
        rel_2grams.append(rel_2gram)
        rels_text.append('|||'.join([s, v, o]))


    sel_ids = []
    max_score = 0
    pre_max_score = 0
    max_scores = []
    for _ in range(min(len(rels), 5)):
        max_id = -1
        for i in range(len(rels)):
            if i in sel_ids:
                continue
            sel_ids.append(i)

            cur_rel_1grams = set.union(*map(set, [rel_1grams[idd] for idd in sel_ids]))
            cur_rel_2grams = set.union(*map(set, [rel_2grams[idd] for idd in sel_ids]))
            
            rouge_1 = cal_rouge(cur_rel_1grams, tgt_sents_1grams)['f']
            rouge_2 = cal_rouge(cur_rel_2grams, tgt_sents_2grams)['f']
            score = 0.25*rouge_1+0.75*rouge_2

            if score > max_score:
                max_score = score
                max_id = i

            sel_ids = sel_ids[:-1]

        if max_id == -1:
            break
        if max_score - pre_max_score < 0.01:
            break
        sel_ids.append(max_id)
        max_scores.append(max_score)
        pre_max_score = max_score


    for sel_id in sel_ids:
        x = rels_text[sel_id]
        new_rels.append(x)

    cnt += len(sel_ids)
    tot += 1

    max_scores = [str(x) for x in max_scores]
    out.write(f'{" ".join(max_scores)}\t' + '\t'.join(new_rels) + '\n')
