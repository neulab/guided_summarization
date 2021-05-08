## A script to get the keywords from oracle-selected sentences
## To use this, you need to `pip install summa` first
## The first argument is the path to oracle-selected sentences of input documents
## The second argument is the path to the reference summaries
## The third argument is the path to the output file 
from summa import keywords
import sys
oras = open(sys.argv[1]).readlines()
refs = open(sys.argv[2]).readlines()
out=open(sys.argv[3], 'w') 
for l, ref in zip(oras, refs):
    l = l.stirp().replace('<q>', ' ')
    words = (keywords.keywords(l, words = 50, split=True))
    ref = ref.strip()
    new_words = []
    for word in words:
        flag=True
        ws = word.split()
        for w in ws:
            if not w in ref:
                flag=False
                break
        if flag:
            new_words.append(word)
    out.write(' [SEP] '.join(new_words)+'\n')
