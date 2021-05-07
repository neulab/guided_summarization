# A script to prune the relations obtained by Stanford OpenIE
import sys
lines=open(sys.argv[1]).readlines()
out=open(sys.argv[2], 'w')

def decom(r):
    s, v, o = r.split('|||')
    s = s.split('#')[0]
    v = v.split('#')[0]
    o = o.split('#')[0]
    
    return set(' '.join([s, v, o]).split())
    s = s.split(',')
    v = v.split(',')
    o = o.split(',')
    s = [int(x) for x in s]
    v = [int(x) for x in v]
    o = [int(x) for x in o]
    return s, v, o




def compare(r1, r2):
    if r2.issubset(r1):
        return -1
    elif r1.issubset(r2):
        return 1
    else:
        return 0

    

    prev = -2
    for x1, x2 in zip(r1, r2):
        if x1 == x2:
            prev = -2
            continue
        elif x2 in x1:
            if prev == -2:
                prev = 1
            elif prev == 1:
                continue
            else:
                prev = 0
                break
        elif x1 in x2:
            if prev == -2:
                prev = -1
            elif prev == -1:
                continue
            else:
                prev = 0
                break
        else:
            prev = 0
            break

    return prev
    

o_c=a_c=0
for l in lines:
    if l == '\n':
        out.write('\n')
        continue
    l = l.strip().split('\t')
    new_rels=[]
    o_c+=len(l)
    new_decoms=[]
    for rel in l:
        flag=False
        rel_decom = decom(rel)
        for i, exist_rel in enumerate(new_rels):
            exist_rel_decom = new_decoms[i]
            comp = compare(rel_decom, exist_rel_decom) #exist_rel > rel
            if comp == -1: #exist_rel > rel
                new_rels[i] = rel
                new_decoms[i] = rel_decom
                flag=True
                break
            elif comp == 1: #rel >
                flag=True
                break
        if not flag:
            new_rels.append(rel)
            new_decoms.append(rel_decom)
    out.write('\t'.join(new_rels)+'\n')
    a_c += len(new_rels)
