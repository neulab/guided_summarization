
import torch
import sys

OUTPUT_DIR=sys.argv[1]
mode=sys.artv[2]


if mode == 'train':
    total_num = 144
elif mode == 'valid':
    total_num = 7
elif mode == 'test':
    total_num = 6


cnt = 0
for i in range(total_num):
    a=torch.load(f'cnndm.{mode}.{i}.bert.pt')
    out_pt=f'{OUTPUT_DIR}/cnndm.{mode}.{i}.bert.pt'
    out_l = []
    for j in a:
        keys = ['src', 'tgt', 'src_sent_labels', 'segs', 'clss', 'z', 'src_txt', 'tgt_txt']
        it = {}
        for k in keys:
            if k == 'z':
                src = j['src']
                clss = j['clss']
                z = []
                for sent_id, v in enumerate(j['src_sent_labels']):
                    if v == 1:
                        sent_id = int(float(sent_id))
                        start = clss[sent_id]
                        end = clss[sent_id+1] if not sent_id == len(clss) - 1 else None
                        tokens = src[start:end] if not end is None else src[start:]
                        z.extend(tokens)
                it[k] = z
            else:
                it[k] = j[k]
        out_l.append(it)
    torch.save(out_l, out_pt)
