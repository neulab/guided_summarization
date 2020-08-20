mode='train'

if mode == 'train':
    total_num = 144
elif mode == 'valid':
    total_num = 7
elif mode == 'test':
    total_num = 6

import torch


zs=torch.load(f'../{mode}.guidance.pt')


cnt = 0
for i in range(total_num):
    a=torch.load(f'cnndm.{mode}.{i}.bert.pt')
    out_pt=f'guidance_data/cnndm.{mode}.{i}.bert.pt'
    out_l = []
    for j in a:
        keys = ['src', 'tgt', 'src_sent_labels', 'segs', 'clss', 'z', 'src_txt', 'tgt_txt']
        it = {}
        for k in keys:
            if k == 'z':
                it[k] = zs[cnt]
            else:
                it[k] = j[k]
        cnt += 1
    torch.save(out_l, out_pt)
