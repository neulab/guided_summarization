import torch
from fairseq.models.bart import GuidedBARTModel

import sys
bart = GuidedBARTModel.from_pretrained(
    sys.argv[4],
    checkpoint_file=sys.argv[5],
    data_name_or_path=sys.argv[6]
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 16

with open(sys.argv[1]) as source, open(sys.argv[2]) as zs, open(sys.argv[3], 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    zline = zs.readline().strip()
    zlines = [zline]
    for sline, zline in zip(source, zs):
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, zlines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3, guided=True)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []
            zlines = []

        slines.append(sline.strip())
        zlines.append(zline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, zlines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3, guided=True)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
