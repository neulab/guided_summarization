wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

python -m examples.roberta.multiprocessing_bpe_encoder \
--encoder-json encoder.json \
--vocab-bpe vocab.bpe \
--inputs "$1" \
--outputs "$2" \
--workers 60 \
--keep-empty;
