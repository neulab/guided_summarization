BPE_DIR=$1
BIN_DIR=$2
python fairseq_cli/guided_preprocess.py \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref $1"/train.bpe" \
  --validpref $1"/val.bpe" \
  --destdir $2 \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
