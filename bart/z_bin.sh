BPE_DIR=/path/to/the/BPE_input/directory
BIN_DIR=/path/to/the/output/directory
python fairseq_cli/guided_preprocess.py \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref $BPE_DIR"/train.bpe" \
  --validpref $BPE_DIR"/val.bpe" \
  --destdir $BIN_DIR \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
