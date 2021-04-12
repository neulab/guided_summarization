The code is based on https://github.com/pytorch/fairseq/tree/master/examples/bart. You can first run the baseline model following https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md to get familiarized with the whole procedure.

## Data Preparation

First, you can download the preprocessed data from https://github.com/icml-2020-nlp/semsim. 

Then, for each datapoint, you can add its corresponding guidance signal (**raw text**, do not tokenize it). Our highlighted sentence data is available at [this link](https://drive.google.com/file/d/12SpWwfD3syIxcC-SdSNnDOI5sbXJaylC/view?usp=sharing).

Then, you can BPE all the texts (see `z_bpe.sh`), including the source side, target side, and guidance signals. Note that you'll need to store them into a single directory with the name `train.bpe.source`, `train.bpe.target`, `train.bpe.z`, `val.bpe.source`, `val.bpe.target`, `val.bpe.z`.
```
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
INPUT=/path/to/the/input/file
OUTPUT=/path/to/the/output/file
python -m examples.roberta.multiprocessing_bpe_encoder \
--encoder-json encoder.json \
--vocab-bpe vocab.bpe \
--inputs "$INPUT" \
--outputs "$OUTPUT" \
--workers 60 \
--keep-empty;
```

Then, Binariza the dataset (see `z_bin.sh`).
```
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
```
 
## Train
The command for training is:

```
bash z_train.sh DATA_PATH MODEL_PATH
```

DATA_PATH is the path to your data. MODEL_PATH is the path to where you want to save your model.

Our trained sentence-guided model is available at [this link](https://drive.google.com/file/d/1BMKhAh2tG5p8THxugZWMPc7NXqwJDHLw/view?usp=sharing).


## Valid/Test
The command for generating the output is:
```
bash z_test.sh SRC GUIDANCE RESULT_PATH MODEL_DIR MODEL_NAME DATA_BIN
```

An example:

```
bash z_test.sh cnndm.test.src cnndm.test.z cnndm.test.output our_model checkpoint_best.pt cnndm-bin-z
```

For computing ROUGE, you'll need to install [files2rouge](https://github.com/pltrdy/files2rouge/tree/b0979655bbc32b65641e69840c88a2aede5e10a2) and the command is:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

cat test.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.tokenized
cat test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.target
files2rouge test.hypo.tokenized test.hypo.target
```
Note that you need to correctly install [ROUGE-1.5.5](https://github.com/summanlp/evaluation/tree/master/ROUGE-RELEASE-1.5.5).
