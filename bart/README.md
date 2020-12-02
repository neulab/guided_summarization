The code is based on https://github.com/pytorch/fairseq/tree/master/examples/bart. You can first run the baseline model following https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md to get familiarized with the whole procedure.

## Data Preparation

First, download the preprocessed data from https://github.com/icml-2020-nlp/semsim. Then, for each datapoint, add its corresponding guidance signal (**raw text**, do not tokenize it). Our highlighted sentence data is available at https://drive.google.com/file/d/12SpWwfD3syIxcC-SdSNnDOI5sbXJaylC/view?usp=sharing and the sentence-guided model is available at https://drive.google.com/file/d/1BMKhAh2tG5p8THxugZWMPc7NXqwJDHLw/view?usp=sharing.

Then, BPE all the texts (see z_bpe.sh), including the source side, target side, and guidance signals, and store them into a single directory with the name `train.bpe.source`, `train.bpe.target`, `train.bpe.z`, `val.bpe.source`, `val.bpe.target`, `val.bpe.z`.

Binariza the dataset (see z_bin.sh).
 


## Train

```
bash z_train.sh DATA_PATH MODEL_PATH
```

DATA_PATH is the path to your data. MODEL_PATH is the path to where you want to save your model.



## Valid/Test

```
bash z_test.sh SRC GUIDANCE RESULT_PATH MODEL_DIR MODEL_NAME DATA_BIN
```

An example:

```
bash z_test.sh cnndm.test.src cnndm.test.z cnndm.test.output our_model checkpoint_best.pt cnndm-bin-z
```
