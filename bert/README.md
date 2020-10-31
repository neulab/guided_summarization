The code is based on https://github.com/nlpyang/PreSumm.

## Data Preparation

First, download the preprocessed data from https://github.com/nlpyang/PreSumm. Then, for each datapoint, add its corresponding guidance signal (`example_add_guidance.py` is one example of how to do this and `highligted_sentence_data.py` shows how we create the highlighted sentence data from the preprocessed data).

## Train

Install all the requirements:
```
pip -r install requirements.txt

```

Then, train the mode:
```
bash train.sh DATA_PATH MODEL_PATH LOG_PATH
```
DATA_PATH is the path to your data. MODEL_PATH is the path to where you want to save your model. LOG_PATH is the path to where you want to save your log file.

First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use -visible_gpus -1, after downloading, you could kill the process and rerun the code with multi-GPUs.


## Valid/Test

```
bash test.sh DATA_PATH MODEL_PATH RESULT_PATH
```

DATA_PATH is the path to your data. MODEL_PATH is the path to your model. RESULT_PATH is the path to where you want to save your results.
