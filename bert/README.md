The code is based on https://github.com/nlpyang/PreSumm.

## Preparation

First, you need to install all the requirements:
```
pip -r install requirements.txt
```

To prepare the CNN/DM data, you can download the preprocessed data from [the PreSumm repo](https://github.com/nlpyang/PreSumm). Then, for each datapoint, add its corresponding guidance signal.

`highligted_sentence_data.py` shows how we create the highlighted sentence data from the preprocessed data.

## Train

To train the model, you can run the following script:
```
bash train.sh DATA_PATH MODEL_PATH LOG_PATH
```
DATA_PATH is the path to your data. MODEL_PATH is the path to where you want to save your model. LOG_PATH is the path to where you want to save your log file.

First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use -visible_gpus -1, after downloading, you could kill the process and rerun the code with multi-GPUs.

Training got stuck: there is an [issue](https://github.com/nlpyang/PreSumm/issues/135) with the PreSumm code that the training may get stuck. One workaround is to reload an intermediate checkpoint and resume the training. 



## Valid/Test
For testing, you can run the following command:

```
bash test.sh DATA_PATH MODEL_PATH RESULT_PATH
```

DATA_PATH is the path to your data. MODEL_PATH is the path to your model. RESULT_PATH is the path to where you want to save your results.
