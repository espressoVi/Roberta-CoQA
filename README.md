# Semantic Faithfullness of Language Models
Code generating results for this [paper](https://arxiv.org/abs/2212.10696).
Implementation of Intervention Based Training for conversational question answering (CoQA dataset) on bert.

## Features:
1) Train on the original dataset
2) Train on the combined O, TS, TS-R dataset
3) Evaluation on O,TS, TS-R
4) Attention metrics eta, p_sep, p_rationale (blockwise)

## Usage:
#### Clone this repository
#### Install the necessary requirements 
from the environment.yml file by entering
```
conda env create -f environment.yml
conda activate RobustProject
```
#### Download
The coqa dataset is to be downloaded and place in ./data, files can be downloaded from 

https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json  and
https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json

You should have the files 
```
./data/coqa-dev-v1.0.json
./data/coqa-train-v1.0.json
```
#### Running
```python main.py --train [O|C] --eval [O|TS|RG] --output [directory]```

--train O for original training
--train C for combined training
--eval O for eval on original dataset
--eval TS for eval on TS dataset
--eval RG for eval on TS-R dataset
--output is the ouput directory for saving/loading weights and saving predictions and logs
e.g.
```
python main.py --train O --eval O --output Roberta_orig (original training and eval on O)
python main.py --eval TS --output Roberta_orig (evaluate the model at XLNet_orig on TS dataset)
python main.py --train C --eval O --output Roberta_orig (combined training and eval on O)
```

#### Evaluation

Following eval you will have a predictions.json file at the provided directory. Then run
```
python evaluate-v1.0.py --data-file data/coqa-dev-v1.0.json --pred-file [directory]/predictions.json
```
#### Repeat
The previous two steps needs to be repeated for original and combined training and evaluation on all datasets.

#### run attention-block.py with
```python attention-block.py --output [directory] ```

To get blockwise rationale statistics of the model stored at directory. Output is stored in [directory].

#### run attention-qr.py with
```python attention-qr.py --output [directory] ```

To get eta and p_sep statistics of the model stored at directory. Output is stored in [directory]
