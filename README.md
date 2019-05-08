# Language Model Baselines for PyTorch 

This repository contains the code used for the paper:
+ [Improving Neural Language Modeling via Adversarial Training]()

This code was originally forked from the [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm), [MoS-awd-lstm-lm](https://github.com/zihangdai/mos) and [doc-lm](https://github.com/nttcslab-nlp/doc_lm).

Except the method in our paper, we also implement a recent proposed regularization called [PartialShuffle](https://github.com/ofirpress/PartialShuffle). We find that combining this techique with our method can further improve the performance for langauge models.

The model comes with instructions to train: word level language models over the Penn Treebank (PTB), [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2), and [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT103) datasets. (The code and pre-trained model for WikiText-103 will be merged into the branch soon.)

We only focus on training lanaguage models on small scale datasets in this repository. We think that 
+ Building a better density estimation itself should have scientific values.
+ If a method can outperform very strong language model baselines on small scale datasets, it is more likely to perform better on large scale datasets. This repository can help researchers to quickly verify their ideas on small scale datasets.
+ Language models on large scale datasets can  be used to perform unsupervised feature learning, and the techiques included in this repository all only depends on the SoftMax and Embedding Layers. That is to say, you can transfer these techiques with different NNs (e.g. transformer, CNNs, RNNs, etc).
+ Language generation tasks, e.g. seq2seq, are different from language modeling. However, it also depends on the representation of the embeddings and softmax weights. Again, the techiques included in this repository all focus on the representation of embeddings and softmax weights. Some of them (e.g. MoS, Adversarial) has been proved to be useful for machine translation tasks.

If you use this code or our results in your research, you can cite:

```

``` 
.

## DoC-LM + Adv + PartialShuffle

Open the folder `doc-lm` and you can use the doc-lm, which can achieve best performance but cost a lot of time. 
Since it will take much time to train DoC from stratch, we download the [pretrained-model](https://github.com/nttcslab-nlp/doc_lm) and finetune the model once. 

### DoC with AWD-LSTM on PTB

We list the results without dynamic evaluation:

| Method      | Valid PPL     | Test PPL     |
| :---------- | :-----------:  | :-----------: |
| DoC     | 54.18     | 52.38     |
| DoC + PartialShuffle    | 53.85     | 52.10     |
| DoC + Adv +  PartialShuffle    | 53.52     | 51.82     |

Using PartialShuffle only, run the following comands and you can get approximately `53.85` / `52.10` (validation / testing).

Using Adversarial Training and PartialShuffle, run the following comands and you can get approximately `53.68` / `51.83` (validation / testing). [pretrained-model]()

### DoC with AWD-LSTM on WT2

Using PartialShuffle only, run the following comands and you can get approximately `53.85` / `52.10` (validation / testing).

Using Adversarial Training and PartialShuffle, run the following comands and you can get approximately `53.85` / `52.10` (validation / testing).

## MoS-AWD-LSTM + Adv + PartialShuffle

Open the folder `mos-awd-lstm-lm` and you can use the MoS-awd-lstm-lm, which can achieve good performance but also cost a lot of time.

### PTB with MoS-AWD-LSTM

We first list the results without dynamic evaluation:

| Method      | Valid PPL     | Test PPL     |
| :---------- | :-----------:  | :-----------: |
| MoS     | 56.54     | 54.44     |
| MoS + PartialShuffle    | 55.89     | 53.92     |
| MoS + Adv     | 55.08     | 52.97     |
| MoS + Adv +  PartialShuffle    | 54.66     | 52.62     |


If you want to use `Adv` only, run the following command:
+ `python3 -u main.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 1000 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15 --save PTB --single_gpu --switch 200`
+ `python3 -u finetune.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 25.0 --epoch 1000 --nhid 960 --emsize 280 --n_experts 15  --save PATH_TO_FOLDER --single_gpu -gaussian 0 --epsilon 0.028` 
+ `cp PATH_TO_FOLDER/finetune_model.pt PATH_TO_FOLDER/model.pt` and run `python3 -u finetune.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 25.0 --epoch 1000 --nhid 960 --emsize 280 --n_experts 15  --save PATH_TO_FOLDER --single_gpu -gaussian 0  --epsilon 0.028` (for three times)
+ `source search_dy_hyper.sh` to search the hyper-parameter for dynamic evaluation (lambda, epsilon, learning rate) on validation set, and then apply it on test set.

To use PartialShuffle, add a command `--partial`, we try to use PartialShuffle only in the last finetune and get `54.92` / `52.78` (validation / testing). You can download the [pretrained-model]() or train it from scratch.

### WT2 with MoS-AWD-LSTM
If you want to use `Adv` only, Run the following command:
+ `python3 -u main.py --epochs 1000 --data data/wikitext-2 --save WT2 --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --nhidlast 650 --emsize 300 --batch_size 15 --lr 15.0 --dropoutl 0.29 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu --switch 200`
+ `python3 -u finetune.py --epochs 1000 --data data/wikitext-2 --save PATH_TO_FOLDER --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --emsize 300 --batch_size 15 --lr 20.0 --dropoutl 0.29 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu -gaussian 0  --epsilon 0.028` 
+ `cp PATH_TO_FOLDER/finetune_model.pt PATH_TO_FOLDER/model.pt` and run `python3 -u finetune.py --epochs 1000 --data data/wikitext-2 --save PATH_TO_FOLDER --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --emsize 300 --batch_size 15 --lr 20.0 --dropoutl 0.29 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu -gaussian 0  --epsilon 0.028` (for three times)
+ `source search_dy_hyper.sh` to search the hyper-parameter for dynamic evaluation (lambda, epsilon, learning rate) on validation set, and then apply it on test set.

To use PartialShuffle, add a command `--partial`.

## AWD-LSTM-LM + Adv 

Open the folder `awd-lstm-lm` and you can use the awd-lstm-lm, which can achieve good performance and cost less time.

### PTB with AWD-LSTM  

Run the following command:
+ `nohup python3 -u main.py --nonmono 5 --batch_size 20 --data data/penn --dropouti 0.3 --dropouth 0.25 --dropout 0.40 --alpha 2 --beta 1 --seed 141 --epoch 4000 --save ptb.pt --switch 200 >> ptb.log 2>&1 &`
+ `source search_dy_hyper.sh` to search the hyper-parameter for dynamic evaluation (lambda, epsilon, learning rate) on validation set, and then apply it on test set.

You can download the [pretrained-model]() or train it from scratch.

### WT2 with AWD-LSTM
Run the following command:
+ `nohup python3 -u main.py --epochs 4000 --nonmono 5 --emsize 400 --batch_size 80 --dropouti 0.5 --data data/wikitext-2 --dropouth 0.2 --seed 1882 --save wt2.pt --gaussian 0.175 --switch 200 >> wt2.log  2>&1 &`
+ `source search_dy_hyper.sh` to search the hyper-parameter for dynamic evaluation (lambda, epsilon, learning rate) on validation set, and then apply it on test set.

You can download the [pretrained-model]() or train it from scratch.

