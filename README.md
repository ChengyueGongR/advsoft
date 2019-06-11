# Language Modeling in PyTorch 

This repository contains the code used for the paper:
+ [Improving Neural Language Modeling via Adversarial Training]()

This code was originally forked from the [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm) and [MoS-awd-lstm-lm](https://github.com/zihangdai/mos).

Except the method in our paper, we also implement a recent proposed regularization called [PartialShuffle](https://github.com/ofirpress/PartialShuffle). We find that combining this techique with our method can further improve the performance for langauge models.

The model comes with instructions to train: word level language models over the Penn Treebank (PTB), [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2), and [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT103) datasets. (The code and pre-trained model for WikiText-103 will be merged into the branch soon.)


If you use this code or our results in your research, you can choose to cite:

```
@InProceedings{pmlr-v97-wang19f,
  title = 	 {Improving Neural Language Modeling via Adversarial Training},
  author = 	 {Wang, Dilin and Gong, Chengyue and Liu, Qiang},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {6555--6565},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
}

``` 

### Warning
Also the repo is implemented in `pytorch 0.4`, we have found that the post process can only work well with `pytorch 0.2`. Therefore, we add a patch for dynamic evaluation and it should be run under `pytorch 0.2`.
We are now trying to fix this problem. If you have any idea, feel free to talk with us.

## MoS-AWD-LSTM + Adv + PartialShuffle

Open the folder `mos-awd-lstm-lm` and you can use the MoS-awd-lstm-lm, which can achieve good performance but also cost a lot of time.

### PTB with MoS-AWD-LSTM

We first list the results without dynamic evaluation:

| Method      | Valid PPL     | Test PPL     |
| :---------- | :-----------:  | :-----------: |
| MoS     | 56.54     | 54.44     |
| MoS + PartialShuffle    | 55.89     | 53.92     |
| MoS + Adv     | 55.08     | 52.97     |
| MoS + Adv +  PartialShuffle  | 54.10     |  52.20     |


If you want to use `Adv` only, run the following command:
+ `python3 -u main.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 20.0 --epoch 1000 --nhid 960 --nhidlast 620 --emsize 280 --n_experts 15 --save PTB --single_gpu --switch 200`
+ `python3 -u finetune.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 25.0 --epoch 1000 --nhid 960 --emsize 280 --n_experts 15  --save PATH_TO_FOLDER --single_gpu -gaussian 0 --epsilon 0.028` 
+ `cp PATH_TO_FOLDER/finetune_model.pt PATH_TO_FOLDER/model.pt` and run `python3 -u finetune.py --data data/penn --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 --batch_size 12 --lr 25.0 --epoch 1000 --nhid 960 --emsize 280 --n_experts 15  --save PATH_TO_FOLDER --single_gpu -gaussian 0  --epsilon 0.028` (tiwce)
+ `cp PATH_TO_FOLDER/finetune_model.pt PATH_TO_FOLDER/model.pt` and run `python3 -u finetune.py --data data/penn --dropouti 0.4 --dropoutl 0.5 --dropouth 0.225 --seed 28 --batch_size 12 --lr 25.0 --epoch 1000 --nhid 960 --emsize 280 --n_experts 15  --save PATH_TO_FOLDER --single_gpu -gaussian 0  --epsilon 0.028` 
+ `source search_dy_hyper.sh` to search the hyper-parameter for dynamic evaluation (lambda, epsilon, learning rate) on validation set, and then apply it on test set.

To use PartialShuffle, add a command `--partial`, we try to use PartialShuffle only in the last finetune and get `54.92` / `52.78` (validation / testing). You can download the [pretrained-model](https://drive.google.com/open?id=1w8hF9e-DUGKJPnH9DMtU6G22FOTzFQJJ) along with the log file or train it from scratch.

### WT2 with MoS-AWD-LSTM
If you want to use `Adv` only, Run the following command:
+ `python3 -u main.py --epochs 1000 --data data/wikitext-2 --save WT2 --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --nhidlast 650 --emsize 300 --batch_size 15 --lr 15.0 --dropoutl 0.29 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu --switch 200`
+ `python3 -u finetune.py --epochs 1000 --data data/wikitext-2 --save PATH_TO_FOLDER --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --emsize 300 --batch_size 15 --lr 20.0 --dropoutl 0.29 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu -gaussian 0  --epsilon 0.028` 
+ `cp PATH_TO_FOLDER/finetune_model.pt PATH_TO_FOLDER/model.pt` and run `python3 -u finetune.py --epochs 1000 --data data/wikitext-2 --save PATH_TO_FOLDER --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --emsize 300 --batch_size 15 --lr 20.0 --dropoutl 0.29 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu -gaussian 0  --epsilon 0.028` (twice)
+ `cp PATH_TO_FOLDER/finetune_model.pt PATH_TO_FOLDER/model.pt` and run `python3 -u finetune.py --epochs 1000 --data data/wikitext-2 --save PATH_TO_FOLDER --dropouth 0.2 --seed 1882 --n_experts 15 --nhid 1150 --emsize 300 --batch_size 15 --lr 20.0 --dropoutl 0.5 --small_batch_size 5 --max_seq_len_delta 20 --dropouti 0.55 --single_gpu -gaussian 0  --epsilon 0.028` 
+ `source search_dy_hyper.sh` to search the hyper-parameter for dynamic evaluation (lambda, epsilon, learning rate) on validation set, and then apply it on test set.

To use PartialShuffle, add a command `--partial`.

## AWD-LSTM-LM + Adv 

Open the folder `awd-lstm-lm` and you can use the awd-lstm-lm, which can achieve good performance and cost less time.

### PTB with AWD-LSTM  

Run the following command:
+ `nohup python3 -u main.py --nonmono 5 --batch_size 20 --data data/penn --dropouti 0.3 --dropouth 0.25 --dropout 0.40 --alpha 2 --beta 1 --seed 141 --epoch 4000 --save ptb.pt --switch 200 >> ptb.log 2>&1 &`
+ `source search_dy_hyper.sh` to search the hyper-parameter for dynamic evaluation (lambda, epsilon, learning rate) on validation set, and then apply it on test set.

You can download the [pretrained-model]() along with the log file or train it from scratch.

### WT2 with AWD-LSTM
Run the following command:
+ `nohup python3 -u main.py --epochs 4000 --nonmono 5 --emsize 400 --batch_size 80 --dropouti 0.5 --data data/wikitext-2 --dropouth 0.2 --seed 1882 --save wt2.pt --gaussian 0.175 --switch 200 >> wt2.log  2>&1 &`
+ `source search_dy_hyper.sh` to search the hyper-parameter for dynamic evaluation (lambda, epsilon, learning rate) on validation set, and then apply it on test set.

You can download the [pretrained-model]() along with the log file or train it from scratch.

