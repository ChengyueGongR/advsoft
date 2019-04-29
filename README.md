# Language Model Baselines for PyTorch 

This repository contains the code used for the paper:
+ [Improving Neural Language Modeling via Adversarial Training]()

This code was originally forked from the [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm), [MoS-awd-lstm-lm](https://github.com/zihangdai/mos) and [doc-lm](https://github.com/nttcslab-nlp/doc_lm).

Except the method in our paper, we also implement a recent proposed regularization called [PartialShuffle](https://github.com/ofirpress/PartialShuffle). We find that combining this techique with our method can further improve the performance for langauge models.

The model comes with instructions to train: word level language models over the Penn Treebank (PTB), [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2), and [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT103) datasets. (The code and pre-trained model for WikiText-103 will be merged into the branch soon.)

Although seq2seq tasks and training language models on large scale datasets seems more important, we only focus on training lanaguage models on small scale datasets in this repository. We think that 
+ Building a better density estimation itself should have scientific values.
+ If a method can outperform very strong language model baselines on small scale datasets, it is more likely to perform better on large scale datasets. This repository can help researchers to quickly verify their ideas on small scale datasets.
+ Language models on large scale datasets can  be used to perform unsupervised feature learning, and the techiques included in this repository all only depends on the SoftMax and Embedding Layers. That is to say, you can transfer these techiques with different NNs (e.g. transformer, CNNs, RNNs, etc).
+ Language generation tasks, e.g. seq2seq, are different from language modeling. However, it also depends on the representation of the embeddings and softmax weights. Again, the techiques included in this repository all focus on the representation of embeddings and softmax weights. Some of them (e.g. MoS, Adversarial) has been proved to be useful for machine translation tasks.



