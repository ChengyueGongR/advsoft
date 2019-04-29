# Language Model Baselines for PyTorch 

This repository contains the code used for the paper:
+ [Improving Neural Language Modeling via Adversarial Training]()

This code was originally forked from the [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm), [MoS-awd-lstm-lm](https://github.com/zihangdai/mos) and [doc-lm](https://github.com/nttcslab-nlp/doc_lm).

Except the method in our paper, we also implement a recent proposed regularization called [PartialShuffle](https://github.com/ofirpress/PartialShuffle). We find that combining this techique with our method can further improve the performance for langauge models.

The model comes with instructions to train:
+ word level language models over the Penn Treebank (PTB), [WikiText-2](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT2), and [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (WT103) datasets. (The code and pre-trained model for WikiText-103 will be merged into the branch soon.)



