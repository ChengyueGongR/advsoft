import torch
import numpy as np

def partial_shuffle(data):
    # Method from "Partially Shuffling the Training Data to Improve Language Models" by Ofir Press
    # https://arxiv.org/abs/1903.04167
    data = data.t()  
    N = data.shape[0]  # batch size
    M = data.shape[1]  # length of each batch element (or equivalently, row)

    splits = torch.from_numpy(np.random.randint(M, size=N))
    shifted = []
    for i, row in enumerate(data):
        shifted.append(
            torch.cat((row[splits[i]:], row[:splits[i]])) #partial shuffle of a single row
        )
    print('The training data has been partially shuffled!')
    return torch.stack(shifted).t()
