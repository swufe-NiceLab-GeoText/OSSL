import torch
import torch.nn.functional as F
import numpy as np
def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.sum((input1 - input2)**2)

def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = 1e-8)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

def get_dataset_args(dataset):
    map_size, input_size, output_size, SOS_token, EOS_token = 0, 0, 0, 0, 0
    if dataset == 'porto':
        map_size = (51, 158)
        input_size = 51 * 158 + 1
        output_size = 51 * 158 + 3
        SOS_token = 8059
        EOS_token = 8060
    elif dataset == 'chengdu':
        map_size = (117, 129)
        input_size = 117 * 129 + 1
        output_size = 117 * 129 + 3
        SOS_token = 15094
        EOS_token = 15095
    else:
        raise ValueError("get_dataset_args function get wrong value")

    return map_size, input_size, output_size, SOS_token, EOS_token
