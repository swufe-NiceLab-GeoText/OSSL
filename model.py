import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from S3 import S3
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score as F1
from sklearn.metrics import roc_auc_score as auc

import datetime
from scipy.optimize import linear_sum_assignment
from collections import Counter

def build_NCD(raw_embedding_dim, embedding_dim, hidden_dim, z_dim, num_labels, num_unlabels, pretrained_embedding, device):
    encoder = Encoder(raw_embedding_dim, embedding_dim, hidden_dim, pretrained_embedding, device)
    classifier = Classifier(embedding_dim, z_dim, num_labels, num_unlabels)
    model = NCD(encoder, classifier).to(device)
    simnet = SimNet(embedding_dim * 2, 100, 1)
    return model

# 编码器
class Encoder(nn.Module):
    def __init__(self, raw_embedding_dim, embedding_dim, hidden_dim, pretrained_embedding, device, num_layers = 1):
        super(Encoder, self).__init__()
        # 设置输入参数
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.linear = nn.Linear(raw_embedding_dim, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, padding_idx=0)
        self.S3 = S3(d_model=embedding_dim, d_state=hidden_dim, seqlen_First=True)
    def forward(self, src, lengths):
        embedded = self.linear(self.embedding(src))
        mask = torch.arange(embedded.shape[1]).expand(embedded.shape[0], embedded.shape[1]).to(self.device) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand_as(embedded)
        masked_data = embedded * mask.float()

        out = self.S3(masked_data)
        return out


class Classifier(nn.Module):
    def __init__(self, embedding_dim, z_dim, num_labels, num_unlabels):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, z_dim, bias=False)
        self.head = nn.Linear(z_dim, num_labels+num_unlabels, bias=False)

    def forward(self, src):
        z = self.fc(src)
        out = self.head(F.normalize(z))

        return out

class NCD(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

    def forward(self, src, length):
        encoder_output = self.encoder(src, length)
        out = self.classifier(encoder_output)

        return encoder_output, out

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class SimNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(SimNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.linear2 = MetaLinear(hidden, output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)



