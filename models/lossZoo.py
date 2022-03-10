#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from torch.nn import Softmax
import torch.nn as nn

def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 


def im(outputs_test, gent=True):
    epsilon = 1e-10
    softmax_out = nn.Softmax(dim=1)(outputs_test)
    entropy_loss = torch.mean(entropy(softmax_out))
    if gent:
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax * torch.log2(msoftmax + epsilon))
        entropy_loss -= gentropy_loss
    im_loss = entropy_loss * 1.0
    return im_loss


def adv(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(features.device)
    return torch.nn.BCELoss()(ad_out, dc_target)


def adv_local(features, ad_net, is_source=False, weights=None):
    ad_out = ad_net(features).squeeze(3)
    batch_size = ad_out.size(0)
    num_heads = ad_out.size(1)
    seq_len = ad_out.size(2)
    
    if is_source:
        label = torch.from_numpy(np.array([[[1]*seq_len]*num_heads] * batch_size)).float().to(features.device)
    else:
        label = torch.from_numpy(np.array([[[0]*seq_len]*num_heads] * batch_size)).float().to(features.device)

    return ad_out, torch.nn.BCELoss()(ad_out, label)
