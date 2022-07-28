import torch
import torch.nn as nn


def distillation_loss(logits, soft_logits, T):
    outputs = torch.log_softmax(logits / T, dim=1)
    soft_targets = torch.softmax(soft_logits / T, dim=1)
    loss = - torch.sum(outputs * soft_targets, dim=1, keepdim=False)
    loss = torch.mean(loss, dim=0, keepdim=False)
    return loss


def predictive_entropy(logits):
    log_p = torch.log_softmax(logits, dim=1)
    p = torch.softmax(logits, dim=1)
    H = - torch.sum(p * log_p, dim=1, keepdim=False)
    H = torch.mean(H, dim=0, keepdim=False)
    return H


def DMER(logits, soft_logits, T):
    loss = distillation_loss(logits, soft_logits, T)
    H = predictive_entropy(logits)
    return loss - H


def distillation_bce_loss(logits, soft_logits, T):
    # outputs = torch.log(logits)
    # loss = - torch.sum(outputs * soft_logits, dim=1, keepdim=False)
    # loss = torch.mean(loss, dim=0, keepdim=False)
    return nn.BCELoss()(logits, soft_logits)
