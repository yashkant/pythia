# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg

""" Losses are enclosed within nn.Module sub-classes.

    Parameters
    ----------
    pred_score: Tensor of size [N,K] with logits
    target_score: Tensor of size [N,K] with target scores

    With N samples in each batch and K label categories

    Returns
    ----------
    loss: Single valued tensor, normalized only across the batch.
"""


def get_loss_criterion(loss_list):
    """Returns a list of losses to be used.

        Parameter
        ----------
        loss_list: List of the losses supported. Max length is 2.
    """
    loss_criterions = []
    if len(loss_list) == 2:
        if loss_list[0] != 'softmaxKL':
            print('Training with Complement Objective only supports softmaxKL'
                  ' as the primary loss. Current primary loss is: ',
                  loss_list[0])
            # raise NotImplementedError
    elif len(loss_list) > 2:
        raise NotImplementedError

    for loss_config in loss_list:
        if loss_config == 'logitBCE':
            loss_criterion = LogitBinaryCrossEntropy()
        elif loss_config == 'softmaxKL':
            loss_criterion = SoftmaxKlDivLoss()
        elif loss_config == 'wrong':
            loss_criterion = wrong_loss()
        elif loss_config == 'combinedLoss':
            loss_criterion = CombinedLoss()
        elif loss_config == 'complementEntropy':
            loss_criterion = ComplementEntropyLoss()
        else:
            raise NotImplementedError
        loss_criterions.append(loss_criterion)
    return loss_criterions


class LogitBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(LogitBinaryCrossEntropy, self).__init__()

    def forward(self, pred_score, target_score, weights=None):
        loss = F.binary_cross_entropy_with_logits(pred_score,
                                                  target_score,
                                                  size_average=True)
        loss = loss * target_score.size(1)
        print("loss BCE: ", loss)
        return loss


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


def complement_entropy_loss(x, y):
    """ Returns the complement entropy loss as proposed in the report.

        This implementation is faithful with Equation (6) in the report's
        section (2.4).

        Equation (6) talks about the complement entropy, we calculate the
        negative of its value i.e. complement entropy loss.
    """
    # --------------------------------------------------------------------------
    # Negated complement entropy (loss) for each label with zero target score
    # --------------------------------------------------------------------------
    y_is_0 = torch.eq(y, 0)
    # print("y_is_0 shape: ", y_is_0.shape)
    x_remove_0 = x.clone().masked_fill_(y_is_0, 0)
    # print("x_remove_0 shape: ", x_remove_0.shape)
    xr_sum = torch.sum(x_remove_0, dim=1, keepdim=True)
    # print("xr_sum shape: ", xr_sum.shape)
    one_min_xr_sum = 1-xr_sum
    # print("one_min_xr_sum shape: ", one_min_xr_sum.shape)
    one_min_xr_sum.masked_fill_(one_min_xr_sum <= 0, 1e-7)  # Numerical issues
    # print("one_min_xr_sum shape: ", one_min_xr_sum.shape)
    px = x / one_min_xr_sum
    # print("px shape: ", px.shape)
    log_px = torch.log(px + 1e-10)  # Numerical issues
    # print("log_px shape: ", log_px.shape)
    new_x = px * log_px
    # print("new_x shape: ", new_x.shape)
    loss = new_x * (y_is_0.float())  # Remove non-zero labels loss
    # print("loss shape: ", loss.shape)

    # --------------------------------------------------------------------------
    # Normalize the loss to balance it with cross entropy loss
    # --------------------------------------------------------------------------
    num_labels = y.size()[1]
    # print("num_labels shape: ", num_labels)
    zero_labels = torch.sum(y_is_0, dim=1, keepdim=True).float()
    # print("zero_labels: ", zero_labels)
    non_zero_labels = num_labels - zero_labels
    # print("non_zero_labels: ", non_zero_labels)
    # print("labels sum: ", torch.sum(y, 1))
    # zero_labels.masked_fill_(torch.eq(zero_labels, 0), 1e-7)  # num. issues
    # print("zero_labels shape: ", zero_labels.shape)
    # normalize = 1/1000000000000
    normalize = 1 / num_labels
    # print("normalize: ", normalize)
    # zero_labels.masked_fill_(torch.eq(zero_labels, 0), 0)
    # print("zero_labels shape: ", zero_labels.shape)
    loss = loss * normalize
    # print("loss shape: ", loss)
    loss_return = torch.sum(loss, dim=1, keepdim=True)
    # print("loss_return shape:", loss_return.shape)
    return loss_return  # Sum the loss over the labels


def conv_soft_to_hard_onehot_scores(scores):
    hard_scores = torch.zeros_like(scores)
    scores_max, _ = torch.max(scores, dim=1, keepdim=True)
    scores_max_exp = scores_max.expand_as(scores)
    max_bool = torch.eq(scores, scores_max_exp)

    for i, num_scores in enumerate(max_bool):
        nonzero_inds = num_scores.nonzero().squeeze(-1)
        rand_no = torch.LongTensor(1).random_(0, torch.sum(num_scores))
        rand_ind = nonzero_inds[rand_no]
        hard_scores[i][rand_ind] = 1

    return hard_scores


class weighted_softmax_loss(nn.Module):
    def __init__(self):
        super(weighted_softmax_loss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


class ComplementEntropyLoss(nn.Module):
    """ Complement Entropy that maximizes entropy of non-ground truth
        labels. It was proposed to complement the classification loss.

        Paper Link : https://openreview.net/pdf?id=HyM7AiA5YX

        The same approach can be extended to multi-label classification problems
        with Softmax KL divergence loss.

        Report Link :
        https://drive.google.com/file/d/16NtLvZvBPq1cRVeCSq7sXXg0C8NkSi4l/view

        This implementation is faithful with Equation (6) in the report's
        section (2.4).

        All the target scores with non-zero values are treated as positive
        labels while calculating the complement entropy.

        When using Softmax KL divergence loss, predictions corresponding to
        incorrect labels do not directly contribute to the training
        (parameter updates).

        This complementary loss could be used to add an explicit objective for
        maximizing the entropy of the incorrect labels.

        While training, we alternate between the primary and the complement
        objective.
    """

    def __init__(self):
        super(ComplementEntropyLoss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        if cfg.hard_scores:
            tar = conv_soft_to_hard_onehot_scores(tar)

        res = F.softmax(pred_score, dim=1)
        loss = complement_entropy_loss(res, tar)
        # loss = loss*tar_sum
        loss = torch.sum(loss) / loss.size(0)
        # print("loss comp: ", loss)
        return loss


class SoftmaxKlDivLoss(nn.Module):
    def __init__(self):
        super(SoftmaxKlDivLoss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        if cfg.hard_scores:
            tar = conv_soft_to_hard_onehot_scores(tar)

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        # print("loss KL: ", loss)
        return loss


class wrong_loss(nn.Module):
    def __init__(self):
        super(wrong_loss, self).__init__()

    def forward(self, pred_score, target_score):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, size_average=True)
        loss *= target_score.size(1)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

        self.weight_softmax = None
        self.weight_complement = None
        self.weight_complement_decay_factor = None
        self.weight_complement_decay_iters = None

        if cfg.weight_softmax is not None:
            self.weight_softmax = cfg.weight_softmax

        if cfg.weight_complement is not None:
            self.weight_complement = cfg.weight_complement

        if cfg.weight_complement_decay:
            self.weight_complement_decay_factor = cfg.weight_complement_decay_factor
            self.weight_complement_decay_iters = cfg.weight_complement_decay_iters

        print("using softmax weight: ", self.weight_softmax)
        print("using complement weight: ", self.weight_complement)
        print("using complement weight factor: ", self.weight_complement_decay_factor)
        print("using complement weight iters: ", self.weight_complement_decay_iters)

    def forward(self, pred_score, target_score, iter=None):
        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss1 = kl_div(res, tar)
        loss1 = torch.sum(loss1) / loss1.size(0)
        loss = loss1

        if iter is not None and cfg.weight_complement_decay and\
                ((iter+1) % self.weight_complement_decay_iters == 0):
            self.weight_complement *= self.weight_complement_decay_factor
            print("Decaying complement_weight at", iter, "to", self.weight_complement)

        if self.weight_softmax is not None:
            loss2 = F.binary_cross_entropy_with_logits(pred_score,
                                                       target_score,
                                                       size_average=True)
            loss2 *= target_score.size(1)
            loss = self.weight_softmax * loss1 + loss2

        # ----------------------------------------------------------------------
        # Combine complement entropy loss pre-multiplied with a weight
        # ----------------------------------------------------------------------
        if self.weight_complement is not None:
            res = F.softmax(pred_score, dim=1)
            loss3 = complement_entropy_loss(res, tar)
            loss3 = torch.sum(loss3) / loss3.size(0)
            loss += self.weight_complement * loss3

        return loss
