# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:40:00 2021

@author: CY
"""
import torch
import torch.nn.functional as F
import numpy as np

def per_class_iu(hist: np.ndarray) -> np.array:
    """
    :param hist: bin count matrix of size CxC where C is the number of output classes.
    :return: list of IOU for each class
    """
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def mIOU(c_matrix: np.ndarray) -> float:
    """
    Calculates the mIOU for a given confusion matrix
    :param c_matrix: CxC confusion matrix
    :return: effection mIOU
    """
    if type(c_matrix) != np.ndarray:
        return 0
    class_iu = per_class_iu(c_matrix)
    m_iou = np.nanmean(class_iu)    # ignoring Nans
    return m_iou

def pixel_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    :param output: output logits
    :param target: targets
    :return: pixelwise accuracy for semantic segmentation
    """
    softmax_out = F.softmax(output, dim=1)
    pred = torch.argmax(softmax_out, dim=1).squeeze(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 19]
    correct = correct.view(-1)
    score = correct.float().sum(0) / correct.size(0)
    return score.item()

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    :param outputs: output logits
    :param targets: target labels
    :return: accuracy score for the batch
    """
    softmax_out = F.softmax(outputs, dim=1)
    preds = torch.argmax(softmax_out, dim=1)
    correct = preds.eq(targets)
    score = correct.float().sum(0) / correct.size(0)
    return score.item()