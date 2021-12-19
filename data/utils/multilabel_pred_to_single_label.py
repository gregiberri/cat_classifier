import torch
from torch.nn import functional
import numpy as np


def single_object_multilabel_pred_to_single_label(pred, labels_names, positive_threshold):
    pred = torch.nn.functional.sigmoid(pred)

    labels_names = np.array(labels_names)

    # if there are 2 preds over the threshold, then
    max_pred_i = torch.argmax(pred, dim=-1)

    # get the minimum prediction
    min_pred, _ = torch.min(pred, dim=-1)

    single_label_preds = torch.empty(len(pred), dtype=torch.long)
    single_label_preds[max_pred_i == 0] = 0
    single_label_preds[max_pred_i == 1] = 1
    single_label_preds[torch.sum(pred > positive_threshold, dim=-1) > 1] = int(
        np.argwhere(labels_names == 'mindket_cica'))
    single_label_preds[min_pred < positive_threshold] = int(np.argwhere(labels_names == 'egyik_sem'))

    return single_label_preds


def multilabel_pred_to_single_label(pred, labels_names, positive_threshold):
    pred = torch.nn.functional.sigmoid(pred)

    labels_names = np.array(labels_names)

    # if there are 2 preds over the threshold, then
    max_pred_i = torch.argmax(pred, dim=-1)

    # get the minimum prediction
    min_pred, _ = torch.min(pred, dim=-1)

    single_label_preds = torch.empty(len(pred), dtype=torch.long)
    single_label_preds[max_pred_i == 0] = 0
    single_label_preds[max_pred_i == 1] = 1
    single_label_preds[max_pred_i == 2] = 2
    single_label_preds[min_pred < positive_threshold] = int(np.argwhere(labels_names == 'egyik_sem'))

    return single_label_preds
