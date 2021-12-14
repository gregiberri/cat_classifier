import torch
import numpy as np

def multilabel_pred_to_single_label(pred, labels_names, positive_threshold):
    labels_names = np.array(labels_names)

    # the pred with the maximal prediction is 1
    best_pred_max = torch.argmax(pred, dim=-1)
    best_pred_max[torch.sum(pred > positive_threshold, dim=-1) > 1] = 2

    single_label_preds = torch.empty(len(pred), dtype=torch.long)
    for i in range(len(best_pred_max)):
        if best_pred_max[i] == 0:
            single_label_preds[i] = int(np.argwhere(labels_names == 'mindket_cica'))
        elif best_pred_max[i] == 1:
            single_label_preds[i] = int(np.argwhere(labels_names == 'pali'))
        else:
            single_label_preds[i] = int(np.argwhere(labels_names == 'mindket_cica'))

    return single_label_preds
