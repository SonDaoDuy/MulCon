import os
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from sklearn.metrics import average_precision_score

def meanAP(gt, pred):
    return average_precision_score(gt, pred)

def precisionAtK(Y_pred_orig, Y_true_orig, k, verbose=False):  # k = 1,3,5
    Y_pred = Y_pred_orig.copy()
    Y_true = Y_true_orig.copy()
    row_sum = np.asarray(Y_true.sum(axis=1)).reshape(-1)
    indices = row_sum.argsort()
    row_sum.sort()
    start = 0
    while start < len(indices) and row_sum[start] == 0:
        start += 1
    indices = indices[start:]
    Y_pred = Y_pred[indices, :]
    Y_true = Y_true[indices, :]
    p = np.zeros(k)
    #print("Y_pred:", Y_pred.shape)
    #print("Y_true:", Y_true.shape)
    assert Y_pred.shape == Y_true.shape
    n_items, n_labels = Y_pred.shape
    prevMatch = 0
    for i in range(1, k + 1):
        Jidx = np.argmax(Y_pred, 1)
        prevMatch += np.sum(Y_true[np.arange(n_items), Jidx])
        Y_pred[np.arange(n_items), Jidx] = -np.inf
        p[i - 1] = prevMatch / (i * n_items)
    return tuple(p[[0, 2, 4]])

def cal_auc(pred, gt, thr=0.5):
    mAP = roc_auc_score(gt, pred, average='macro')
    # # Compute ROC curve and ROC area for each class
    # pred = (pred >= thr).astype(np.int)
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # n_classes = pred.shape[0]
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    return mAP

def average_precision(pred, gt_label, difficult_examples=False):
    pred = torch.from_numpy(pred)
    gt_label = torch.from_numpy(gt_label)
    ap = torch.zeros(pred.size(1))
    for k in range(pred.size(1)):
        # sort scores
        scores = pred[:, k]
        target = gt_label[:, k]
        # sort examples
        sorted_scores, indices = torch.sort(scores, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        ap[k] = precision_at_i
    return ap.mean()
