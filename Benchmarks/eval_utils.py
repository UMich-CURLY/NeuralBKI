import torch
import numpy as np

# Intersection, union for one frame
def iou_one_frame(pred, target, n_classes=21):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection[cls] = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
        union[cls] = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection[cls]
    return intersection, union

def geo_complete_score(pred, target, empty_class_idx=20):
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    nonempty_preds  = pred != empty_class_idx
    nonempty_target = target != empty_class_idx

    TP = np.sum((nonempty_preds == 1) & (nonempty_target == 1))
    FP = np.sum((nonempty_preds == 1) & (nonempty_target == 0))
    TN = np.sum((nonempty_preds == 0) & (nonempty_target == 0))
    FN = np.sum((nonempty_preds == 0) & (nonempty_target == 1))
 
    precision, recall, iou = TP / (TP + FP), \
        TP / (TP + FN), \
        TP / (TP + FP + FN)
    return precision, recall, iou
