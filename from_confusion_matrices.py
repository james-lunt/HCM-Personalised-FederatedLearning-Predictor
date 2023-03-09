import numpy as np
import pickle
with open("variables/log.pkl", "rb") as f:
    data = pickle.load(f)

confusion_mat = data['final_confusion_matrices']

def metrics_from_confusion_matrices(confusion_matrices):
    # a list of confusion matrices and get back accuracy
    acc_list, precision, recall, f1_scores = [], [], [], []
    for cm in confusion_matrices:
        tn, fp, fn, tp = cm.ravel()
        acc = (tn+tp) / (tn+fp+fn+tp) #AUC
        pre = tp / (tp+fp) if tp+fp != 0 else 0
        rec = tp / (tp+fn) if tp+fn != 0 else 0
        f1 = 2*(pre*rec/(pre+rec)) if pre+rec != 0 else 0

        acc_list.append(acc)
        precision.append(pre)
        recall.append(rec)
        f1_scores.append(f1)

    return acc_list, precision, recall, f1_scores

