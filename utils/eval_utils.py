import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, recall_score, roc_auc_score


def eval_model(labels: np.ndarray, pred: np.ndarray, time: str = "", file_name: str = "", save=True):
    AUROC = roc_auc_score(labels, pred)
    print(f"AUROC: {AUROC}")
    pred = pred.copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    f1_micro = f1_score(labels, pred, average='micro')
    f1_macro = f1_score(labels, pred, average='macro')
    f1 = f1_score(labels, pred, average="binary")
    recall = recall_score(labels, pred)
    g_mean = geometric_mean_score(labels, pred)
    true_list = labels == pred
    acc = np.sum(true_list) / labels.shape[0]
    anomaly_true = (labels == 1) & (pred == 1)
    print("predict anomaly sum: ", np.sum(pred == 1), "anomaly predict true", np.sum(anomaly_true))
    print(f"auroc: {AUROC} f1 score: {f1}, macro: {f1_macro}")
    if save or file_name != "":
        file_dir = f'../results/{file_name}.txt'
        f = open(file_dir, 'a+')
        f.write(
            f"AUROC: {AUROC}\tF1-Macro: {f1_macro}\tgmean: {g_mean}\ttime:{time}\n")
        f.close()
        print(f'save to file name: {file_name}')
    return f1_micro, f1_macro, recall, g_mean
