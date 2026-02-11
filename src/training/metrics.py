import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_metrics(y_true, y_pred):
    """
    Calculates Accuracy, Precision, Recall (Macro)
    y_true: dataset labels [Batch]
    y_pred: model predictions [Batch] (classes, not logits)
    """
    # Move to CPU for sklearn
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec
    }
