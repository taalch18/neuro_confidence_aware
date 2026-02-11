import torch
import numpy as np

def compute_coverage_vs_accuracy(confidences, predictions, labels, thresholds=None):
    """
    Computes accuracy at various coverage levels.
    
    Args:
        confidences (torch.Tensor): [N]
        predictions (torch.Tensor): [N]
        labels (torch.Tensor): [N]
        thresholds (list): List of thresholds to evaluate.
        
    Returns:
        dict: {thr: {'coverage': float, 'accuracy': float}}
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1.0, 21)
        
    results = {}
    
    for t in thresholds:
        mask = confidences >= t
        coverage = mask.float().mean().item()
        
        if mask.sum() > 0:
            correct = predictions[mask].eq(labels[mask]).float().sum()
            acc = (correct / mask.sum()).item()
        else:
            acc = 0.0 # Or nan
            
        results[t] = {'coverage': coverage, 'accuracy': acc}
        
    return results

def compute_calibration_error(confidences, predictions, labels, n_bins=15):
    """
    Computes Expected Calibration Error (ECE).
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    accuracies = predictions.eq(labels)
    
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece
