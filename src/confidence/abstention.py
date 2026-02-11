import torch
import numpy as np

class Abstainer:
    """
    Implements threshold-based abstention.
    
    If confidence < threshold, the model abstains from predicting.
    This allows trading off coverage (fraction of samples predicted) 
    for selective accuracy (accuracy on predicted samples).
    """
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold (float): Confidence threshold (0.0 to 1.0).
        """
        self.threshold = threshold
        
    def predict(self, confidence, predictions):
        """
        Applies abstention logic.
        
        Args:
            confidence (torch.Tensor): Confidence scores [N].
            predictions (torch.Tensor): Model predictions (class indices) [N].
            
        Returns:
            dict: {
                'prediction': Tensor [N] (predictions with -1 for abstained),
                'mask': Tensor [N] (bool, True if accepted),
                'coverage': float (fraction accepted),
                'selective_acc': float (accuracy on accepted, if labels provided elsewhere)
            }
        """
        # Create mask: True if confidence >= threshold
        mask = confidence >= self.threshold
        
        # Filter predictions
        final_preds = predictions.clone()
        final_preds[~mask] = -1 # -1 indicates abstention
        
        coverage = mask.float().mean().item()
        
        return {
            'final_preds': final_preds,
            'mask': mask,
            'coverage': coverage
        }

    def evaluate_selective_acc(self, mask, predictions, labels):
        """
        Computes accuracy only on accepted samples.
        """
        if mask.sum() == 0:
            return 0.0
            
        # Select instances where mask is True
        sel_preds = predictions[mask]
        sel_labels = labels[mask]
        
        correct = sel_preds.eq(sel_labels).float().sum()
        return (correct / len(sel_labels)).item()
