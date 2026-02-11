import torch
import torch.nn.functional as F

def get_msp(logits):
    """
    Computes Maximum Softmax Probability (MSP).
    
    MSP is the standard baseline for confidence. It assumes that the model's 
    highest softmax probability reflects the probability of correctness.
    While simple, MSP is often overconfident for out-of-distribution samples.
    
    Args:
        logits (torch.Tensor): Raw logits [Batch, Classes]
        
    Returns:
        torch.Tensor: Confidence scores [Batch] (0.0 to 1.0)
    """
    probs = F.softmax(logits, dim=1)
    confidence, _ = torch.max(probs, dim=1)
    return confidence

def get_entropy(logits):
    """
    Computes Predictive Entropy (Negative Entropy).
    
    Entropy captures uncertainty across the entire distribution, not just the peak.
    High entropy = high uncertainty (uniform distribution).
    Low entropy = low uncertainty (peaked distribution).
    
    We return negative entropy so that higher values indicate higher 'confidence' (certainty).
    
    Args:
        logits (torch.Tensor): Raw logits [Batch, Classes]
        
    Returns:
        torch.Tensor: Negative entropy scores [Batch]
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    
    # Entropy H(x) = -sum(p * log(p))
    # We want "Confidence" so we return -H(x) = sum(p * log(p))
    # This aligns directionality: higher is better/more certain
    neg_entropy = torch.sum(probs * log_probs, dim=1)
    return neg_entropy
