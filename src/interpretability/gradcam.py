import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Implements Grad-CAM for the ResNet backbone.
    
    Grad-CAM highlights regions that most influence the model's decision, 
    not necessarily medically relevant anatomy.
    
    It computes the gradient of the target class score with respect to 
    the feature maps of the last convolutional layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients
        # Use register_full_backward_hook for newer pytorch versions
        self.target_layer.register_full_backward_hook(self.save_gradients)
        # Hook for activations
        self.target_layer.register_forward_hook(self.save_activations)
        
    def save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple, usually (tensor,)
        self.gradients = grad_output[0]
        
    def save_activations(self, module, input, output):
        self.activations = output
        
    def __call__(self, x, class_idx=None):
        """
        Args:
            x (tensor): Input image [1, C, H, W]
            class_idx (int, optional): Target class index. If None, uses predicted class.
        """
        # Forward pass
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
            
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass for target class
        target = logits[0, class_idx]
        target.backward()
        
        # Get pooled gradients (weights)
        # [1, C, H, W] -> [1, C, 1, 1] -> [C]
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        # activations: [1, C, H, W]
        activations = self.activations[0]
        
        # Weighted combination
        # shape [H, W]
        heatmap = torch.zeros(activations.shape[1:], dtype=torch.float32, device=x.device)
        for i, weight in enumerate(pooled_gradients):
            heatmap += weight * activations[i]
            
        # ReLU extraction
        heatmap = F.relu(heatmap)
        
        # Normalize to 0-1
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.detach().cpu().numpy(), class_idx, logits
