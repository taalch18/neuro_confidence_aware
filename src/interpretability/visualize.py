import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_heatmap(img_tensor, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays a heatmap on an original image.
    
    Args:
        img_tensor (tensor): [3, H, W] normalized tensor
        heatmap (numpy array): [H, W] single channel (0-1)
        alpha (float): Transparency of heatmap
        
    Returns:
        numpy array: [H, W, 3] RGB image with overlay (0-255 uint8)
    """
    # Denormalize image tensor
    # Assuming ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    # [3, H, W] -> [H, W, 3]
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Resize heatmap to match image using cv2
    heatmap = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    
    # Scale heatmap to 0-255
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Overlay
    # cv2 uses BGR, we need RGB for saving via PIL or plotting
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay

def save_visualization(original, heatmap_overlay, save_path):
    """
    Saves the visualization side-by-side (Original | GradCAM).
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Prepare original
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = original.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    ax[0].imshow(img)
    ax[0].set_title("Original (Normalized)")
    ax[0].axis('off')
    
    ax[1].imshow(heatmap_overlay)
    ax[1].set_title("Grad-CAM Overlay")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
