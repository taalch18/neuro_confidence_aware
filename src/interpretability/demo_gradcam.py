import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from pathlib import Path
from src.config import SPLITS_DIR, SEED, CLASS_NAMES
from src.data.loader import get_dataloaders
from src.models.cnn_backbone import get_model
from src.interpretability.gradcam import GradCAM
from src.interpretability.visualize import overlay_heatmap, save_visualization
from src.confidence.confidence_utils import get_msp

def run_demo(checkpoint_path, save_dir):
    device = torch.device("cpu") # User requested CPU compatible
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    print("Loading model...")
    model = get_model(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Initialize GradCAM (Layer 4 is the last resnet block)
    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    
    # Load Data (Use Test Set)
    print("Loading test data...")
    _, _, test_loader = get_dataloaders(SPLITS_DIR, batch_size=1) # bs=1 for easy iteration
    
    print("Searching for interesting samples...")
    count = 0
    max_samples = 5
    
    # Collect samples: 2 Correct High Conf, 2 Incorrect High Conf, 1 Low Conf
    # Since we can't easily validata correct/incorrect without running many, we'll just pick first 5
    # and annotate them.
    
    for i, (image, label) in enumerate(test_loader):
        if count >= max_samples:
            break
            
        image = image.to(device)
        # Forward
        heatmap, pred_idx, logits = gradcam(image)
        
        # Confidence
        conf = get_msp(logits).item()
        prob = F.softmax(logits, dim=1)[0, pred_idx].item()
        
        # Overlay
        overlay = overlay_heatmap(image[0], heatmap)
        
        # Save
        label_name = CLASS_NAMES[label.item()+1]
        pred_name = CLASS_NAMES[pred_idx+1]
        status = "CORRECT" if label.item() == pred_idx else "WRONG"
        
        filename = f"sample_{i}_L{label_name}_P{pred_name}_{status}_Conf{conf:.2f}.png"
        save_path = save_dir / filename
        
        save_visualization(image[0], overlay, save_path)
        print(f"Saved {filename}")
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="gradcam_results")
    args = parser.parse_args()
    
    run_demo(args.checkpoint, args.save_dir)
