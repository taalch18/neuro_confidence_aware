import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.config import SPLITS_DIR, SEED
from src.data.loader import get_dataloaders
from src.models.cnn_backbone import get_model
from src.confidence.temperature_scaling import TemperatureScaler
from src.confidence.confidence_utils import get_msp
from src.confidence.evaluation import compute_calibration_error, compute_coverage_vs_accuracy

def evaluate_metrics():
    """
    These metrics quantify probability reliability and selective prediction behavior. 
    They do not measure clinical correctness.
    """
    device = torch.device("cpu")
    print("Loading resources for evaluation...")
    
    # 1. Load Data (Validation Set)
    # We use batch_size=1 to easily accumulate, or larger for speed
    _, val_loader, _ = get_dataloaders(SPLITS_DIR, batch_size=32)
    
    # 2. Load Model
    model = get_model(pretrained=False)
    checkpoint_path = "checkpoints/best_model.pth"
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.to(device)
    model.eval()
    
    # 3. Get Logits & Labels (Uncalibrated)
    logits_list = []
    labels_list = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            logits_list.append(outputs)
            labels_list.append(labels)
            
    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)
    
    # --- Metrics Before Calibration ---
    conf_before = get_msp(logits)
    probs_before = F.softmax(logits, dim=1)
    preds_before = probs_before.argmax(dim=1)
    
    ece_before = compute_calibration_error(conf_before, preds_before, labels)
    
    # High Conf Error Before (Conf >= 0.7)
    mask_high_before = conf_before >= 0.7
    if mask_high_before.sum() > 0:
        incorrect_high_before = (preds_before[mask_high_before] != labels[mask_high_before]).float().mean().item()
    else:
        incorrect_high_before = 0.0
        
    # --- Calibration (Temperature Scaling) ---
    # We optimize T on the validation set itself (Note: In strict analysis, we'd use a separate calib set)
    # The TemperatureScaler class expects a loader to optimize, but we already have logits.
    # We can reconstruct a simple interface or just optimize directly here for the script.
    # Re-using the class is cleaner as requested.
    
    scaler = TemperatureScaler(model)
    scaler = scaler.to(device)
    
    # Optimize T
    # We need to pass the loader to set_temperature, or we can hack it since we have logits?
    # The class `set_temperature` iterates the loader. Let's let it do its thing.
    print("Optimizing temperature...")
    scaler.set_temperature(val_loader, device)
    
    # Get Calibrated Logits
    # scaler.set_temperature updates self.temperature
    # We can just apply it to our existing logits
    with torch.no_grad():
        logits_after = scaler.temperature_scale(logits)
        
    conf_after = get_msp(logits_after)
    probs_after = F.softmax(logits_after, dim=1)
    preds_after = probs_after.argmax(dim=1) # Predictions shouldn't change, but good to ensure
    
    ece_after = compute_calibration_error(conf_after, preds_after, labels)
    
    # High Conf Error After
    mask_high_after = conf_after >= 0.7
    if mask_high_after.sum() > 0:
        incorrect_high_after = (preds_after[mask_high_after] != labels[mask_high_after]).float().mean().item()
    else:
        incorrect_high_after = 0.0
    # --- Output Task 0: Baseline Accuracy ---
    print("\n" + "="*30)
    print("Baseline Accuracy Evaluation (Validation Set)")
    overall_acc = (preds_before == labels).float().mean().item()
    print(f"Overall Accuracy: {overall_acc:.4f}")
    
    # Per-class accuracy
    from src.config import CLASS_NAMES
    # CLASS_NAMES is {1: 'Meningioma', ...} but labels are 0-indexed (0, 1, 2)
    print("Per-Class Accuracy:")
    for class_idx in range(3):
        mask = labels == class_idx
        if mask.sum() > 0:
            acc = (preds_before[mask] == labels[mask]).float().mean().item()
            class_name = CLASS_NAMES.get(class_idx + 1, f"Class {class_idx}")
            print(f"  {class_name}: {acc:.4f} ({mask.sum().item()} samples)")

    print("\n" + "="*30)
    print("Calibration Results:")
    print(f"ECE (before scaling): {ece_before:.4f}")
    print(f"ECE (after scaling):  {ece_after:.4f}")
    
    # --- Output Task 3: High-Confidence Error ---
    print("\nHigh-Confidence Error Rate:")
    print(f"Before calibration (conf >= 0.7): {incorrect_high_before*100:.2f}%")
    print(f"After calibration  (conf >= 0.7): {incorrect_high_after*100:.2f}%")
    
    # --- Output Task 2: Selective Prediction ---
    # We want coverage at 100%, ~80%, ~60%
    # We can find thresholds dynamically or just test a few.
    # Probing thresholds: 0.0 (100%), 0.5, 0.7, 0.8, 0.9 might give us ranges.
    # Let's simple check a range map.
    
    print("\n" + "="*30)
    print("Selective Prediction Results:")
    
    # 100% Coverage
    res_100 = compute_coverage_vs_accuracy(conf_after, preds_after, labels, thresholds=[0.0])[0.0]
    print(f"Coverage: {res_100['coverage']*100:.0f}% | Accuracy: {res_100['accuracy']:.4f}")
    
    # We want ~80% and ~60%. We can search or just sweep and pick closest.
    sweep_thresholds = np.linspace(0.1, 0.95, 18)
    sweep_results = compute_coverage_vs_accuracy(conf_after, preds_after, labels, thresholds=sweep_thresholds)
    
    # Find closest to 0.8 coverage
    best_80 = None
    dist_80 = 1.0
    
    # Find closest to 0.6 coverage
    best_60 = None
    dist_60 = 1.0
    
    for t, res in sweep_results.items():
        cov = res['coverage']
        if abs(cov - 0.8) < dist_80:
            dist_80 = abs(cov - 0.8)
            best_80 = res
        if abs(cov - 0.6) < dist_60:
            dist_60 = abs(cov - 0.6)
            best_60 = res
            
    if best_80:
        print(f"Coverage:  {best_80['coverage']*100:.0f}% | Selective Accuracy: {best_80['accuracy']:.4f}")
    else:
        print("Coverage:  ~80% | Not reachable with current confidence distribution")
        
    if best_60:
        print(f"Coverage:  {best_60['coverage']*100:.0f}% | Selective Accuracy: {best_60['accuracy']:.4f}")
    else:
        print("Coverage:  ~60% | Not reachable with current confidence distribution")
        
    print("="*30)

if __name__ == "__main__":
    evaluate_metrics()
