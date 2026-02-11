
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.config import SPLITS_DIR, CLASS_NAMES
    from src.data.loader import get_dataloaders
    from src.models.cnn_backbone import get_model
    from src.confidence.temperature_scaling import TemperatureScaler
    from src.confidence.confidence_utils import get_msp
    from src.confidence.evaluation import compute_calibration_error, compute_coverage_vs_accuracy
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def generate_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check Checkpoint (Relative to script root: brain_tumor_confidence/)
    # script is in brain_tumor_confidence/scripts/
    # we want brain_tumor_confidence/checkpoints/best_model.pth
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    ckpt_path = root_dir / "checkpoints" / "best_model.pth"
    
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found at {ckpt_path}. Cannot generate plots.")
        return

    # Check Data
    if not Path(SPLITS_DIR / "val.csv").exists():
        print(f"❌ Splits not found. Cannot load data.")
        return
        
    # Load Model & Scaler
    print("Loading model...")
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    
    scaler = TemperatureScaler(model).to(device)
    
    # Load Data (Validation)
    print("Loading validation set...")
    _, val_loader, _ = get_dataloaders(SPLITS_DIR, batch_size=32)
    scaler.set_temperature(val_loader, device) # Calibrate
    
    # Get Logits
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = scaler(images.to(device)) # Calibrated logits
            logits_list.append(outputs)
            labels_list.append(labels.to(device))
            
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, 1)
    
    # --- Plot 1: Reliability / Calibration Curve ---
    print("Generating Reliability Curve...")
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    acc_in_bin = []
    conf_in_bin = []
    prop_in_bin = []
    
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
        prop = in_bin.float().mean().item()
        if prop > 0:
            acc = (predictions[in_bin] == labels[in_bin]).float().mean().item()
            conf = confidences[in_bin].mean().item()
            acc_in_bin.append(acc)
            conf_in_bin.append(conf)
            prop_in_bin.append(prop)
        else:
            acc_in_bin.append(None) # Empty bin
            conf_in_bin.append(None)
            prop_in_bin.append(0)
            
    # Remove Nones for plotting
    plot_acc = [x for x in acc_in_bin if x is not None]
    plot_conf = [x for x in conf_in_bin if x is not None]
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(plot_conf, plot_acc, "s-", label="Model (Post-Scaling)")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_dir / "reliability_curve.png", dpi=300)
    plt.close()
    print(f"✅ Saved reliability_curve.png")
    
    # --- Plot 2: Coverage vs Accuracy (Risk-Coverage) ---
    print("Generating Coverage vs Accuracy Plot...")
    thresholds = np.linspace(0.0, 0.99, 100)
    coverages = []
    accuracies = []
    
    for t in thresholds:
        mask = confidences >= t
        cov = mask.float().mean().item()
        if mask.sum() > 0:
            acc = (predictions[mask] == labels[mask]).float().mean().item()
        else:
            acc = 1.0 # Or nan, usually treated as perfect if no predictions made? Or abstain. Let's use last valid acc.
            # Actually standard is separate curve.
        coverages.append(cov)
        accuracies.append(acc)
        
    plt.figure(figsize=(8, 5))
    plt.plot(coverages, accuracies, "b-", linewidth=2)
    plt.xlabel("Coverage (Proportion of Dataset Predicted)")
    plt.ylabel("Selective Accuracy")
    plt.title("Risk-Coverage Curve")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0.4, 1.05) # Zoom in on the high accuracy region
    # Invert x axis? Usually Coverage is 1 -> 0
    plt.gca().invert_xaxis()
    plt.axvline(x=0.8, color='r', linestyle='--', label="80% Coverage Target")
    plt.legend()
    plt.savefig(output_dir / "coverage_vs_accuracy.png", dpi=300)
    plt.close()
    print(f"✅ Saved coverage_vs_accuracy.png")

if __name__ == "__main__":
    generate_plots()
