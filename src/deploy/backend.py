import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
import base64

from src.config import CLASS_NAMES, SEED
from src.models.cnn_backbone import get_model
from src.confidence.temperature_scaling import TemperatureScaler
from src.confidence.abstention import Abstainer
from src.confidence.confidence_utils import get_msp, get_entropy
from src.interpretability.gradcam import GradCAM
from src.interpretability.visualize import overlay_heatmap
from src.data.loader import get_transforms

app = FastAPI(title="Neuro Disease Detection Backend")

# Globals
model = None
gradcam = None
device = torch.device("cpu")

def load_system():
    global model, gradcam
    print("Loading system...")
    
    # 1. Load Backbone
    base_model = get_model(pretrained=False)
    # Using 'checkpoints/best_model.pth' - assuming execution from root
    checkpoint_path = "checkpoints/best_model.pth"
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        base_model.load_state_dict(state_dict)
        print(f"Loaded weights from {checkpoint_path}")
    except Exception as e:
        print(f"Warning: Could not load weights: {e}. Using random weights.")
    
    base_model.to(device)
    base_model.eval()
    
    # 2. Wrap with Temperature Scaler
    # We didn't save T, so we initialize with default 1.5 or 1.0
    # In a real app, we would load the calibrated scaler.
    model = TemperatureScaler(base_model)
    model.to(device)
    model.eval()
    
    # 3. Init GradCAM
    # Layer4 is the target for ResNet
    gradcam = GradCAM(base_model, base_model.layer4[-1])
    
    print("System loaded.")

@app.on_event("startup")
async def startup_event():
    load_system()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.7),
    temperature: float = Form(1.0)
):
    try:
        # Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        transform = get_transforms(phase='val')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            # Apply manual temperature if requested, else use model's internal
            # But TemperatureScaler.forward uses its internal T. 
            # If we want dynamic T from frontend, we should manually scale.
            
            logits = model.model(img_tensor) # Raw logits from base model
            scaled_logits = logits / temperature
            
            # Confidence
            msp = get_msp(scaled_logits)
            entropy = get_entropy(scaled_logits)
            probs = F.softmax(scaled_logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            
        # Abstention
        abstainer = Abstainer(threshold=threshold)
        # Note: Abstainer expects a batch, returns mask
        # We have batch size 1
        conf_score = msp.item()
        
        should_abstain = conf_score < threshold
        
        label_str = CLASS_NAMES.get(pred_idx + 1, "Unknown")
        
        return {
            "prediction": label_str,
            "confidence": float(conf_score),
            "entropy": float(entropy.item()),
            "abstained": bool(should_abstain),
            "threshold_used": threshold,
            "temperature_used": temperature
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    try:
        # Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        transform = get_transforms(phase='val')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # GradCAM
        # We need gradients, so mode must be able to track grads
        # But image tensor leaf variable might struggle if we didn't require grad on input?
        # GradCAM hook handles it on the layer weights usually.
        # But we need input.requires_grad for some methods? No, GradCAM uses layer gradients.
        
        heatmap, pred_idx, _ = gradcam(img_tensor)
        
        # Overlay
        # Normalize original image to 0-1 tensor for visualization helper
        # Actually overlay_heatmap expects the normalized tensor that went into the model
        overlay = overlay_heatmap(img_tensor[0], heatmap)
        
        # Encode
        im_overlay = Image.fromarray(overlay)
        buf = io.BytesIO()
        im_overlay.save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return {"image": img_str}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
