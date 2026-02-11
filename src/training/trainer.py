import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from src.training.metrics import calculate_metrics

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds)
        all_labels.extend(labels)
        
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(torch.stack(all_labels), torch.stack(all_preds))
    metrics['loss'] = epoch_loss
    
    return metrics

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(torch.stack(all_labels), torch.stack(all_preds))
    metrics['loss'] = epoch_loss
    
    return metrics

def train_model(model, train_loader, val_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    # Using Adam as it generally converges faster for prototypes
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    best_val_loss = float('inf')
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    limit_batches = config.get('limit_batches', None)
    
    results = []
    
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
            if limit_batches and i >= limit_batches:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds)
            all_labels.extend(labels)
            
        epoch_loss = running_loss / (len(all_labels) if len(all_labels) > 0 else 1)
        train_metrics = calculate_metrics(torch.stack(all_labels), torch.stack(all_preds))
        train_metrics['loss'] = epoch_loss

        # Val
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(val_loader, desc="Validation")):
                if limit_batches and i >= limit_batches:
                    break
                    
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds)
                all_labels.extend(labels)
                
        epoch_loss = running_loss / (len(all_labels) if len(all_labels) > 0 else 1)
        val_metrics = calculate_metrics(torch.stack(all_labels), torch.stack(all_preds))
        val_metrics['loss'] = epoch_loss
        
        print(f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = save_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            
        results.append({
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        })
        
    return results
