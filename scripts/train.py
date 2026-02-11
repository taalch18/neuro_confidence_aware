import torch
import argparse
from src.config import SPLITS_DIR, SEED
from src.data.loader import get_dataloaders
from src.models.cnn_backbone import get_model
from src.training.trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Train Baseline Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches of train/val for debugging")
    
    args = parser.parse_args()
    
    # Reproducibility
    torch.manual_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(SPLITS_DIR, batch_size=args.batch_size)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Init Model
    model = get_model(pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # Config definition
    config = {
        'epochs': args.epochs if not args.fast_dev_run else 1,
        'lr': args.lr,
        'save_dir': args.save_dir,
        'limit_batches': 2 if args.fast_dev_run else None
    }
    
    # Train
    print("Starting training...")
    train_model(model, train_loader, val_loader, config, device)
    
    print("Training Complete.")

if __name__ == "__main__":
    main()
