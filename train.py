import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import os
import argparse
from torchvision import transforms

from src.seraph import SERAPH
from src.utils.dataset_manager import Mill19DatasetManager
from src.utils.dataloader import Mill19Dataset

def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

def train(args):
    # 0. Dataset Setup (Automation)
    manager = Mill19DatasetManager(root_dir=args.data_root, ak=args.ak, sk=args.sk)
    
    if args.download:
        print("Starting automated dataset download...")
        if manager.download_dataset():
            manager.preprocess()
        else:
            print("Dataset download failed. Proceeding with existing data if available.")

    # 1. Setup Model and Optimizer
    config = {
        'num_entities': args.num_entities,
        'image_height': args.height,
        'image_width': args.width,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SERAPH(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 2. Real Dataset & DataLoader (Mill 19)
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = Mill19Dataset(root_dir=args.data_root, scene=args.scene, split="train", transform=transform)
    if len(dataset) == 0:
        print(f"Error: No data found for scene '{args.scene}' in {args.data_root}. Try running with --download.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 3. Resume from checkpoint
    start_epoch = load_checkpoint(model, optimizer, args.checkpoint)
    
    # 4. Training Loop
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    csv_file = open(args.results_csv, 'a' if start_epoch > 0 else 'w', newline='')
    writer = csv.writer(csv_file)
    if start_epoch == 0:
        writer.writerow(['epoch', 'phase', 'loss'])

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Phase 1: Organization
        p1_losses = []
        pbar = tqdm(dataloader, desc=f"Phase 1: Organization")
        model.train()
        for images in pbar:
            # SERAPH expects (B, K, 3, H, W) where K views are used
            # For simplicity, we repeat or stack dummy views if only 1 image per batch
            if images.dim() == 4:
                images = images.unsqueeze(1) # (B, 1, 3, H, W)
            
            images = images.to(device)
            optimizer.zero_grad()
            
            p1_out = model(images, mode="train_organization")
            loss = p1_out["layout_loss"]
            
            loss.backward()
            optimizer.step()
            
            p1_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
            
        avg_p1_loss = sum(p1_losses) / len(p1_losses)
        writer.writerow([epoch+1, 'Phase 1', avg_p1_loss])
        
        # Phase 2: Fine-Tuning
        p2_losses = []
        pbar = tqdm(dataloader, desc=f"Phase 2: Fine-tuning")
        for images in pbar:
            if images.dim() == 4:
                images = images.unsqueeze(1)
            images = images.to(device)
            optimizer.zero_grad()
            
            rendered, p1_out = model(images, mode="train_fine_tuning")
            
            # Reconstruction Loss
            # Target is the middle frame or the only frame
            loss = torch.mean((rendered - images[:, 0])**2)
            
            loss.backward()
            optimizer.step()
            
            p2_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
            
        avg_p2_loss = sum(p2_losses) / len(p2_losses)
        writer.writerow([epoch+1, 'Phase 2', avg_p2_loss])
        
        save_checkpoint(model, optimizer, epoch+1, args.checkpoint)

    csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SERAPH Training Script (Mill 19)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_entities", type=int, default=50)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pth")
    parser.add_argument("--results_csv", type=str, default="results/results.csv")
    parser.add_argument("--data_root", type=str, default="data/mill19")
    parser.add_argument("--scene", type=str, default="rubble", choices=["rubble", "residential", "sci-art", "building"])
    parser.add_argument("--download", action="store_true", help="Auto-download dataset via OpenXLab")
    parser.add_argument("--ak", type=str, help="OpenXLab Access Key")
    parser.add_argument("--sk", type=str, help="OpenXLab Secret Key")
    
    args = parser.parse_args()
    train(args)
