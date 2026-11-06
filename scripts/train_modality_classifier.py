#!/usr/bin/env python3
"""Train Modality Classifier on MedNIST using torchvision (pre-split)."""
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import signal, sys, time

classes = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\nReceived signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, args, is_best=False, suffix=""):
    """Save checkpoint with all training state."""
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_acc": best_val_acc,
        "class_labels": classes,
        "backbone": args.backbone,
        "num_classes": 6,
        "image_size": 64,
        "val_accuracy": best_val_acc,
        "timestamp": time.time(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if suffix:
        ckpt_path = output_path.with_stem(output_path.stem + suffix)
        torch.save(ckpt, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")
    
    if is_best:
        torch.save(ckpt, output_path)
        print(f"  Best model saved: {output_path} (ValAcc={best_val_acc:.1f}%)")
    
    latest_path = output_path.with_stem(output_path.stem + "_latest")
    torch.save(ckpt, latest_path)

def load_checkpoint(model, optimizer, scheduler, args):
    """Load checkpoint for resume training."""
    latest_path = Path(args.output).with_stem(Path(args.output).stem + "_latest")
    if not latest_path.exists():
        return 0, 0.0
    
    print(f"Resuming from checkpoint: {latest_path}")
    ckpt = torch.load(latest_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    print(f"  Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.1f}%")
    return start_epoch, best_val_acc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/mednist")
    parser.add_argument("--output", default="models/pretrained/cancers/modality_classifier.pt")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Save checkpoint every N epochs")
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Force CUDA if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    train_tf = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    
    print("Loading MedNIST (pre-split)...")
    train_ds = datasets.ImageFolder(Path(args.data_dir) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(args.data_dir) / "val", transform=val_tf)
    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=val_tf)
    
    print(f"Classes found: {train_ds.classes}")
    print(f"Train:{len(train_ds)} Val:{len(val_ds)} Test:{len(test_ds)}")
    train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=0)
    val_dl = DataLoader(val_ds, args.batch_size, False, num_workers=0)
    test_dl = DataLoader(test_ds, args.batch_size, False, num_workers=0)
    
    if args.backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 6)
    else:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 6)
    
    m = m.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)
    
    start_epoch = 0
    best = 0.0
    if args.resume:
        start_epoch, best = load_checkpoint(m, opt, sched, args)
    
    print(f"Starting training from epoch {start_epoch+1}/{args.epochs}")
    
    try:
        for ep in range(start_epoch, args.epochs):
            if shutdown_requested:
                print("\nShutdown requested, saving checkpoint...")
                save_checkpoint(m, opt, sched, ep, best, args, suffix="_interrupted")
                break
            
            m.train()
            tl, tc, tt = 0, 0, 0
            for x, y in train_dl:
                if shutdown_requested:
                    break
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = m(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                tl += loss.item()
                _, p = out.max(1)
                tt += y.size(0)
                tc += p.eq(y).sum().item()
            ta = 100*tc/tt
            
            m.eval()
            vl, vc, vt = 0, 0, 0
            with torch.no_grad():
                for x, y in val_dl:
                    if shutdown_requested:
                        break
                    x, y = x.to(device), y.to(device)
                    out = m(x)
                    loss = crit(out, y)
                    vl += loss.item()
                    _, p = out.max(1)
                    vt += y.size(0)
                    vc += p.eq(y).sum().item()
            va = 100*vc/vt
            sched.step(vl)
            
            is_best = va > best
            if is_best:
                best = va
            
            print(f"Ep {ep+1}/{args.epochs}: TrL={tl/len(train_dl):.4f} TrA={ta:.1f}% VlL={vl/len(val_dl):.4f} VlA={va:.1f}%")
            
            if (ep + 1) % args.checkpoint_interval == 0 or is_best or ep == args.epochs - 1:
                save_checkpoint(m, opt, sched, ep, best, args, is_best=is_best)
        
        if not shutdown_requested:
            print("Training completed!")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        save_checkpoint(m, opt, sched, ep, best, args, suffix="_interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        save_checkpoint(m, opt, sched, ep, best, args, suffix="_crashed")
        raise
    
    m.eval()
    tc, tt = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            _, p = m(x).max(1)
            tt += y.size(0)
            tc += p.eq(y).sum().item()
    print(f"Test Acc: {100*tc/tt:.1f}% | Best Val: {best:.1f}% | Saved: {args.output}")

if __name__ == "__main__":
    main()