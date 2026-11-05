#!/usr/bin/env python3
"""Train Modality Classifier on MedNIST using existing splits (no MONAI)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import datasets, transforms as T
from torchvision.models import resnet18, ResNet18_Weights

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/mednist")
    parser.add_argument("--output", default="models/pretrained/modality_classifier.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classes = ["AbdomenCT", "BreastMRI", "CXR", "ChestCT", "Hand", "HeadCT"]

    train_tf = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    val_tf = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

    train_ds = datasets.ImageFolder(root=f"{args.data_dir}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(root=f"{args.data_dir}/val", transform=val_tf)
    test_ds = datasets.ImageFolder(root=f"{args.data_dir}/test", transform=val_tf)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Classes: {train_ds.classes}")

    train_dl = DataLoader(train_ds, args.batch_size, True, num_workers=0)
    val_dl = DataLoader(val_ds, args.batch_size, False, num_workers=0)
    test_dl = DataLoader(test_ds, args.batch_size, False, num_workers=0)

    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 6)
    m = m.to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)

    best = 0
    for ep in range(args.epochs):
        m.train()
        tl, tc, tt = 0, 0, 0
        for x, y in train_dl:
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
        ta = 100 * tc / tt

        m.eval()
        vl, vc, vt = 0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = m(x)
                loss = crit(out, y)
                vl += loss.item()
                _, p = out.max(1)
                vt += y.size(0)
                vc += p.eq(y).sum().item()
        va = 100 * vc / vt
        sched.step(vl)

        print(f"Ep {ep+1}/{args.epochs}: TrL={tl/len(train_dl):.4f} TrA={ta:.1f}% VlL={vl/len(val_dl):.4f} VlA={va:.1f}%")
        if va > best:
            best = va
            torch.save({
                "model_state_dict": m.state_dict(),
                "class_labels": classes,
                "backbone": "resnet18",
                "num_classes": 6,
                "image_size": 64,
                "val_accuracy": va
            }, args.output)
            print(f"  Saved (ValAcc={va:.1f}%)")

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