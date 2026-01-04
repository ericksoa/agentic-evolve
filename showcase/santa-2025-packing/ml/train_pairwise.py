"""
Training script for pairwise ranking model.

Usage:
    python train_pairwise.py --data pairwise_data.jsonl --epochs 50
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

from pairwise_model import create_ranking_model


class PairwiseDataset(Dataset):
    """Dataset of pairwise comparisons."""

    def __init__(self, jsonl_path: str, max_n: int = 30):
        self.samples = []
        self.max_n = max_n

        print(f"Loading pairwise data from {jsonl_path}...")
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)

                # Pad/truncate features to max_n * 3 + 1
                features_a = data['features_a']
                features_b = data['features_b']

                # Expected: max_n * 3 + 1 = 91 for max_n=30
                expected_len = max_n * 3 + 1

                if len(features_a) != expected_len:
                    # Pad or truncate
                    if len(features_a) < expected_len:
                        features_a = features_a + [0.0] * (expected_len - len(features_a))
                    else:
                        features_a = features_a[:expected_len]
                    if len(features_b) < expected_len:
                        features_b = features_b + [0.0] * (expected_len - len(features_b))
                    else:
                        features_b = features_b[:expected_len]

                features_a = torch.tensor(features_a, dtype=torch.float32)
                features_b = torch.tensor(features_b, dtype=torch.float32)
                target_n = torch.tensor([data['target_n']], dtype=torch.float32)
                label = torch.tensor([data['label']], dtype=torch.float32)

                self.samples.append((features_a, features_b, target_n, label))

        print(f"Loaded {len(self.samples)} pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_epoch(model, dataloader, optimizer, criterion, device, model_type):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features_a, features_b, target_n, label in dataloader:
        features_a = features_a.to(device)
        features_b = features_b.to(device)
        target_n = target_n.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        if model_type == "pairwise":
            pred = model(features_a, features_b, target_n)
            loss = criterion(pred, label)
            predicted = (pred > 0.5).float()
        else:  # margin
            score_a, score_b = model(features_a, features_b, target_n)
            # Margin ranking loss: we want score_better - score_worse > margin
            # label=1 means A is better, label=0 means B is better
            # Convert to: target = 1 if A better else -1
            target = 2 * label - 1  # Maps 0->-1, 1->1
            loss = criterion(score_a, score_b, target)
            predicted = (score_a > score_b).float()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (predicted == label).sum().item()
        total += label.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device, model_type):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features_a, features_b, target_n, label in dataloader:
            features_a = features_a.to(device)
            features_b = features_b.to(device)
            target_n = target_n.to(device)
            label = label.to(device)

            if model_type == "pairwise":
                pred = model(features_a, features_b, target_n)
                loss = criterion(pred, label)
                predicted = (pred > 0.5).float()
            else:  # margin
                score_a, score_b = model(features_a, features_b, target_n)
                target = 2 * label - 1
                loss = criterion(score_a, score_b, target)
                predicted = (score_a > score_b).float()

            total_loss += loss.item()
            correct += (predicted == label).sum().item()
            total += label.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to pairwise data JSONL')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-n', type=int, default=30)
    parser.add_argument('--model-type', type=str, default='pairwise',
                        choices=['pairwise', 'margin'])
    parser.add_argument('--output', type=str, default='ranking_model.pt')
    parser.add_argument('--val-split', type=float, default=0.1)
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data
    dataset = PairwiseDataset(args.data, max_n=args.max_n)

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = create_ranking_model(args.model_type, max_n=args.max_n)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.model_type == "pairwise":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MarginRankingLoss(margin=0.1)

    best_val_acc = 0
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.model_type
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, args.model_type
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'max_n': args.max_n,
                'model_type': args.model_type,
            }, args.output)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best epoch: {best_epoch}, val_acc: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()
