#!/usr/bin/env python
"""LSTM 모델 학습 스크립트"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from crypto_ai.analyzer import ChartAnalyzer, get_device
from crypto_ai.preprocessing import DataPipeline, DataConfig, INPUT_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crypto price prediction model")

    # Data
    parser.add_argument("--symbol", type=str, default="BTC", help="Trading pair symbol")
    parser.add_argument("--interval", type=str, default="1h", choices=["1h", "4h", "1d"], help="Candle interval (1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    parser.add_argument("--seq-length", type=int, default=60, help="Sequence length")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Model
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layers")

    # Paths (symbol will be appended automatically)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/lstm", help="Checkpoint base directory")
    parser.add_argument("--log-dir", type=str, default="runs/lstm", help="TensorBoard log base directory")

    return parser.parse_args()


class Trainer:
    """모델 학습 클래스"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: Path,
        log_dir: Path,
    ):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir / f"run_{timestamp}")

        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def train_epoch(
        self,
        train_loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(
        self,
        val_loader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            outputs = self.model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """체크포인트 저장"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
        }

        # 최신 체크포인트
        torch.save(checkpoint, self.checkpoint_dir / "last.pt")

        # 베스트 체크포인트
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> int:
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return checkpoint["epoch"]

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        lr: float,
        weight_decay: float,
        patience: int,
    ) -> dict:
        """학습 루프"""
        # 클래스 가중치 계산 (불균형 처리)
        class_weights = self._compute_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        epochs_without_improvement = 0

        print(f"\nTraining on {self.device}")
        print(f"Class weights: {class_weights.tolist()}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)

            # Scheduler step
            scheduler.step(val_loss)

            # Logging
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
            self.writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            # Best model check
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, optimizer, scheduler, val_loss, is_best)

            # Print progress
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                f"{'*' if is_best else ''}"
            )

            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        self.writer.close()

        # Load best model
        self.load_checkpoint(self.checkpoint_dir / "best.pt")

        return {
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "final_epoch": epoch,
        }

    @torch.no_grad()
    def evaluate(self, test_loader) -> dict:
        """테스트 세트 평가"""
        self.model.eval()

        all_preds = []
        all_labels = []

        for x, y in test_loader:
            x = x.to(self.device)
            outputs = self.model(x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())

        import numpy as np
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        # Per-class accuracy
        class_names = ["하락", "횡보", "상승"]
        per_class = {}
        for i, name in enumerate(class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class[name] = (all_preds[mask] == all_labels[mask]).mean()
            else:
                per_class[name] = 0.0

        return {
            "accuracy": accuracy,
            "per_class_accuracy": per_class,
            "predictions": all_preds,
            "labels": all_labels,
        }

    @staticmethod
    def _compute_class_weights(train_loader) -> torch.Tensor:
        """클래스 가중치 계산"""
        import numpy as np

        all_labels = []
        for _, y in train_loader:
            all_labels.extend(y.numpy())

        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)

        # Inverse frequency weighting
        weights = len(all_labels) / (len(unique) * counts)
        weights = weights / weights.sum() * len(unique)  # Normalize

        return torch.tensor(weights, dtype=torch.float32)


def main():
    args = parse_args()

    print("=" * 60)
    print("Crypto AI - Model Training")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Days: {args.days}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)

    # Device
    device = get_device()
    print(f"Device: {device}")

    # Symbol and interval specific paths
    symbol = args.symbol.upper()
    interval = args.interval
    checkpoint_dir = Path(args.checkpoint_dir) / symbol / interval
    log_dir = Path(args.log_dir) / symbol / interval
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Log Dir: {log_dir}")

    # Data
    config = DataConfig(
        symbol=symbol,
        interval=args.interval,
        days=args.days,
        sequence_length=args.seq_length,
    )
    pipeline = DataPipeline(config)
    train_ds, val_ds, test_ds = pipeline.prepare_dataset()
    train_loader, val_loader, test_loader = pipeline.get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size
    )

    # Model (13 features from preprocessing)
    model = ChartAnalyzer(
        input_size=INPUT_SIZE,  # 13 features
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    model = model.to(device)

    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    # Train
    result = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"Best Epoch: {result['best_epoch']}")
    print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print("=" * 60)

    # Test evaluation
    print("\nTest Set Evaluation:")
    test_result = trainer.evaluate(test_loader)
    print(f"Overall Accuracy: {test_result['accuracy']:.4f}")
    print("Per-class Accuracy:")
    for name, acc in test_result["per_class_accuracy"].items():
        print(f"  {name}: {acc:.4f}")


if __name__ == "__main__":
    main()
