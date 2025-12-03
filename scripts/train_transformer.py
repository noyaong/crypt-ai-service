#!/usr/bin/env python
"""Transformer 모델 학습 스크립트 (멀티태스크 지원)"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from crypto_ai.analyzer import get_device
from crypto_ai.preprocessing import DataPipeline, DataConfig, INPUT_SIZE
from crypto_ai.transformer import CryptoTransformer, MultiTaskLoss, visualize_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer crypto prediction model")

    # Data
    parser.add_argument("--symbol", type=str, default="BTC", help="Trading pair symbol")
    parser.add_argument("--interval", type=str, default="1h", choices=["1h", "4h", "1d"], help="Candle interval (1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    parser.add_argument("--seq-length", type=int, default=60, help="Sequence length")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # Model
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=256, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Multi-task
    parser.add_argument("--multi-task", action="store_true", help="Enable multi-task learning")
    parser.add_argument("--volatility-weight", type=float, default=0.3, help="Volatility loss weight")
    parser.add_argument("--volume-weight", type=float, default=0.2, help="Volume loss weight")

    # Paths (symbol will be appended automatically)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/transformer", help="Checkpoint base directory")
    parser.add_argument("--log-dir", type=str, default="runs/transformer", help="TensorBoard log base directory")

    return parser.parse_args()


class TransformerTrainer:
    """Transformer 모델 학습 클래스"""

    def __init__(
        self,
        model: CryptoTransformer,
        device: torch.device,
        checkpoint_dir: Path,
        log_dir: Path,
        multi_task: bool = False,
    ):
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.multi_task = multi_task

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
    ) -> dict[str, float]:
        """1 에폭 학습"""
        self.model.train()
        total_losses = {"total": 0.0, "direction": 0.0}
        if self.multi_task:
            total_losses["volatility"] = 0.0
            total_losses["volume"] = 0.0

        correct = 0
        total = 0

        for batch_idx, (x, targets) in enumerate(train_loader):
            x = x.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            optimizer.zero_grad()
            outputs = self.model(x)

            if self.multi_task:
                losses = criterion(outputs, targets)
                loss = losses["total"]
                for k, v in losses.items():
                    total_losses[k] += v.item()
            else:
                loss = criterion(outputs["direction"], targets["direction"])
                total_losses["total"] += loss.item()
                total_losses["direction"] += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            _, predicted = outputs["direction"].max(1)
            correct += predicted.eq(targets["direction"]).sum().item()
            total += targets["direction"].size(0)

        num_batches = len(train_loader)
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_losses["accuracy"] = correct / total

        return avg_losses

    @torch.no_grad()
    def validate(
        self,
        val_loader,
        criterion: nn.Module,
    ) -> dict[str, float]:
        """검증"""
        self.model.eval()
        total_losses = {"total": 0.0, "direction": 0.0}
        if self.multi_task:
            total_losses["volatility"] = 0.0
            total_losses["volume"] = 0.0

        correct = 0
        total = 0

        for x, targets in val_loader:
            x = x.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            outputs = self.model(x)

            if self.multi_task:
                losses = criterion(outputs, targets)
                for k, v in losses.items():
                    total_losses[k] += v.item()
            else:
                loss = criterion(outputs["direction"], targets["direction"])
                total_losses["total"] += loss.item()
                total_losses["direction"] += loss.item()

            _, predicted = outputs["direction"].max(1)
            correct += predicted.eq(targets["direction"]).sum().item()
            total += targets["direction"].size(0)

        num_batches = len(val_loader)
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        avg_losses["accuracy"] = correct / total

        return avg_losses

    def save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler,
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
            "multi_task": self.multi_task,
        }

        torch.save(checkpoint, self.checkpoint_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
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
        volatility_weight: float = 0.3,
        volume_weight: float = 0.2,
    ) -> dict:
        """학습 루프"""
        # 클래스 가중치 계산
        class_weights = self._compute_class_weights(train_loader)

        if self.multi_task:
            criterion = MultiTaskLoss(
                direction_weight=1.0,
                volatility_weight=volatility_weight,
                volume_weight=volume_weight,
                class_weights=class_weights.to(self.device),
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        epochs_without_improvement = 0

        print(f"\nTraining on {self.device}")
        print(f"Multi-task: {self.multi_task}")
        print(f"Class weights: {class_weights.tolist()}")
        print("-" * 80)

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, criterion, optimizer)

            # Validate
            val_metrics = self.validate(val_loader, criterion)

            # Scheduler step
            scheduler.step()

            # Logging
            self.writer.add_scalars("Loss/Total", {
                "train": train_metrics["total"],
                "val": val_metrics["total"]
            }, epoch)
            self.writer.add_scalars("Accuracy", {
                "train": train_metrics["accuracy"],
                "val": val_metrics["accuracy"]
            }, epoch)
            self.writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            if self.multi_task:
                self.writer.add_scalars("Loss/Direction", {
                    "train": train_metrics["direction"],
                    "val": val_metrics["direction"]
                }, epoch)
                self.writer.add_scalars("Loss/Volatility", {
                    "train": train_metrics["volatility"],
                    "val": val_metrics["volatility"]
                }, epoch)
                self.writer.add_scalars("Loss/Volume", {
                    "train": train_metrics["volume"],
                    "val": val_metrics["volume"]
                }, epoch)

            # Best model check
            val_loss = val_metrics["total"]
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
            log_str = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_metrics['total']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['total']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
            )
            if is_best:
                log_str += " *"
            print(log_str)

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
        all_probs = []

        for x, targets in test_loader:
            x = x.to(self.device)
            outputs = self.model(x)
            probs = torch.softmax(outputs["direction"], dim=-1)

            _, predicted = outputs["direction"].max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets["direction"].numpy())
            all_probs.extend(probs.cpu().numpy())

        import numpy as np
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = (all_preds == all_labels).mean()

        # Per-class metrics
        class_names = ["하락", "횡보", "상승"]
        per_class = {}
        for i, name in enumerate(class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class[name] = {
                    "accuracy": (all_preds[mask] == all_labels[mask]).mean(),
                    "count": int(mask.sum()),
                    "avg_confidence": all_probs[mask, i].mean(),
                }
            else:
                per_class[name] = {"accuracy": 0.0, "count": 0, "avg_confidence": 0.0}

        return {
            "accuracy": accuracy,
            "per_class": per_class,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }

    @torch.no_grad()
    def visualize_sample_attention(self, test_loader, save_path: str | None = None):
        """샘플 데이터의 Attention 시각화"""
        self.model.eval()

        x, targets = next(iter(test_loader))
        x = x[:1].to(self.device)  # 단일 샘플

        outputs = self.model(x, return_attention=True)
        attention_weights = outputs["attention"]

        if save_path:
            visualize_attention(attention_weights, save_path=save_path)
        else:
            visualize_attention(attention_weights)

    @staticmethod
    def _compute_class_weights(train_loader) -> torch.Tensor:
        """클래스 가중치 계산"""
        import numpy as np

        all_labels = []
        for _, targets in train_loader:
            all_labels.extend(targets["direction"].numpy())

        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)

        weights = len(all_labels) / (len(unique) * counts)
        weights = weights / weights.sum() * len(unique)

        return torch.tensor(weights, dtype=torch.float32)


def main():
    args = parse_args()

    print("=" * 80)
    print("Crypto AI - Transformer Model Training")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Days: {args.days}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Model: d_model={args.d_model}, heads={args.num_heads}, layers={args.num_layers}")
    print(f"Multi-task: {args.multi_task}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("=" * 80)

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
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        sequence_length=args.seq_length,
        multi_task=args.multi_task,
    )
    pipeline = DataPipeline(config)
    train_ds, val_ds, test_ds = pipeline.prepare_dataset(multi_task=args.multi_task)
    train_loader, val_loader, test_loader = pipeline.get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size
    )

    # Model
    model = CryptoTransformer(
        input_size=INPUT_SIZE,  # 13 features
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        multi_task=args.multi_task,
    )
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Trainer
    trainer = TransformerTrainer(
        model=model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        multi_task=args.multi_task,
    )

    # Train
    result = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        volatility_weight=args.volatility_weight,
        volume_weight=args.volume_weight,
    )

    print("\n" + "=" * 80)
    print("Training Complete")
    print(f"Best Epoch: {result['best_epoch']}")
    print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print("=" * 80)

    # Test evaluation
    print("\nTest Set Evaluation:")
    test_result = trainer.evaluate(test_loader)
    print(f"Overall Accuracy: {test_result['accuracy']:.4f}")
    print("\nPer-class Results:")
    for name, metrics in test_result["per_class"].items():
        print(f"  {name}: Acc={metrics['accuracy']:.4f}, "
              f"Count={metrics['count']}, "
              f"AvgConf={metrics['avg_confidence']:.4f}")

    # Save attention visualization
    print("\nSaving attention visualization...")
    trainer.visualize_sample_attention(
        test_loader,
        save_path=str(checkpoint_dir / "attention.png")
    )


if __name__ == "__main__":
    main()
