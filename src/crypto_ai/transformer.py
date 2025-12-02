"""Transformer 기반 암호화폐 예측 모델"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from crypto_ai.analyzer import get_device


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with attention weights storage"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # 시각화용 저장

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()  # 시각화용 저장
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(context)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class CryptoTransformer(nn.Module):
    """
    Transformer 기반 암호화폐 가격 예측 모델

    Features:
    - Multi-head self-attention으로 시계열 패턴 학습
    - Positional encoding으로 시간 순서 인코딩
    - Multi-task learning 지원 (가격 방향 + 변동성 + 거래량)
    """

    def __init__(
        self,
        input_size: int = 9,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_classes: int = 3,
        multi_task: bool = False,
    ):
        """
        Args:
            input_size: 입력 특성 수
            d_model: 모델 차원
            num_heads: Attention 헤드 수
            num_layers: Transformer 레이어 수
            d_ff: Feed-forward 차원
            dropout: 드롭아웃 비율
            num_classes: 분류 클래스 수 (상승/하락/횡보)
            multi_task: 멀티태스크 학습 활성화
        """
        super().__init__()

        self.d_model = d_model
        self.multi_task = multi_task

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head (가격 방향)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Multi-task heads
        if multi_task:
            # 변동성 예측 (회귀)
            self.volatility_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

            # 거래량 변화 예측 (회귀)
            self.volume_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        self._init_weights()
        self.to(get_device())

    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
            return_attention: attention weights 반환 여부

        Returns:
            dict with keys:
                - "direction": (batch, num_classes) 방향 예측 logits
                - "volatility": (batch, 1) 변동성 예측 (multi_task=True)
                - "volume": (batch, 1) 거래량 변화 예측 (multi_task=True)
                - "attention": list of attention weights (return_attention=True)
        """
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        attention_weights = []
        for layer in self.encoder_layers:
            x = layer(x)
            if return_attention:
                attention_weights.append(layer.self_attention.attention_weights)

        # Global pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)

        # Outputs
        outputs = {
            "direction": self.classifier(x),
        }

        if self.multi_task:
            outputs["volatility"] = self.volatility_head(x)
            outputs["volume"] = self.volume_head(x)

        if return_attention:
            outputs["attention"] = attention_weights

        return outputs

    def predict(self, x: torch.Tensor) -> dict:
        """추론 (소프트맥스 적용)"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probs = F.softmax(outputs["direction"], dim=-1)

            direction_idx = probs.argmax(dim=-1)
            direction_names = ["하락", "횡보", "상승"]

            result = {
                "direction": direction_idx,
                "direction_name": [direction_names[i] for i in direction_idx.cpu().numpy()],
                "probabilities": probs,
                "confidence": probs.max(dim=-1).values,
            }

            if self.multi_task:
                result["volatility"] = outputs["volatility"]
                result["volume"] = outputs["volume"]

            return result


class MultiTaskLoss(nn.Module):
    """멀티태스크 학습용 복합 손실 함수"""

    def __init__(
        self,
        direction_weight: float = 1.0,
        volatility_weight: float = 0.3,
        volume_weight: float = 0.2,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight
        self.volume_weight = volume_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            outputs: 모델 출력
            targets: {
                "direction": (batch,) 방향 레이블,
                "volatility": (batch, 1) 변동성 타겟 (선택),
                "volume": (batch, 1) 거래량 타겟 (선택),
            }
        """
        losses = {}

        # Direction loss (분류)
        losses["direction"] = self.ce_loss(outputs["direction"], targets["direction"])

        total_loss = self.direction_weight * losses["direction"]

        # Volatility loss (회귀)
        if "volatility" in outputs and "volatility" in targets:
            losses["volatility"] = self.mse_loss(outputs["volatility"], targets["volatility"])
            total_loss += self.volatility_weight * losses["volatility"]

        # Volume loss (회귀)
        if "volume" in outputs and "volume" in targets:
            losses["volume"] = self.mse_loss(outputs["volume"], targets["volume"])
            total_loss += self.volume_weight * losses["volume"]

        losses["total"] = total_loss
        return losses


# ============================================================
# Attention 시각화 유틸리티
# ============================================================

def visualize_attention(
    attention_weights: list[torch.Tensor],
    timestamps: list | None = None,
    layer_idx: int = -1,
    head_idx: int = 0,
    save_path: str | None = None,
) -> None:
    """
    Attention weights 시각화

    Args:
        attention_weights: 레이어별 attention weights 리스트
        timestamps: 시간 레이블 (선택)
        layer_idx: 시각화할 레이어 인덱스 (-1=마지막)
        head_idx: 시각화할 헤드 인덱스
        save_path: 저장 경로 (선택)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization: uv sync --extra viz")
        return

    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    seq_len = attn.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")

    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")

    if timestamps is not None and len(timestamps) == seq_len:
        step = max(1, seq_len // 10)
        ax.set_xticks(range(0, seq_len, step))
        ax.set_xticklabels([str(timestamps[i])[:10] for i in range(0, seq_len, step)], rotation=45)

    plt.colorbar(im)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved attention visualization to {save_path}")
    else:
        plt.show()


def get_attention_summary(attention_weights: list[torch.Tensor]) -> dict:
    """Attention 패턴 요약 통계"""
    summaries = []

    for layer_idx, attn in enumerate(attention_weights):
        # attn shape: (batch, heads, seq, seq)
        avg_attn = attn.mean(dim=1)  # Average over heads

        # 최근 시점에 대한 attention 분포
        last_step_attn = avg_attn[0, -1]  # (seq,)

        # 가장 많이 attention 받는 위치들
        top_k = 5
        top_values, top_indices = last_step_attn.topk(top_k)

        summaries.append({
            "layer": layer_idx,
            "top_attended_positions": top_indices.cpu().numpy().tolist(),
            "top_attention_values": top_values.cpu().numpy().tolist(),
            "entropy": -(last_step_attn * last_step_attn.log()).sum().item(),
        })

    return {"layer_summaries": summaries}


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")

    # 모델 생성
    model = CryptoTransformer(
        input_size=9,
        d_model=64,
        num_heads=4,
        num_layers=3,
        multi_task=True,
    )

    # 더미 입력
    batch_size = 4
    seq_len = 60
    x = torch.randn(batch_size, seq_len, 9).to(device)

    # Forward pass
    outputs = model(x, return_attention=True)

    print(f"Direction logits: {outputs['direction'].shape}")
    print(f"Volatility: {outputs['volatility'].shape}")
    print(f"Volume: {outputs['volume'].shape}")
    print(f"Attention layers: {len(outputs['attention'])}")
    print(f"Attention shape: {outputs['attention'][0].shape}")

    # 추론
    result = model.predict(x)
    print(f"\nPredictions: {result['direction_name']}")
    print(f"Confidence: {result['confidence']}")

    # Attention 요약
    summary = get_attention_summary(outputs["attention"])
    print(f"\nAttention summary: {summary}")
