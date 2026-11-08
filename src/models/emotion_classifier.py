"""
Multi-task Emotion Classifier.

Architecture:
    RoBERTa-base encoder → [CLS] pooling → shared hidden → 3 task heads:
        1. Emotion classification (7 Ekman emotions + neutral)
        2. Sarcasm detection (binary)
        3. Escalation regression (0→1 score)

Designed to fit within 6GB VRAM with fp16 + gradient checkpointing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig

from src.config import ModelConfig, MODEL_CONFIG


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    Down-weights easy/confident predictions so the model focuses on hard examples.
    Much more stable than class-weighted CrossEntropyLoss for imbalanced data.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    Args:
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
               gamma=0 is equivalent to CrossEntropyLoss. Default: 2.0.
        label_smoothing: Label smoothing factor. Default: 0.1.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output, shape (batch, num_classes).
            targets: Ground-truth class indices, shape (batch,).

        Returns:
            Scalar loss.
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class EmotionClassifier(RobertaPreTrainedModel):
    """Multi-task RoBERTa model for emotion, sarcasm, and escalation detection.

    Produces three outputs per input:
        - emotion_logits: (batch, num_emotion_labels) — 7-class classification
        - sarcasm_logits: (batch, 2) — binary sarcasm classification
        - escalation_score: (batch, 1) — 0-to-1 escalation regression
    """

    def __init__(self, config: RobertaConfig, model_config: ModelConfig = MODEL_CONFIG):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        hidden_size = config.hidden_size
        dropout = model_config.dropout

        # Shared projection layer
        self.shared_dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, model_config.num_emotion_labels),
        )

        self.sarcasm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, model_config.num_sarcasm_labels),
        )

        self.escalation_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

        # Loss functions — focal loss handles class imbalance by focusing on hard examples
        self.emotion_loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
        self.sarcasm_loss_fn = nn.CrossEntropyLoss()
        self.escalation_loss_fn = nn.MSELoss()

        # Loss weights for multi-task balancing
        self.emotion_weight = 1.0
        self.sarcasm_weight = 0.5
        self.escalation_weight = 0.3

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        emotion_labels: torch.Tensor | None = None,
        sarcasm_labels: torch.Tensor | None = None,
        escalation_labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len).
            attention_mask: Attention mask, shape (batch, seq_len).
            emotion_labels: Ground-truth emotion class IDs, shape (batch,).
            sarcasm_labels: Ground-truth sarcasm labels, shape (batch,).
            escalation_labels: Ground-truth escalation scores, shape (batch, 1).

        Returns:
            Dictionary with logits and optional loss.
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Shared projection
        shared = self.shared_dense(cls_output)

        # Task heads
        emotion_logits = self.emotion_head(shared)
        sarcasm_logits = self.sarcasm_head(shared)
        escalation_score = self.escalation_head(shared)

        result: dict[str, torch.Tensor] = {
            "emotion_logits": emotion_logits,
            "sarcasm_logits": sarcasm_logits,
            "escalation_score": escalation_score,
        }

        # Compute loss if labels provided
        total_loss = torch.tensor(0.0, device=emotion_logits.device)
        has_loss = False

        if emotion_labels is not None:
            emotion_loss = self.emotion_loss_fn(emotion_logits, emotion_labels)
            total_loss = total_loss + self.emotion_weight * emotion_loss
            result["emotion_loss"] = emotion_loss
            has_loss = True

        if sarcasm_labels is not None:
            sarcasm_loss = self.sarcasm_loss_fn(sarcasm_logits, sarcasm_labels)
            total_loss = total_loss + self.sarcasm_weight * sarcasm_loss
            result["sarcasm_loss"] = sarcasm_loss
            has_loss = True

        if escalation_labels is not None:
            escalation_loss = self.escalation_loss_fn(
                escalation_score.squeeze(-1), escalation_labels.float()
            )
            total_loss = total_loss + self.escalation_weight * escalation_loss
            result["escalation_loss"] = escalation_loss
            has_loss = True

        if has_loss:
            result["loss"] = total_loss

        return result


def create_emotion_classifier(
    model_config: ModelConfig = MODEL_CONFIG,
    pretrained: bool = True,
) -> EmotionClassifier:
    """Factory function to create the EmotionClassifier.

    Args:
        model_config: Model hyperparameters.
        pretrained: If True, loads pre-trained RoBERTa weights.

    Returns:
        Initialized EmotionClassifier.
    """
    if pretrained:
        config = RobertaConfig.from_pretrained(model_config.base_model_name)
        model = EmotionClassifier.from_pretrained(
            model_config.base_model_name,
            config=config,
            model_config=model_config,
            ignore_mismatched_sizes=True,
        )
    else:
        config = RobertaConfig.from_pretrained(model_config.base_model_name)
        model = EmotionClassifier(config, model_config=model_config)

    return model
