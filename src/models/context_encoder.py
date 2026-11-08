"""
Context-aware encoder for conversational emotion recognition.

Wraps utterance-level RoBERTa embeddings with a bidirectional GRU that models
inter-turn dependencies and speaker-aware context flow.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ModelConfig, MODEL_CONFIG


class ConversationContextEncoder(nn.Module):
    """Bidirectional GRU over utterance embeddings for conversational context.

    Takes a sequence of per-utterance [CLS] embeddings and produces
    context-enriched representations that capture:
        - Temporal flow of the conversation
        - Speaker-aware state transitions
        - Emotional momentum / escalation trajectories

    Input:
        utterance_embeddings: (num_turns, hidden_size) — [CLS] from RoBERTa
        speaker_ids: (num_turns,) — integer speaker identifiers

    Output:
        context_embeddings: (num_turns, context_hidden * 2) — bidirectional
    """

    def __init__(self, model_config: ModelConfig = MODEL_CONFIG):
        super().__init__()

        input_size = model_config.hidden_size
        hidden_size = model_config.context_gru_hidden
        num_layers = model_config.context_gru_layers
        dropout = model_config.dropout

        # Speaker embedding (support up to 20 speakers in a conversation)
        self.speaker_embedding = nn.Embedding(20, 64)

        # Projection to combine utterance embedding + speaker embedding
        self.input_projection = nn.Linear(input_size + 64, input_size)

        # Bidirectional GRU for sequential context
        self.context_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Output projection back to hidden_size
        self.output_projection = nn.Linear(hidden_size * 2, input_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        utterance_embeddings: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a conversation's utterance embeddings with context.

        Args:
            utterance_embeddings: (batch, num_turns, hidden_size) or
                                  (num_turns, hidden_size).
            speaker_ids: (batch, num_turns) or (num_turns,) — integer IDs.

        Returns:
            Context-enriched embeddings, same shape as input.
        """
        # Handle unbatched input
        if utterance_embeddings.dim() == 2:
            utterance_embeddings = utterance_embeddings.unsqueeze(0)
            if speaker_ids is not None:
                speaker_ids = speaker_ids.unsqueeze(0)

        batch_size, num_turns, hidden_size = utterance_embeddings.shape

        # Add speaker information if available
        if speaker_ids is not None:
            speaker_ids = speaker_ids.clamp(0, 19)  # Safety clamp
            speaker_emb = self.speaker_embedding(speaker_ids)  # (B, T, 64)
            combined = torch.cat([utterance_embeddings, speaker_emb], dim=-1)
            combined = self.input_projection(combined)
        else:
            combined = utterance_embeddings

        # Run through bidirectional GRU
        gru_output, _ = self.context_gru(combined)  # (B, T, hidden*2)

        # Layer norm + dropout
        gru_output = self.layer_norm(gru_output)
        gru_output = self.dropout(gru_output)

        # Project back to original hidden size
        context_output = self.output_projection(gru_output)  # (B, T, hidden_size)

        # Residual connection
        output = utterance_embeddings + context_output

        return output.squeeze(0) if batch_size == 1 else output
