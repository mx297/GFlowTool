import torch
import torch.nn as nn


class PromptConditionedLogZ(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_prompt_embeds: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_prompt_embeds).squeeze(-1)