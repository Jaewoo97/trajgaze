"""
Text query encoder for TrajGaze_v2.

Simple word-level encoder: tokenize → embedding → 2-layer transformer → mean pool → project.
No pretrained model required. Vocab built from dataset at init or passed in.

Output: (B, D_QUERY=128) float32 tensor.

For Stage 1 (no query): pass None → returns zero vector.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import torch
import torch.nn as nn

D_QUERY   = 128
VOCAB_SIZE = 8192   # character-trigram hashing vocabulary
MAX_LEN    = 64     # max tokens per query


def tokenize(text: str, max_len: int = MAX_LEN) -> list[int]:
    """
    Simple character-trigram hashing tokenizer.

    Splits text on whitespace and punctuation, extracts trigrams from each token,
    hashes to [1, VOCAB_SIZE). Index 0 is padding.
    """
    text = text.lower()
    words = re.split(r"[\s\W]+", text)
    ids = []
    for word in words:
        if not word:
            continue
        # Add word-level hash
        word_id = (hash(word) % (VOCAB_SIZE - 1)) + 1
        ids.append(word_id)
        # Add trigrams for subword coverage
        padded = f"#{word}#"
        for i in range(len(padded) - 2):
            tri = padded[i:i+3]
            tri_id = (hash(tri) % (VOCAB_SIZE - 1)) + 1
            ids.append(tri_id)
        if len(ids) >= max_len:
            break
    return ids[:max_len]


def pad_tokens(
    tokens_list: list[list[int]],
    max_len: int = MAX_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad list of token id lists to max_len.

    Returns:
        ids:  (B, max_len) long tensor
        mask: (B, max_len) bool tensor — True = real token, False = padding
    """
    B = len(tokens_list)
    ids  = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, toks in enumerate(tokens_list):
        L = min(len(toks), max_len)
        ids[i, :L]  = torch.tensor(toks[:L], dtype=torch.long)
        mask[i, :L] = True
    return ids, mask


class QueryEncoder(nn.Module):
    """
    Encodes a text query (question) into a D_QUERY-dim vector.

    Architecture:
      embedding (VOCAB_SIZE, D_QUERY)
      + sinusoidal PE
      → 2-layer transformer encoder
      → masked mean pool
      → linear(D_QUERY, D_QUERY) + LayerNorm
    """

    def __init__(
        self,
        d_model:    int = D_QUERY,
        vocab_size: int = VOCAB_SIZE,
        n_layers:   int = 2,
        n_heads:    int = 4,
        max_len:    int = MAX_LEN,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model         = d_model,
                nhead           = n_heads,
                dim_feedforward = d_model * 4,
                dropout         = 0.1,
                activation      = "gelu",
                batch_first     = True,
                norm_first      = True,
            ),
            num_layers = n_layers,
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        queries: Optional[list[str]],
        device:  torch.device,
    ) -> torch.Tensor:
        """
        Args:
            queries: list of B text strings, or None for null query
            device:  target device

        Returns:
            (B, D_QUERY) float32
        """
        B = len(queries) if queries is not None else 1

        if queries is None or all(not q for q in queries):
            return torch.zeros(B, self.d_model, device=device)

        tokens_list = [tokenize(q) for q in queries]
        ids, mask = pad_tokens(tokens_list, self.max_len)
        ids  = ids.to(device)
        mask = mask.to(device)

        x = self.embedding(ids)                          # (B, L, D)
        x = x + self.pe[:, :x.shape[1]]                 # add PE
        # TransformerEncoder uses src_key_padding_mask where True = ignore
        pad_mask = ~mask                                 # (B, L)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L, D)

        # Masked mean pool (exclude padding)
        mask_f = mask.unsqueeze(-1).float()              # (B, L, 1)
        x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)  # (B, D)
        return self.proj(x)                              # (B, D)
