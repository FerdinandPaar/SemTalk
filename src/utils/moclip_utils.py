"""
MoCLIP / TMR Text Encoder Utility Module
=========================================
Provides a self-contained TMR text encoder (DistilBERT + ACTORStyleEncoder head)
that produces 256-d motion-aligned text embeddings -- a drop-in replacement for
CLIP's 512-d text embeddings in SemTalk.

The TMR model (Petrovich et al., ICCV 2023) contrastively trains a text encoder
and a motion encoder on HumanML3D, so text embeddings live in the same space as
motion embeddings.

Usage
-----
    from src.utils.moclip_utils import load_tmr_text_encoder

    encoder = load_tmr_text_encoder(
        text_encoder_path="weights/moclip_checkpoints/models/tmr_humanml3d_guoh3dfeats/last_weights/text_encoder.pt",
        distilbert_path="distilbert-base-uncased",
        device="cuda:0",
    )
    emb = encoder.encode(["a person walks forward"])  # -> [1, 256] tensor
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Dict, List, Optional
from collections import OrderedDict
from einops import repeat


# ============================================================================
# TMR ACTORStyleEncoder -- replicated verbatim from Mathux/TMR
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class ACTORStyleEncoder(nn.Module):
    """Encoder from TMR / TEMOS / ACTOR -- maps variable-length token sequences
    into a fixed-size latent vector (256-d by default)."""

    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.projection = nn.Linear(nfeats, latent_dim)
        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))
        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )
        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

    def forward(self, x_dict: Dict) -> Tensor:
        x = x_dict["x"]
        mask = x_dict["mask"]
        x = self.projection(x)
        device = x.device
        bs = len(x)
        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)
        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        return final[:, : self.nbtokens]


# ============================================================================
# TMRTextEncoder -- full pipeline: text -> DistilBERT -> ACTORStyleEncoder -> 256d
# ============================================================================

class TMRTextEncoder(nn.Module):
    """
    Self-contained text encoder that replicates the TMR pipeline:
        raw text  ->  DistilBERT tokenizer + model (768-d)
                  ->  ACTORStyleEncoder head     (768 -> 256-d)

    The output is a 256-d vector per sentence that lives in TMR's
    contrastive motion-text latent space.
    """

    OUTPUT_DIM = 256  # exposed for downstream config

    def __init__(
        self,
        text_encoder_path: str,
        distilbert_path: str = "distilbert-base-uncased",
        device: str = "cpu",
        vae: bool = True,
    ):
        super().__init__()
        self.device = device

        # ---- 1. Load DistilBERT backbone ----
        from transformers import AutoTokenizer, AutoModel
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(distilbert_path)
        self.backbone = AutoModel.from_pretrained(distilbert_path)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ---- 2. Build ACTORStyleEncoder head (same arch as TMR) ----
        self.head = ACTORStyleEncoder(
            nfeats=768,       # DistilBERT hidden size
            vae=vae,          # TMR uses VAE mode (2 tokens: mu + logvar)
            latent_dim=256,
            ff_size=1024,
            num_layers=6,
            num_heads=4,
            dropout=0.1,
            activation="gelu",
        )

        # ---- 3. Load TMR-trained head weights ----
        sd = torch.load(text_encoder_path, map_location="cpu")
        if isinstance(sd, dict) and ("state_dict" in sd):
            sd = sd["state_dict"]
        missing, unexpected = self.head.load_state_dict(sd, strict=False)
        print(f"[TMR] Loaded text encoder head from {text_encoder_path}")
        if missing:
            print(f"[TMR]   missing keys:    {missing}")
        if unexpected:
            print(f"[TMR]   unexpected keys:  {unexpected}")

        # ---- 4. Freeze everything ----
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad = False

        self.to(device)

    def train(self, mode: bool = True):
        """Override to keep everything frozen."""
        self.training = False
        for module in self.children():
            module.train(False)
        return self

    @torch.no_grad()
    def _distilbert_encode(self, texts: List[str]) -> Dict:
        """Run DistilBERT and return {x: [B, T, 768], mask: [B, T]}."""
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        output = self.backbone(**encoded)
        mask = encoded["attention_mask"].bool()
        return {"x": output.last_hidden_state, "mask": mask}

    @torch.no_grad()
    def encode(self, texts) -> Tensor:
        """
        Encode text(s) into 256-d motion-aligned embeddings.

        Parameters
        ----------
        texts : str or List[str]
            Input sentence(s).

        Returns
        -------
        emb : Tensor  [B, 256]   (float32, on self.device)
        """
        if isinstance(texts, str):
            texts = [texts]

        x_dict = self._distilbert_encode(texts)
        # head returns [B, nbtokens, 256]; first token is mu
        out = self.head(x_dict)       # [B, 2, 256] if vae else [B, 1, 256]
        mu = out[:, 0, :]             # [B, 256] -- the mean / deterministic vector
        return mu.float()


# ============================================================================
# Helper: load a ready-to-use TMRTextEncoder
# ============================================================================

def load_tmr_text_encoder(
    text_encoder_path: str,
    distilbert_path: str = "distilbert-base-uncased",
    device: str = "cuda:0",
) -> TMRTextEncoder:
    """
    Convenience function to build and return a frozen TMRTextEncoder.

    Parameters
    ----------
    text_encoder_path : str
        Path to the TMR ``text_encoder.pt`` checkpoint.
    distilbert_path : str
        HuggingFace model name or local path for DistilBERT.
    device : str
        Target device.

    Returns
    -------
    TMRTextEncoder
        Frozen encoder with ``.encode(texts) -> [B, 256]`` method.
    """
    encoder = TMRTextEncoder(
        text_encoder_path=text_encoder_path,
        distilbert_path=distilbert_path,
        device=device,
    )
    return encoder


def get_tmr_output_dim() -> int:
    """TMR text encoder always outputs 256-d vectors."""
    return 256


# ============================================================================
# Diagnostic helpers
# ============================================================================

def inspect_checkpoint(ckpt_path: str):
    """Print keys + shapes of a checkpoint for debugging."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    print(f"Checkpoint has {len(sd)} keys. First 20:")
    for i, (k, v) in enumerate(sd.items()):
        if i >= 20:
            break
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v)
        print(f"  {k}: {shape}")


# ============================================================================
# Legacy CLIP overlay (kept for backward compatibility)
# ============================================================================

def load_moclip_text_encoder(ckpt_path, device="cuda:0", clip_version="ViT-B/32"):
    """
    Original CLIP-overlay approach. Kept for backward compatibility.
    For the recommended TMR-based approach, use load_tmr_text_encoder() instead.
    """
    import clip as _clip
    clip_model, _ = _clip.load(clip_version, device=device, jit=False)
    _clip.model.convert_weights(clip_model)

    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        if "state_dict" in raw:
            raw = raw["state_dict"]
        elif "model" in raw:
            raw = raw["model"]

    target_sd = clip_model.state_dict()
    mapped = {}
    for k, v in raw.items():
        if k in target_sd:
            mapped[k] = v
    if mapped:
        clip_model.load_state_dict(mapped, strict=False)
        print(f"[MoCLIP-legacy] Overlaid {len(mapped)} weights onto {clip_version}.")
    else:
        print(f"[MoCLIP-legacy] WARNING: 0 keys matched -- using vanilla {clip_version}.")

    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return clip_model
