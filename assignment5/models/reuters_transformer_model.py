import math
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class CustomTransformerConfig(PretrainedConfig):
    model_type = "custom_transformer_encoder"
    def __init__(
        self,
        vocab_size=30_522,
        num_labels=46,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.2,
        max_position_embeddings=256,
        layer_norm_eps=1e-5,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10_000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)

        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B,T,D]
        return x + self.pe[: x.size(1), :]


class TransformerBlock(nn.Module):
    def __init__(self, cfg: CustomTransformerConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads

        assert d % h == 0

        self.h, self.hd = h, d // h
        self.ln1 = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(d, eps=cfg.layer_norm_eps)

        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

        self.drop = nn.Dropout(cfg.dropout)
        self.ff  = nn.Sequential(nn.Linear(d, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, d))

    def forward(self, x, key_padding_mask: Optional[torch.Tensor]):
        """
        x: [B,T,D]
        key_padding_mask: [B,T] bool where True=PAD (mask out)
        """
        y = self.ln1(x).float()
        B, T, D = y.shape
        H, Hd = self.h, self.hd

        q = self.q(y).view(B, T, H, Hd).transpose(1, 2)  # [B,H,T,Hd]
        k = self.k(y).view(B, T, H, Hd).transpose(1, 2)
        v = self.v(y).view(B, T, H, Hd).transpose(1, 2)

        # Attention mask [B,1,1,Tk] True = mask
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Hd)   # [B,H,T,T]

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e4)
            all_masked = attn_mask.all(dim=-1, keepdim=True)            # [B,1,T,1]
            scores = torch.where(all_masked, torch.zeros_like(scores), scores)

        scores = scores - scores.amax(dim=-1, keepdim=True)
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.drop.p, training=self.training)

        attn_out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, T, D).to(x.dtype)
        x = x + self.drop(self.o(attn_out))
        y = self.ln2(x)
        x = x + self.drop(self.ff(y))
        return x


class CustomTransformerForSequenceClassification(PreTrainedModel):
    """
    IMPORTANT: This model returns logits only.
    The trainer is the *only* place that computes loss (CE), so train & eval match.
    """
    config_class = CustomTransformerConfig
    base_model_prefix = "custom_tfmr"

    def __init__(self, config: CustomTransformerConfig):
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.pos = PositionalEncoding(config.d_model, config.max_position_embeddings)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.drop = nn.Dropout(config.dropout)
        self.cls = nn.Linear(config.d_model, config.num_labels)

        self.post_init()

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value

    def forward(self, input_ids=None, attention_mask=None, labels=None, **_):
        x = self.emb(input_ids)
        x = self.pos(x)
        x = self.drop(x)

        kpm = (attention_mask == 0) if attention_mask is not None else None
        for blk in self.layers:
            x = blk(x, kpm)

        x = self.ln_f(x)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = x.mean(dim=1)

        logits = self.cls(self.drop(pooled))
        return SequenceClassifierOutput(logits=logits)
