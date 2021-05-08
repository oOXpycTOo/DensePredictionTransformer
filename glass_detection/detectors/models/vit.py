import torch
import torch.nn as nn

from .util import patchify, flatten_patches, TransformerLayer, SelfAttention, get_transformer_layers

class ViT(nn.Module):
    def __init__(self,
                height: int = 256,
                width: int = 256,
                patch_size: int = 16,
                input_dim: int = 3,
                hidden_dim: int = 512,
                output_dim: int = 512,
                n_layers: int = 4,
                n_heads: int = 4,
                mlp_dim: int = 1024,
                attention_dropout: float = 0.1,
                dropout: float = 0.1,
                reduction: str = 'token') -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        if self.height % self.patch_size != 0 or self.width % self.patch_size != 0:
            err_msg = ('Image size should be divisible by patch size.\n',
                        f'Image size (hxw): ({self.height}x{self.width}), patch size: {self.patch_size}.')
            raise RuntimeError(err_msg)
        self.n_patches = (self.height * self.width) // (self.patch_size * self.patch_size)
        self.reduction = reduction
        
        self.patch_embedding = nn.Conv2d(
            input_dim,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, hidden_dim))
        self.class_embedding = nn.Parameter(
            torch.zeros(1, 1, hidden_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformer_layers = get_transformer_layers(
            n_layers,
            hidden_dim,
            n_heads,
            mlp_dim,
            attention_dropout,
            dropout)
        self.encoded_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.zeros_(self.out.weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.size()
        x_class = self.class_embedding.expand(b, 1, -1)
        x_patch = self.patch_embedding(patches).flatten(2)
        x = torch.cat([x_class, x_patch], dim=1) + self.pos_embedding
        x = self.embedding_dropout(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x[:, 0] if self.reduction == 'token' else x.mean(dim=1) 
        x = self.encoded_norm(x)
        x = self.out(x)
        return x
