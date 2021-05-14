from argparse import ArgumentError, Namespace
import json
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import torch
from torch.functional import align_tensors
import torch.nn as nn

from .util import patchify, flatten_patches, TransformerLayer, SelfAttention, get_transformer_layers


# TODO: To be implemented
def convert_weights():
    pass


def visualize_vit_pos_embeddings(emb: torch.Tensor,
                                 patch_h: int, patch_w: int,
                                 nrows: int, ncols: int,
                                 **fig_params) -> None:
    img = emb[0, 1:].detach().cpu().numpy().reshape(patch_h, patch_w, nrows, ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, **fig_params)
    for i in range(nrows):
        for j in range(ncols):
            ax[i][j].imshow(img[:, :, i, j])
    return fig


class ViT(nn.Module):
    def __init__(self,
                height: int = 384,
                width: int = 384,
                patch_size: int = 16,
                input_dim: int = 3,
                hidden_dim: int = 1024,
                output_dim: int = 1000,
                n_layers: int = 24,
                n_heads: int = 16,
                mlp_dim: int = 4096,
                attention_dropout: float = 0.0,
                dropout: float = 0.0,
                reduction: str = 'token') -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        if self.height % self.patch_size != 0 or self.width % self.patch_size != 0:
            err_msg = ('Image size should be divisible by patch size.\n',
                        f'Image size (hxw): ({self.height}x{self.width}), patch size: {self.patch_size}.')
            raise RuntimeError(err_msg)
        self.n_patches_h = self.height // self.patch_size
        self.n_patches_w = self.width // self.patch_size
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
        self.encoded_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.out = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.zeros_(self.out.weight)

    def interpolate_pos_embedding_(self, height: int, width: int) -> None:
        n_patches_h = height // self.patch_size
        n_patches_w = width // self.patch_size
        pos_token, pos_grid = self.pos_embedding[:, 0:1], self.pos_embedding[:, 1:]
        pos_emb = pos_grid.transpose(1, 2).reshape(1, -1, self.n_patches_h, self.n_patches_w)
        pos_emb = torch.nn.functional.interpolate(pos_emb, (n_patches_h, n_patches_w), mode='bilinear', align_corners=True)
        pos_emb = pos_emb.reshape(1, -1, n_patches_h * n_patches_w).transpose(1, 2)
        pos_emb = torch.cat([pos_token, pos_emb], dim=1)
        self.pos_embedding = nn.Parameter(pos_emb)
        self.height = height
        self.width = width
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w
        self.n_patches = n_patches_w * n_patches_h

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.size()
        x_class = self.class_embedding.expand(b, 1, -1)
        x_patch = self.patch_embedding(images).flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, S, C)
        x = torch.cat([x_class, x_patch], dim=1) + self.pos_embedding
        x = self.embedding_dropout(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x[:, 0] if self.reduction == 'token' else x.mean(dim=1) 
        x = self.encoded_norm(x)
        x = self.out(x)
        return x

def create_vit_from_json(path: Path) -> ViT:
    with open(path, 'r') as json_file:
        json_dict = json.load(json_file)
        return ViT(**json_dict)

def create_vit_from_args(arguments: Namespace) -> ViT:
    arguments = vars(arguments)
    return ViT(**arguments)

def get_vit_instance(source: Union[Namespace, Path, Dict[str]]) -> ViT:
    if isinstance(source, Namespace):
        return create_vit_from_args(source)
    elif isinstance(source, Path):
        return create_vit_from_json(source)
    elif isinstance(source, dict):
        return ViT(**source)
    else:
        raise ArgumentError('Argument should be of type Path, Namespace or Dict[str]')