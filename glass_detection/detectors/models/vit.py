from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .util import get_transformer_layers


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


HEIGHT = 384
WIDTH = 384
PATCH_SIZE = 16
INPUT_DIM = 3
HIDDEN_DIM = 1024
N_CLASSES = 1000
N_LAYERS = 24
N_HEADS = 16
MLP_DIM = 4096
ATTENTION_DROPOUT = 0.0
DROPOUT = 0.0
REDUCTION = 'token'


class ViT(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: Optional[Namespace] = None) -> None:
        super().__init__()
        args = vars(args) if args is not None else {}
        self.height = data_config['height']
        self.width = data_config['width']
        self.patch_size = data_config['patch_size']
        if self.height % self.patch_size != 0 or self.width % self.patch_size != 0:
            err_msg = ('Image size should be divisible by patch size.\n',
                        f'Image size (hxw): ({self.height}x{self.width}), patch size: {self.patch_size}.')
            raise RuntimeError(err_msg)
        self.n_patches_h = self.height // self.patch_size
        self.n_patches_w = self.width // self.patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        self.reduction = args.get('vit_reduction', REDUCTION)
        self.hidden_dim = args.get('vit_hidden_dim', HIDDEN_DIM)


        input_dim = args.get('vit_input_dim', INPUT_DIM)
        n_classes = args.get('vit_n_classes', N_CLASSES)
        n_heads = args.get('vit_n_heads', N_HEADS)
        n_layers = args.get('vit_n_layers', N_LAYERS)
        mlp_dim = args.get('vit_mlp_dim', MLP_DIM)
        attention_dropout = args.get('vit_attn_dropout', ATTENTION_DROPOUT)
        dropout = args.get('vit_dropout', DROPOUT)
        
        self.patch_embedding = nn.Conv2d(
            input_dim,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, self.hidden_dim))
        self.class_embedding = nn.Parameter(
            torch.zeros(1, 1, self.hidden_dim))
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformer_layers = get_transformer_layers(
            n_layers,
            self.hidden_dim,
            n_heads,
            mlp_dim,
            attention_dropout,
            dropout)
        self.encoded_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.out = nn.Linear(self.hidden_dim, n_classes)
        torch.nn.init.zeros_(self.out.weight)

    def resize_pos_embedding_(self, height: int, width: int) -> None:
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

    @staticmethod
    def from_parameters(height: int = 384,
            width: int = 384,
            patch_size: int = 16,
            input_dim: int = 3,
            hidden_dim: int = 1024,
            n_classes: int = 1000,
            n_layers: int = 24,
            n_heads: int = 16,
            mlp_dim: int = 4096,
            attention_dropout: float = 0.0,
            dropout: float = 0.0,
            reduction: str = 'token') -> 'ViT':
        args = Namespace(
            vit_input_dim=input_dim,
            vit_hidden_dim=hidden_dim,
            vit_n_classes=n_classes,
            vit_n_layers=n_layers,
            vit_n_heads=n_heads,
            vit_mlp_dim=mlp_dim,
            vit_attn_dropout=attention_dropout,
            vit_dropout=dropout,
            vit_reduction=reduction)
        data_config = {'height': height, 'width': width, 'patch_size': patch_size}
        return ViT(data_config, args)

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> None:
        parser.add_argument('--vit_input_dim', type=int, default=INPUT_DIM)
        parser.add_argument('--vit_hidden_dim', type=int, default=HIDDEN_DIM)
        parser.add_argument('--vit_n_classes', type=int, default=N_CLASSES)
        parser.add_argument('--vit_n_layers', type=int, default=N_LAYERS)
        parser.add_argument('--vit_n_heads', type=int, default=N_HEADS)
        parser.add_argument('--vit_mlp_dim', type=int, default=MLP_DIM)
        parser.add_argument('--vit_attn_dropout', type=float, default=ATTENTION_DROPOUT)
        parser.add_argument('--vit_dropout', type=float, default=DROPOUT)
        parser.add_argument('--vit_reduction', type=str, default=REDUCTION,
            help='How to compute final probabilities, using zero-token (token) or by taking average along all features (mean)')


def get_vit_instance(vit_model: str = 'vit_l_16', path_to_model: Optional[Path] = None) -> ViT:
    if path_to_model is None:
        path_to_model = Path(__file__).resolve().parents[3] / 'models' / f'{vit_model}.pt'
    if vit_model == 'vit_l_16':
        model = ViT.from_parameters(384, 384)
        state_dict = torch.load(str(path_to_model), map_location='cpu')
        model.load_state_dict(state_dict)
        return model
    else:
        raise NotImplementedError(f'Not implemented factory for model: {vit_model}')
