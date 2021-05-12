import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
        self.unify = nn.Linear(hidden_dim, hidden_dim)
        self.unify_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x tensor of shape (B, S, C)
        Returns:
            A tensor of shape (B, S, C)
        """
        b, s, c = x.size()
        n_heads = self.n_heads
        c = c // n_heads
        qkv = self.qkv(x).view(b, s, 3, n_heads, c).permute(2, 0, 3, 1, 4)  # (QKV, B, N_HEADS, S, C)
        query = qkv[0]
        key = qkv[1]
        value = qkv[2]
        del qkv

        query /= c ** (1 / 4)
        key /= c ** (1 / 4)

        weights = torch.einsum('bhik,bhjk->bhij', query, key)  # (B, N_HEADS, S_i, S_j)
        weights = self.softmax(weights)
        weights = self.attn_dropout(weights)
        output = torch.einsum('bhij,bhjk->bihk', weights, value)
        output = output.contiguous().view(b, s, c * n_heads)
        output = self.unify(output)
        return self.unify_dropout(output)


class TransformerLayer(nn.Module):
    def __init__(self,
        hidden_dim: int,
        n_heads: int,
        mlp_dim: int,
        attention_dropout: float = 0.1,
        dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.self_attention = SelfAttention(hidden_dim, n_heads, attention_dropout)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_dim),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(mlp_dim, hidden_dim),
                                  nn.Dropout(dropout))
        self.mlp.apply(self.__init_weights)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Identity()

    def __init_weights(self, module: nn.Module) -> None:
        if type(module) == nn.Linear:
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_x = self.layer_norm_1(x)
        out_x = self.self_attention(out_x)
        out_x = self.dropout(out_x)
        out_x = x + out_x
        out_y = self.layer_norm_2(out_x)
        out_y = self.mlp(out_y)
        out_y = self.dropout(out_y)
        return out_x + out_y


def patchify(images: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """Transforms images of shape (B, C, H, W) to (B, C, N, P^2),
        B - batch size,
        C - number of channels,
        H - image height,
        W - image width,
        P - patch size
        N - number of patches
    """
    b, c, h, w = image.size()
    n = h * w // (patch_size * patch_size)
    if h % patch_size != 0 or w % patch_size != 0:
        err_msg = ('Image size should be divisible by patch size.\n',
                    f'Image size (hxw): ({h}x{w}), patch size: {patch_size}.')
        raise RuntimeError(err_msg)
    patches = images.unfold(2).unfold(3)  # transforms to (B, C, SQRT(N), SQRT(N), P, P)
    return patches

def flatten_patches(patches: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """Flattens the patches (B, C, SQRT(N), SQRT(N), P, P) -> (B, N, C * P^2)"""
    patches = patches.contiguous().view(b, c, n, patch_size * patch_size)
    patches = patches.permute(0, 2, 1, 3).contiguous.view(b, n, c * patches * patches)
    return patches

def get_transformer_layers(
        n_layers: int,
        hidden_dim: int,
        n_heads: int,
        mlp_dim: int,
        attention_dropout: float,
        dropout: float) -> nn.ModuleList:
    blocks = []
    for i in range(n_layers):
        blocks.append(TransformerLayer(hidden_dim, n_heads, mlp_dim, attention_dropout, dropout))
    return nn.ModuleList(blocks)
