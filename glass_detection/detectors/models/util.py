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

class RedoutConverter(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class IgnoreRedoutConverter(RedoutConverter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:]


class AddRedoutConverter(RedoutConverter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        redout_token = x[:, 0:1]
        return x[:, 1:] + redout_token


class MLPProjRedoutConverter(RedoutConverter):
    def __init__(self, hidden_dim: int, *args, **kwargs) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, c = x.size()
        redout_token = x[:, 0:1].expand(b, s-1, c)
        x = torch.cat([x[:, 1:], redout_token], dim=-1)
        return self.mlp(x)

class Concatenate(nn.Module):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, c = x.size()
        return x.reshape(b, self.height, self.width, c)


class Resample(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, scale: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1),
            self.__get_conv_instance(output_dim, scale)
        )

    def __get_conv_instance(self, dim: int, scale: float) -> nn.Module:
        if scale < 1:
            scale = int(1 / scale)
            return nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=scale,
                padding=1)
        else:
            scale = int(scale)
            return nn.ConvTranspose2d(
                dim,
                dim,
                kernel_size=scale,
                padding=0,
                stride=scale
            )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class ResidualConvUnit(nn.Module):
    def __init__(self, dim: int, use_bn: bool = False) -> None:
        super().__init__()
        use_bias = not use_bn
        self.conv_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)
        self.conv_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)
        self.relu = nn.ReLU()
        self.use_bn = use_bn
        if use_bn:
            self.bn_1 = nn.BatchNorm2d(dim)
            self.bn_2 = nn.BatchNorm2d(dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv_1(out)
        if self.use_bn:
            out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        if self.use_bn:
            out = self.bn_2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, hidden_dim: int, use_bn: bool = True):
        super().__init__()
        self.res_conv_1 = ResidualConvUnit(hidden_dim, use_bn)
        self.res_conv_2 = ResidualConvUnit(hidden_dim, use_bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.project = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)


    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        out = x[0]
        if len(x) > 1:
            x_upper_layer = self.res_conv_1(x[1])
            out = self.res_conv_2(out + x_upper_layer)
        out = self.upsample(out)
        return self.project(out)


class Reassemble(nn.Module):
    def __init__(self,
                n_patches_h: int,
                n_patches_w: int,
                input_dim: int,
                output_dim: int,
                scale: float,
                redout_type: str = 'proj') -> None:
        super().__init__()
        self.redout_converter = get_redout_converter(redout_type, hidden_dim=input_dim)
        self.concat = Concatenate(n_patches_h, n_patches_w)
        self.resample = Resample(input_dim, output_dim, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.redout_converter(x)
        x = self.concat(x)
        x = x.permute(0, 3, 1, 2)
        return self.resample(x)


def get_redout_converter(converter_type: str, *args, **kwargs):
    CONVERTERS = {'ignore': IgnoreRedoutConverter, 'add': AddRedoutConverter, 'proj': MLPProjRedoutConverter}
    return CONVERTERS[converter_type](*args, **kwargs)


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
