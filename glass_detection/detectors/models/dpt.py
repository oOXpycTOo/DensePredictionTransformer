from argparse import Namespace
from pathlib import Path
import timm
import torch
import torch.nn as nn

from typing import Callable, Dict, List, Union

from glass_detection.detectors.models.vit import ViT, get_vit_instance


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


def get_redout_converter(converter_type: str, *args, **kwargs):
    CONVERTERS = {'ignore': IgnoreRedoutConverter, 'add': AddRedoutConverter, 'proj': MLPProjRedoutConverter}
    return CONVERTERS[converter_type](*args, **kwargs)


class Concatenate(nn.Module):
    def __init__(self,
        height: int,
        width: int) -> None:
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
                n_tiles,
                input_dim: int,
                output_dim: int,
                scale: float,
                redout_type: str = 'proj') -> None:
        super().__init__()
        self.redout_converter = get_redout_converter(redout_type, hidden_dim=input_dim)
        self.concat = Concatenate(n_tiles, n_tiles)
        self.resample = Resample(input_dim, output_dim, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.redout_converter(x)
        x = self.concat(x)
        x = x.permute(0, 3, 1, 2)
        return self.resample(x)


class DensePredictionTransformer(nn.Module):
    LAYER_ACTIVATIONS = []
    def __init__(self,
                vit_arguments: Union[Dict[str], Namespace, Path],
                height: int = 384,
                width: int = 384,
                n_classes: int = 150,
                patch_size: int = 16,
                hidden_dim: int = 1024,
                redout_type: str = 'ignore',
                hook_layers: List[int] = [5, 12, 18, 24],
                features: List[int] = [256, 512, 1024, 1024],
                use_bn_in_fusion_layers: bool = True):
        super().__init__()
        self.backbone = get_vit_instance(vit_arguments)
        self.__register_backbone_hooks(hook_layers)
        n_tiles = height // patch_size
        scales = [4, 2, 1, 1/2.]
        self.reassemble_layers = self.__get_reassemble_layers(n_tiles, hidden_dim, features, scales, redout_type)
        self.fusion_maps, self.fusion_layers =\
            self.__get_fusion_layers(hidden_dim // 4, features, scales, use_bn_in_fusion_layers)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=hidden_dim // 4, out_channels=hidden_dim//4, kernel_size=1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=hidden_dim//4, out_channels=n_classes, kernel_size=3, padding=1)
        )

    def __get_reassemble_layers(
            self,
            n_tiles: int,
            hidden_dim: int,
            features: List[int],
            scales: List[int],
            redout_type: str) -> nn.Module:
        reassemble_layers = []
        for n_features, scale in zip(features, scales):
            reassemble_layers.append(Reassemble(n_tiles, hidden_dim, n_features, scale, redout_type))
        return nn.ModuleList(reassemble_layers)

    def __get_fusion_layers(
            self,
            hidden_dim: int,
            features: List[int],
            scales: List[int],
            use_bn: bool = True) -> nn.Module:
        fusion_layers = []
        fusion_maps = []
        for n_features in features:
            fusion_maps.append(nn.Conv2d(n_features, hidden_dim, kernel_size=3, padding=1, bias=False))
            fusion_layers.append(FeatureFusionBlock(hidden_dim, use_bn))
        return nn.ModuleList(fusion_maps), nn.ModuleList(fusion_layers)

    def __register_backbone_hooks(self, layers: List[int]) -> None:
        for i, layer_num in enumerate(layers):
            hook_fn = get_activation_hook(i)
            self.backbone.transformer_layers[layer_num-1].register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone(x)
        outs = []
        for i in range(4):
            outs.append(self.reassemble_layers[i](self.LAYER_ACTIVATIONS[i]))
            outs[i] = self.fusion_maps[i](outs[i])
        outs[-1] = self.fusion_layers[-1](outs[-1])
        for i in reversed(range(3)):
            outs[i] = self.fusion_layers[i](outs[i], outs[i+1])
        return self.head(outs[0])



def get_activation_hook(layer_name: int) -> Callable:
    def activations_hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        DensePredictionTransformer.LAYER_ACTIVATIONS.append(output)
    return activations_hook


if __name__ == '__main__':
    dpt = DensePredictionTransformer(384, 384)
    x = torch.randn(1, 3, 384, 384)
    out = dpt(x)
    print(out.shape)