from argparse import ArgumentParser, Namespace
from pathlib import Path
import timm
import torch
import torch.nn as nn

from typing import Any, Callable, Dict, List, Union

from .vit import get_vit_instance
from .util import Reassemble, FeatureFusionBlock


VIT_MODEL = 'vit_l_16'
HOOK_LAYERS = [6, 12, 18, 24]
FEATURES = [256, 512, 1024, 1024]
REDOUT_TYPE = 'proj'
BN_FUSION_LAYERS = True
HIDDEN_DIM = 256
N_CLASSES = 150


class DensePredictionTransformer(nn.Module):
    LAYER_ACTIVATIONS = []

    def __init__(self, data_config: Dict[str, Any], args: Namespace = None):
        super().__init__()
        args = vars(args) if args is not None else {}

        height = data_config['height']
        width = data_config['width']
        patch_size = data_config['patch_size']
        n_classes = data_config['n_classes']
        vit_model = args.get('vit_model', VIT_MODEL)
        hook_layers = args.get('dpt_hook_layers', HOOK_LAYERS)
        features = args.get('dpt_features', FEATURES)
        redout_type = args.get('dpt_redout_type', REDOUT_TYPE)
        use_bn_in_fusion_layers = args.get('dpt_bn_fusion_layers', BN_FUSION_LAYERS)
        hidden_dim = args.get('dpt_hidden_dim', HIDDEN_DIM)

        self.backbone = self.__setup_backbone(vit_model, height, width)
        self.__register_backbone_hooks(hook_layers)
        n_patches_h = self.backbone.n_patches_h
        n_patches_w = self.backbone.n_patches_w
        scales = [4, 2, 1, 1/2.]  # TODO: Fix this hardcode

        self.reassemble_layers =\
            self.__get_reassemble_layers(n_patches_h,
                                         n_patches_w,
                                         self.backbone.hidden_dim,
                                         features,
                                         scales,
                                         redout_type)
        self.fusion_maps, self.fusion_layers =\
            self.__get_fusion_layers(hidden_dim, features, scales, use_bn_in_fusion_layers)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=hidden_dim, out_channels=n_classes, kernel_size=3, padding=1)
        )

    def __setup_backbone(self, backbone_type: str, height: int, width: int) -> 'ViT':
        model = get_vit_instance(backbone_type)
        model.resize_pos_embedding_(height, width)
        model.eval()
        model.requires_grad_(False)
        return model

    def __get_reassemble_layers(
            self,
            n_patches_h: int,
            n_patches_w: int,
            hidden_dim: int,
            features: List[int],
            scales: List[int],
            redout_type: str) -> nn.Module:
        reassemble_layers = []
        for n_features, scale in zip(features, scales):
            reassemble_layers.append(Reassemble(n_patches_h, n_patches_w, hidden_dim, n_features, scale, redout_type))
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

    @staticmethod
    def from_parameters(height: int = 384,
            width: int = 384,
            patch_size: int = 16,
            n_classes: int = N_CLASSES,
            hidden_dim: int = HIDDEN_DIM,
            redout_type: str = REDOUT_TYPE,
            hook_layers: List[str] = HOOK_LAYERS,
            features: List[str] = FEATURES,
            bn_fusion_layers: bool = BN_FUSION_LAYERS,
            vit_model: str = VIT_MODEL) -> 'DensePredictionTransformer':
        args = Namespace(
            dpt_hidden_dim=hidden_dim,
            dpt_redout_type=redout_type,
            dpt_hook_layers=hook_layers,
            dpt_features=features,
            dpt_bn_fusion_layers=bn_fusion_layers,
            vit_model=VIT_MODEL)
        data_config = {'height': height, 'width': width, 'patch_size': patch_size, 'n_classes': n_classes}
        return DensePredictionTransformer(data_config, args)

    @staticmethod
    def add_to_argparse(parser: ArgumentParser) -> None:
        parser.add_argument('--dpt_hidden_dim', type=int, default=HIDDEN_DIM)
        parser.add_argument('--dpt_redout_type', type=str, default=REDOUT_TYPE,
            help='Which redout function to use [ignore|add|proj] to handle zero token and its interation with other pos tokens.')
        parser.add_argument('--dpt_hook_layers', type=int, nargs='+', default=HOOK_LAYERS)
        parser.add_argument('--dpt_features', type=int, nargs='+', default=FEATURES)
        parser.add_argument('--dpt_bn_fusion_layers', action='store_false', default=True)
        parser.add_argument('--vit_model', type=str, default=VIT_MODEL)


def get_activation_hook(layer_name: int) -> Callable:
    def activations_hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        DensePredictionTransformer.LAYER_ACTIVATIONS.append(output)
    return activations_hook
