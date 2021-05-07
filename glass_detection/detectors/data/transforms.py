import torchvision.transform as T
import torch.nn.functional as F



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
