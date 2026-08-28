"""
ml/models/autoencoder.py
========================
Model B: Convolutional Autoencoder for Anomaly Detection & Heatmap Generation

Architecture:
    Encoder: Conv(3->32) -> Conv(32->64) -> Conv(64->128) -> Conv(128->256)
    Decoder: ConvT(256->128) -> ConvT(128->64) -> ConvT(64->32) -> ConvT(32->3)

    Input/output resolution: 128 x 128 RGB (resize applied externally)
    Bottleneck spatial dimension: 128 // 2^4 = 8 x 8

Design rationale:
  - Trained ONLY on clean images: the model learns the distribution of pristine
    image textures. At inference time, anything outside that distribution
    (defects, corruption, heavy noise) produces a high reconstruction error.
  - Pixel-wise error map = (original - reconstructed)^2, upsampled to the
    input image size, serves as the spatial anomaly heatmap (bonus feature).
  - Architecture depth chosen so the 8x8 bottleneck forces the encoder to
    summarise global structure, making it sensitive to large-scale defects
    and corruptions while staying fast enough for CPU inference.
  - Instance Norm instead of Batch Norm in the decoder: avoids artefacts
    at test-time with batch size = 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE     = 128          # model expects 128x128 RGB input
LATENT_DEPTH = 256          # bottleneck channel depth


class ConvBlock(nn.Module):
    """Encoder block: Conv2d + InstanceNorm + LeakyReLU + optional MaxPool."""
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))   # halves spatial dims
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    """Decoder block: ConvTranspose2d + InstanceNorm + ReLU."""
    def __init__(self, in_ch: int, out_ch: int, output_sigmoid: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
        ]
        if output_sigmoid:
            layers.append(nn.Sigmoid())
        else:
            layers += [
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.ReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = ConvBlock(3,   32,  pool=True)
        self.enc2 = ConvBlock(32,  64,  pool=True)
        self.enc3 = ConvBlock(64,  128, pool=True)
        self.enc4 = ConvBlock(128, LATENT_DEPTH, pool=True)

        # Decoder: 8 -> 16 -> 32 -> 64 -> 128
        self.dec1 = ConvTransposeBlock(LATENT_DEPTH, 128)
        self.dec2 = ConvTransposeBlock(128, 64)
        self.dec3 = ConvTransposeBlock(64,  32)
        self.dec4 = ConvTransposeBlock(32,  3,  output_sigmoid=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc4(self.enc3(self.enc2(self.enc1(x))))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec4(self.dec3(self.dec2(self.dec1(z))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor, return_heatmap: bool = False, top_k_ratio: float = 0.05):
        """
        Compute per-image MSE reconstruction error and localized anomaly metrics.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, 128, 128), values in [0, 1]
        return_heatmap : bool
            If True, also returns the per-pixel error map (B, 1, 128, 128).
        top_k_ratio : float
            Fraction of highest-error pixels used for localized anomaly score (default 5%).

        Returns
        -------
        anomaly_score : torch.Tensor, shape (B,) -- top-k% localized error
        heatmap (optional) : torch.Tensor, shape (B, 1, 128, 128)
        """
        with torch.no_grad():
            recon = self.forward(x)
            pixel_err = ((x - recon) ** 2).mean(dim=1, keepdim=True)   # (B,1,H,W)
            flat_err = pixel_err.view(pixel_err.shape[0], -1)          # (B, H*W)
            
            k = max(1, int(flat_err.shape[1] * top_k_ratio))
            topk_err, _ = torch.topk(flat_err, k=k, dim=1)
            anomaly_score = topk_err.mean(dim=1)                      # (B,)

        if return_heatmap:
            return anomaly_score, pixel_err
        return anomaly_score
