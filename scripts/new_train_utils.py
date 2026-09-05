"""
train_utils.py
Differentiable PyTorch SSIM Loss & Impulse Noise Augmentation
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity (SSIM) Loss.
    Matches evaluate.py parameters (win_size=7, K1=0.01, K2=0.03).
    """

    def __init__(self, window_size: int = 7, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer("window", self._create_window(window_size, channel))

    def _gaussian(self, window_size: int, sigma: float = 1.5) -> torch.Tensor:
        gauss = torch.exp(
            -(torch.arange(window_size) - window_size // 2) ** 2 / (2 * sigma ** 2)
        )
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1d = self._gaussian(window_size).unsqueeze(1)
        _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2d.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2

        window = self.window.to(img1.device, dtype=img1.dtype)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                img1 * img1, window, padding=self.window_size // 2, groups=self.channel
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                img2 * img2, window, padding=self.window_size // 2, groups=self.channel
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                img1 * img2, window, padding=self.window_size // 2, groups=self.channel
            )
            - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - ssim_map.mean()


class CompositeLoss(nn.Module):
    """
    Weighted combination directly targeting the competition score:
    0.7 * L1 + 0.3 * (1 - SSIM)
    """

    def __init__(self, alpha_l1: float = 0.7, beta_ssim: float = 0.3):
        super().__init__()
        self.alpha = alpha_l1
        self.beta = beta_ssim
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=7, channel=3)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss_l1 = self.l1(pred, gt)
        loss_ssim = self.ssim_loss(pred, gt)
        return self.alpha * loss_l1 + self.beta * loss_ssim


def inject_synthetic_impulse_noise(
    tensor: torch.Tensor, prob: float = 0.008
) -> torch.Tensor:
    """
    Injects random hot/dead sensor spikes (0.5% - 1.0%) on the noisy inputs.
    Forces CNN kernels to ignore isolated extreme pixel values.
    """
    corrupted = tensor.clone()
    b, c, h, w = tensor.shape

    # Generate random binary masks for salt (1.0) and pepper (0.0)
    rand_noise = torch.rand((b, 1, h, w), device=tensor.device)
    hot_mask = rand_noise < (prob / 2.0)
    dead_mask = (rand_noise >= (prob / 2.0)) & (rand_noise < prob)

    corrupted = torch.where(hot_mask, torch.tensor(1.0, device=tensor.device), corrupted)
    corrupted = torch.where(dead_mask, torch.tensor(0.0, device=tensor.device), corrupted)
    return corrupted
