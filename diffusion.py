# diffusion.py
import torch
import numpy as np

def linear_beta_schedule(timesteps):
    """生成线性beta调度表
    Args:
        timesteps (int): 总时间步数
    Returns:
        betas (Tensor): [timesteps]
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def forward_diffusion(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """前向扩散过程（闭式解）
    Args:
        x0 (Tensor): 原始图像 [B,C,H,W]
        t (Tensor): 时间步 [B]
    Returns:
        noisy_images (Tensor): 加噪图像 [B,C,H,W]
        noise (Tensor): 添加的噪声 [B,C,H,W]
    """
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise