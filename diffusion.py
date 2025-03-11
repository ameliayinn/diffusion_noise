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

def forward_diffusion_with_different_noise(x0_1, x0_2, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    对两份数据分别进行加噪
    Args:
        x0_1: 第一份数据 [B, C, H, W]
        x0_2: 第二份数据 [B, C, H, W]
        t: 时间步 [B]
        sqrt_alphas_cumprod: 累积 alpha 的平方根 [T]
        sqrt_one_minus_alphas_cumprod: 1 - 累积 alpha 的平方根 [T]
    Returns:
        noisy_x1: 加噪后的第一份数据
        noisy_x2: 加噪后的第二份数据
        noise1: 第一份数据的噪声
        noise2: 第二份数据的噪声
    """
    noise1 = torch.randn_like(x0_1)
    noise2 = torch.randn_like(x0_2)
    
    # 计算加噪后的数据
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    
    noisy_x1 = sqrt_alpha_cumprod_t * x0_1 + sqrt_one_minus_alpha_cumprod_t * noise1
    noisy_x2 = sqrt_alpha_cumprod_t * x0_2 + sqrt_one_minus_alpha_cumprod_t * noise2
    
    return noisy_x1, noisy_x2, noise1, noise2