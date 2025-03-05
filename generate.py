# generate.py
import torch
import torchvision
import matplotlib.pyplot as plt
from unet import UNet
from diffusion import linear_beta_schedule

@torch.no_grad()
def generate_samples(config, model_path, num_images=16):
    """样本生成函数
    调用路径：
    手动调用 -> UNet.forward
                └── 反向扩散过程
    Args:
        config (Config): 配置参数
        model_path (str): 模型路径
        num_images (int): 生成数量
    """
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))  # 加载到CPU再转移
    model.to(config.device).eval()
    
    # 扩散参数移动到设备
    betas = linear_beta_schedule(config.timesteps).to(config.device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_over_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    betas_cumprod = 1. - alphas_cumprod
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
    
    # 生成噪声
    x = torch.randn(num_images, 3, config.image_size, config.image_size).to(config.device)  # [B,3,64,64]
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=config.device)  # [B]
        pred_noise = model(x, t_batch)  # [B,3,64,64]
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        
        # 更新公式
        x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise
    
    # 后处理
    x = (x.clamp(-1, 1) + 1) * 0.5  # [0,1]范围
    grid = torchvision.utils.make_grid(x.cpu(), nrow=4)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig("generated_samples.png")
    plt.close()