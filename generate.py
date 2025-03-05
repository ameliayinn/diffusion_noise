import torch
import matplotlib.pyplot as plt
import torchvision
from .unet import UNet
from .diffusion import linear_beta_schedule

@torch.no_grad()
def generate_samples(config, model_path, num_images=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # 准备扩散参数
    betas = linear_beta_schedule(config.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # 生成过程（保持原有实现）
    # ... (保持原有生成逻辑)