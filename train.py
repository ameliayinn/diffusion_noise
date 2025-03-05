# train.py
import os
import torch
import deepspeed
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from data_loader import load_data
from diffusion import linear_beta_schedule, forward_diffusion
from unet import UNet
import torchvision
import matplotlib.pyplot as plt

@torch.no_grad()
def generate_during_training(model_engine, save_dir, config, num_images=16):
    """在训练过程中生成样本并保存
    Args:
        model_engine: DeepSpeed 模型引擎
        save_dir (str): 保存样本的目录
        config: 配置对象
        num_images (int): 生成的样本数量
    """
    model_engine.eval()
    device = model_engine.device
    
    # 准备扩散参数
    betas = linear_beta_schedule(config.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_over_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # 生成初始噪声
    x = torch.randn(num_images, 3, config.image_size, config.image_size, device=device, dtype=torch.half)
    x = x.to(next(model_engine.parameters()).dtype)
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        pred_noise = model_engine(x, t_batch)
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        noise = torch.randn_like(x) if t > 0 else 0
        
        # 更新公式
        x = sqrt_one_over_alphas[t] * (x - beta_t * pred_noise / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise
    
    # 后处理并转换数据类型
    x = (x.clamp(-1, 1) + 1) * 0.5  # 将图像范围从 [-1, 1] 转换到 [0, 1]
    x = x.to(torch.float32)  # 确保转换为 float32
    grid = torchvision.utils.make_grid(x.cpu(), nrow=4)  # 将图像拼接成网格
    
    # 保存图像
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy())  # 数据现在是 float32
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, "samples.png"))
    plt.close()

def train_deepspeed(config):
    """DeepSpeed训练主函数"""
    # 初始化模型
    model = UNet(time_emb_dim=config.time_emb_dim)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # DeepSpeed配置 (移除scheduler部分)
    ds_config = {
        "train_batch_size": config.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.lr,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": config.fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True
        },
        "steps_per_print": 50,
        "gradient_clipping": 1.0
    }
    
    # 初始化引擎
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_dataset = load_data(config, local_rank)
    
    # 初始化DeepSpeed引擎
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config_params=ds_config,
        training_data=train_dataset,
        dist_init_required=True
    )
    
    # 手动创建PyTorch调度器 (关键修改)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    base_optimizer = optimizer.optimizer  # 访问底层PyTorch优化器
    scheduler = MultiStepLR(
        base_optimizer, 
        milestones=[500, 1000, 1500],  # 在epoch=500、1000、1500时衰减
        gamma=0.1  # 每次衰减为之前的0.1倍
    )
    
    # 准备扩散参数
    betas = linear_beta_schedule(config.timesteps).to(model_engine.device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model_engine.train()
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'sampler'):
            if isinstance(train_loader.batch_sampler.sampler, DistributedSampler):
                train_loader.batch_sampler.sampler.set_epoch(epoch)
        
        for batch in tqdm(train_loader):
            images = batch["image"].to(model_engine.device) # [B,3,64,64]
            images = images.to(torch.float16)
            t = torch.randint(0, config.timesteps, (images.size(0),)).to(model_engine.device)
            
            # 前向扩散
            noisy_images, noise = forward_diffusion(
                images, t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod
            )
            
            # 预测噪声
            pred_noise = model_engine(noisy_images, t)
            
            # 计算损失
            loss = F.mse_loss(pred_noise, noise)
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
        
        # 手动更新学习率 (关键修改)
        scheduler.step()
        
        # 保存检查点
        if model_engine.local_rank == 0:
            print(f"Current lr: {scheduler.get_last_lr()[0]:.8f}")  # 验证学习率变化
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            model_path = os.path.join(
                config.checkpoint_dir,
                f"model_bs_{config.batch_size}_ts_{config.timesteps}_epoch_{epoch+1}.pt"
            )
            torch.save(model_engine.module.state_dict(), model_path)
            
            sample_dir = os.path.join(
                config.samples_dir,
                f"bs_{config.batch_size}_ts_{config.timesteps}_epoch_{epoch+1}"
            )
            os.makedirs(sample_dir, exist_ok=True)
            generate_during_training(model_engine, sample_dir, config, num_images=16)
            
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Samples saved to {sample_dir}")