import os
import torch
import deepspeed
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from .unet import UNet
from .diffusion import linear_beta_schedule, forward_diffusion
from .data_loader import load_data

# def train_deepspeed(config):
#     # 初始化模型
#     model = UNet()
#     parameters = filter(lambda p: p.requires_grad, model.parameters())
    
#     # DeepSpeed配置
#     ds_config = {
#         "train_batch_size": config.batch_size,
#         "gradient_accumulation_steps": 1,
#         "optimizer": {"type": "AdamW", "params": {"lr": config.lr}},
#         "fp16": {"enabled": config.fp16},
#         "zero_optimization": {"stage": 2},
#         "steps_per_print": 50
#     }
    
#     # 初始化引擎
#     local_rank = int(os.getenv("LOCAL_RANK", 0))
#     train_dataset = load_data(config, local_rank)
    
#     engine, _, _, _ = deepspeed.initialize(
#         model=model,
#         model_parameters=parameters,
#         config_params=ds_config,
#         training_data=train_dataset,
#         dist_init_required=True
#     )
    
#     # 训练循环（保持原有训练逻辑）
#     # ... (保持原有训练循环实现)


def train_deepspeed(config):
    """DeepSpeed训练主函数
    调用路径：
    main -> train_deepspeed
              ├── UNet.forward
              ├── forward_diffusion
              └── load_data
    """
    # 初始化模型
    model = UNet()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # DeepSpeed配置
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
            "enabled": True,
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
    
    # 获取Dataset
    train_dataset = load_data(config, local_rank)
    
    # 初始化引擎
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config_params=ds_config,
        training_data=train_dataset,  # 传入Dataset对象
        dist_init_required=True
    )
    
    # 准备扩散参数
    betas = linear_beta_schedule(config.timesteps).to(model_engine.device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # 训练循环
    from tqdm import tqdm
    for epoch in range(config.num_epochs):
        model_engine.train()
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'sampler'):
            if isinstance(train_loader.batch_sampler.sampler, DistributedSampler):
                train_loader.batch_sampler.sampler.set_epoch(epoch)
        
        for batch in tqdm(train_loader):
            images = batch["image"].to(model_engine.device) # [B,3,64,64]
            images = images.to(torch.float16)
            # 随机采样时间步
            t = torch.randint(0, config.timesteps, (images.size(0),)).to(model_engine.device)  # [B]
            
            # 前向扩散
            noisy_images, noise = forward_diffusion(
                images, t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod
            )  # [B,3,64,64], [B,3,64,64]
            
            # 预测噪声
            pred_noise = model_engine(noisy_images, t)  # [B,3,64,64]
            
            # 计算损失
            loss = F.mse_loss(pred_noise, noise)
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
        
        # 保存检查点
        if model_engine.local_rank == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            model_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model_engine.module.state_dict(), model_path)
            
            # 生成样本
            sample_dir = os.path.join(config.samples_dir, f"epoch_{epoch+1}")
            os.makedirs(sample_dir, exist_ok=True)
            generate_during_training(model_engine, sample_dir, config, num_images=16)
            
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Samples saved to {sample_dir}")