# train.py
import os
import torch
import deepspeed
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusion import linear_beta_schedule, forward_diffusion as forward_diffusion
from datasets import concatenate_datasets
from unet import UNetSimulation
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import json
from collections import Counter

class CombinedModel(nn.Module):
    def __init__(self, model1, model2):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x, t):
        # 分别调用两个模型
        output1 = self.model1(x, t)
        output2 = self.model2(x, t)
        return output1, output2

def train_deepspeed(config):
    """函数引用"""
    def get_function(type):
        if type == 'simulation':
            from utils.dataloader_with_different_noise import load_data_simulation as load_data
            from generate_with_different_noise import generate_during_training_simulation as generate_during_training
        else:
            from utils.dataloader_with_different_noise import load_data
            from generate_with_different_noise import generate_during_training
        return load_data, generate_during_training
    
    """DeepSpeed训练主函数"""
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    # 创建带时间戳的 checkpoints, sample 和 logs 文件夹
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, f"checkpoints_{timestamp}")
    config.samples_dir = os.path.join(config.samples_dir, f"samples_{timestamp}")
    config.logs_dir = os.path.join(config.logs_dir, f"logs_{timestamp}")
    
    # 初始化模型
    model1 = UNetSimulation(time_emb_dim=config.time_emb_dim, image_size=config.image_size)
    model2 = UNetSimulation(time_emb_dim=config.time_emb_dim, image_size=config.image_size)
    
    model = CombinedModel(model1, model2)
    
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
    
    # get fuction
    data_type = config.dataset_name
    load_data, generate_during_training = get_function(data_type)
    
    # 初始化引擎
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_dataset1, train_dataset2 = load_data(config, local_rank)
    train_dataset = concatenate_datasets([train_dataset1, train_dataset2])
    # train_dataset = load_data(config, local_rank, seed=42) # 使用固定种子
    # train_dataset_random = load_data(config, local_rank=0, seed=None)  # 不使用固定种子
    
    # 初始化DeepSpeed引擎
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # parameters = list(model1.parameters()) + list(model2.parameters())  # 合并参数
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,  # 传入两个模型
        model_parameters=parameters,
        config_params=ds_config,
        training_data=train_dataset,
        dist_init_required=True
    )
    
    # 分别创建 DataLoader
    train_loader1 = DataLoader(train_dataset1, batch_size=config.batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size=config.batch_size, shuffle=True)

    # 使用 zip 合并两个 DataLoader
    train_loader = zip(train_loader1, train_loader2)
    
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
    
    # 提示开始
    print(f"****START TRAINING****\nimage_size: {config.image_size}, batch_size: {config.batch_size}, timesteps: {config.timesteps}, time_emb_dim: {config.time_emb_dim}")
    
    # 创建带有时间戳的路径
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)
    
    # 构造 CSV 文件路径
    csv_filename = f"is_{config.image_size}_bs_{config.batch_size}_tstep_{config.timesteps}_tdim_{config.time_emb_dim}.csv"
    csv_filepath = os.path.join(config.logs_dir, csv_filename)
    
    # 创建 CSV 文件并写入表头
    if model_engine.local_rank == 0:  # 只在主进程创建
        df = pd.DataFrame(columns=["epoch", "loss", "image_size", "batch_size", "timesteps", "time_emb_dim", "learning_rate"])
        df.to_csv(csv_filepath, index=False)
    
    with open(f'{csv_filepath[:-4]}.json', "w") as f:
        json.dump(vars(config), f, indent=4)  # 将 args 转换为字典并保存为 JSON
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model_engine.train()
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'sampler'):
            if isinstance(train_loader.batch_sampler.sampler, DistributedSampler):
                train_loader.batch_sampler.sampler.set_epoch(epoch)
        
        for batch1, batch2 in tqdm(train_loader):
            images1 = batch1["image"].to(model_engine.device)  # 第一份数据
            images2 = batch2["image"].to(model_engine.device)  # 第二份数据
            # images = images.unsqueeze(1)  # 增加通道维度，形状变为 [batch_size, 1, 10, 10]
            images1 = images1.to(torch.float16)
            images2 = images2.to(torch.float16)
            # print(type(images))  # 应该是 <class 'torch.Tensor'>
            # print(images.shape)  # 应该是 [B, 1, H, W]
            '''
            t = torch.randint(0, config.timesteps, (images1.size(0),)).to(model_engine.device)
            
            # 前向扩散
            noisy_images1, noisy_images2, noise1, noise2 = forward_diffusion(
                images1, images2, t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod
            )
            '''
            
            # 生成时间步长 t1 和 t2
            t1 = torch.randint(0, config.timesteps, (images1.size(0),)).to(model_engine.device)
            t2 = torch.randint(0, config.timesteps, (images2.size(0),)).to(model_engine.device)

            # 前向扩散
            noisy_images1, noise1 = forward_diffusion(images1, t1, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            noisy_images2, noise2 = forward_diffusion(images2, t2, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            
            # 预测噪声
            pred_noise1 = model_engine.module.model1(noisy_images1, t1)  # 使用第一个 UNet
            pred_noise2 = model_engine.module.model2(noisy_images2, t2)  # 使用第二个 UNet
            
            # 计算损失
            loss1 = F.mse_loss(pred_noise1, noise1)
            loss2 = F.mse_loss(pred_noise2, noise2)
            loss = loss1 + loss2  # 合并损失
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
        
        # 手动更新学习率 (关键修改)
        scheduler.step()
        
        # 保存检查点
        if model_engine.local_rank == 0:
            # print(f"Current lr: {scheduler.get_last_lr()[0]:.8f}")  # 验证学习率变化
            
            # 记录 epoch 结果到 CSV
            new_row = {
                "epoch": epoch + 1,
                "loss": loss.item(),
                "image_size": config.image_size,
                "batch_size": config.batch_size,
                "timesteps": config.timesteps,
                "time_emb_dim": config.time_emb_dim,
                "learning_rate": scheduler.get_last_lr()[0]
            }
            df = pd.read_csv(csv_filepath)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True) # 使用 pd.concat 追加数据
            df.to_csv(csv_filepath, index=False)
            
            # 保存模型检查点
            model_path = os.path.join(
                config.checkpoints_dir,
                f"model_is_{config.image_size}_bs_{config.batch_size}_tstep_{config.timesteps}_tdim_{config.time_emb_dim}_epoch_{epoch+1}.pt"
            )
            torch.save(model_engine.module.state_dict(), model_path)
            
            if (epoch + 1) % 20 == 0:
                os.makedirs(config.samples_dir, exist_ok=True)
                
                # 生成样本
                sample_dir = os.path.join(
                    config.samples_dir,
                    f"is_{config.image_size}_bs_{config.batch_size}_tstep_{config.timesteps}_tdim_{config.time_emb_dim}_epoch_{epoch+1}"
                )
                os.makedirs(sample_dir, exist_ok=True)
                
                generate_during_training(model_engine, sample_dir, config, num_images=config.num_images)
            
            # print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Samples saved to {sample_dir}")