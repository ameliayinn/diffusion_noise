# train.py
import os
import torch
import deepspeed
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusion import linear_beta_schedule, forward_diffusion_with_different_noise as forward_diffusion
from datasets import concatenate_datasets
from unet import UNetSimulation
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import json
from collections import Counter

def train_deepspeed(config):
    
    """DeepSpeed训练主函数"""
        
    # 初始化引擎
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_dataset1, train_dataset2 = load_data(config, local_rank)
    train_dataset = concatenate_datasets([train_dataset1, train_dataset2])
    
    # 初始化DeepSpeed引擎
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = list(model1.parameters()) + list(model2.parameters())  # 合并参数
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=[model1, model2],  # 传入两个模型
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
    
    # 提示开始
    print(f"****START TRAINING****\nimage_size: {config.image_size}, batch_size: {config.batch_size}, timesteps: {config.timesteps}, time_emb_dim: {config.time_emb_dim}")
    
    # 训练循环
    for epoch in range(config.num_epochs):
        model_engine.train()
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'sampler'):
            if isinstance(train_loader.batch_sampler.sampler, DistributedSampler):
                train_loader.batch_sampler.sampler.set_epoch(epoch)
        
        for batch in tqdm(train_loader):
            images1 = # 第一份数据
            images2 = # 第二份数据
            # images = images.unsqueeze(1)  # 增加通道维度，形状变为 [batch_size, 1, 10, 10]
            images1 = images1.to(torch.float16)
            images2 = images2.to(torch.float16)
            # print(type(images))  # 应该是 <class 'torch.Tensor'>
            # print(images.shape)  # 应该是 [B, 1, H, W]
            t = torch.randint(0, config.timesteps, (images1.size(0),)).to(model_engine.device)
            
            # 前向扩散
            noisy_images1, noisy_images2, noise1, noise2 = forward_diffusion(
                images1, images2, t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod
            )
            
            # 预测噪声
            pred_noise1 = model_engine.module[0](noisy_images1, t)  # 使用第一个 UNet
            pred_noise2 = model_engine.module[1](noisy_images2, t)  # 使用第二个 UNet
            
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