def train_deepspeed(config):
    """函数引用"""
    def get_function(type):
        if type == 'simulation':
            from archive.dataloader_simulation import load_data
            from generate import generate_during_training_simulation as generate_during_training
        else:
            from utils.dataloader import load_data
            from generate import generate_during_training
        return load_data, generate_during_training
    
    """DeepSpeed训练主函数""" 
    # 初始化模型
    model = UNetSimulation(time_emb_dim=config.time_emb_dim, image_size=config.image_size)
    
    # get fuction
    data_type = config.dataset_name
    load_data, generate_during_training = get_function(data_type)
    
    # 初始化引擎
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    train_dataset = load_data(config, local_rank)
    # train_dataset = load_data(config, local_rank, seed=42) # 使用固定种子
    # train_dataset_random = load_data(config, local_rank=0, seed=None)  # 不使用固定种子
    
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
            images = batch["image"].to(model_engine.device)
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
            if (epoch + 1) % 20 == 0:
                os.makedirs(config.samples_dir, exist_ok=True)
                
                # 生成样本
                sample_dir = os.path.join(
                    config.samples_dir,
                    f"is_{config.image_size}_bs_{config.batch_size}_tstep_{config.timesteps}_tdim_{config.time_emb_dim}_epoch_{epoch+1}"
                )
                os.makedirs(sample_dir, exist_ok=True)
                
                generate_during_training(model_engine, sample_dir, config, num_images=config.num_images)