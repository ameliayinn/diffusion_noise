# generate.py
import torch
import torchvision
import matplotlib.pyplot as plt
from unet import UNet
from diffusion import linear_beta_schedule
import os
import numpy as np

# 还未更改
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
    x1 = torch.randn(num_images, 3, config.image_size, config.image_size, device=device, dtype=torch.half)
    x2 = torch.randn(num_images, 3, config.image_size, config.image_size, device=device, dtype=torch.half)
    x1 = x1.to(next(model_engine.parameters()).dtype)
    x2 = x2.to(next(model_engine.parameters()).dtype)
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        
        # 分别对两份数据进行去噪
        pred_noise1 = model_engine.module.model1(x1, t_batch)  # 使用第一个 UNet
        pred_noise2 = model_engine.module.model2(x2, t_batch)  # 使用第二个 UNet
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        noise1 = torch.randn_like(x1) if t > 0 else 0
        noise2 = torch.randn_like(x2) if t > 0 else 0
        
        # 更新公式
        x1 = sqrt_one_over_alphas[t] * (x1 - beta_t * pred_noise1 / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise1
        x2 = sqrt_one_over_alphas[t] * (x2 - beta_t * pred_noise2 / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise2
    
    # 融合两个去噪结果（简单加权平均）
    x = 0.5 * x1 + 0.5 * x2
    
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

@torch.no_grad()
def generate_during_training_simulation(model_engine, save_dir, config, num_images=16):
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
    x1 = torch.randn(num_images, 1, config.image_size, config.image_size, device=device, dtype=torch.half)  # 第一份噪声
    x2 = torch.randn(num_images, 1, config.image_size, config.image_size, device=device, dtype=torch.half)  # 第二份噪声
    x1 = x1.to(next(model_engine.parameters()).dtype)
    x2 = x2.to(next(model_engine.parameters()).dtype)
    
    # 反向扩散过程
    for t in reversed(range(0, config.timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)
        
        # 分别对两份数据进行去噪
        pred_noise1 = model_engine.module.model1(x1, t_batch)  # 使用第一个 UNet
        pred_noise2 = model_engine.module.model2(x2, t_batch)  # 使用第二个 UNet
        
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        noise1 = torch.randn_like(x1) if t > 0 else 0
        noise2 = torch.randn_like(x2) if t > 0 else 0
        
        # 更新公式
        x1 = sqrt_one_over_alphas[t] * (x1 - beta_t * pred_noise1 / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise1
        x2 = sqrt_one_over_alphas[t] * (x2 - beta_t * pred_noise2 / sqrt_one_minus_alphas_cumprod[t]) + torch.sqrt(beta_t) * noise2
    
    # 融合两个去噪结果（简单加权平均）
    x = 0.5 * x1 + 0.5 * x2
    
    # 后处理并转换数据类型
    # x = (x.clamp(-1, 1) + 1) * 0.5  # 将图像范围从 [-1, 1] 转换到 [0, 1]
    x = x.to(torch.float32)  # 确保转换为 float32
    
    with open(os.path.join(save_dir, "sample.txt"), "w") as f:
        res_list = []
        for i in range(num_images):
            sample = x[i].cpu().numpy()
            mean_list = []
            for row in sample:
                # print("*****" ,type(row), row.shape) # <class 'numpy.ndarray'> (16, 16)
                # f.write(" ".join(f"{float(val):.7f}" for val in row.reshape(-1))) # flaten row to one dimension
                
                mean_row = np.mean(row)
                mean_list.append(mean_row)
                
                '''
                flattened_row = row.flatten().tolist()
                for item in flattened_row:
                    res_list.append(item)
                '''
            
            mean_res = np.mean(mean_list)
            res_list.append(mean_res)
            
        f.write(str(res_list))
    
    # 确定横坐标范围
    min_value = min(res_list)
    max_value = max(res_list)
    with open(os.path.join(save_dir, "range.txt"), "w") as f:
        content = 'min: ' + str(min_value) + ', max: ' + str(max_value)
        f.write(content)
    bins = np.arange(min_value, max_value + 1, 1)  # 左闭右开区间
    
    # 计算频次
    hist, bin_edges = np.histogram(res_list, bins=bins)

    # 或者使用 Counter 计算频次
    # counter = Counter(res_list)
    # hist = [counter.get(i, 0) for i in range(min_value, max_value)]
    # bin_edges = np.arange(min_value, max_value + 1)
    
    # 绘制柱状图
    plt.bar(bin_edges[:-1], hist, width=0.8, align='edge', edgecolor='black')

    # 设置图表标题和坐标轴标签
    plt.title("Frequency Distribution")
    plt.xlabel("Value Range")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_dir, "samples.png"))