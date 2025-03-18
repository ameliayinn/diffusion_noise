# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 注册一个虚拟参数以获取正确的dtype
        self.register_buffer("dummy_param", torch.tensor(0.0))
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        
        # 通过虚拟参数获取dtype
        dtype = self.dummy_param.dtype
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -embeddings)
        
        time = time.to(dtype)  # 确保时间步类型一致
        embeddings = time[:, None] * embeddings[None, :]
        
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DownBlock(nn.Module):
    """修复后的下采样块，确保残差连接正确下采样
    每个下采样块包含：
    主路径：3x3 卷积 + 下采样 + 组归一化（GroupNorm）。
    残差路径：1x1 卷积 + 下采样。
    时间嵌入：通过线性层将时间嵌入向量映射到当前通道数，并用于调整特征图。
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        # 主路径：3x3卷积下采样
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        # 残差路径：1x1卷积下采样
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0)
        
        # 时间嵌入处理
        self.time_emb_proj = nn.Linear(time_emb_dim, out_ch * 2)
        self.act = nn.GELU()

    def forward(self, x, t_emb):
        h = self.conv(x)
        
        # 时间嵌入影响
        scale, shift = self.time_emb_proj(t_emb).chunk(2, dim=1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = self.act(h)
        
        # 残差连接（已下采样）
        return h + self.res_conv(x)

class UpBlock(nn.Module):
    """上采样块（带跳跃连接和时间嵌入）
    每个上采样块包含：
    上采样：使用最近邻插值将特征图分辨率加倍。
    跳跃连接：将下采样路径中对应分辨率的特征图与当前特征图拼接。
    主路径：3x3 卷积 + 组归一化（GroupNorm）。
    时间嵌入：通过线性层将时间嵌入向量映射到当前通道数，并用于调整特征图。
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
        )
        self.time_emb_proj = nn.Linear(time_emb_dim, out_ch * 2)
        self.act = nn.GELU()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        h = self.conv(x)
        # 处理时间嵌入
        scale, shift = self.time_emb_proj(t_emb).chunk(2, dim=1)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = self.act(h)
        return h

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 时间嵌入处理（64 -> 128），将时间步 t 转换为一个高维向量，用于在网络的每一层中引入时间信息。
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(64), # 将时间步 t 映射为一个 64 维的正弦位置编码。
            nn.Linear(64, 128), # 将 64 维的编码映射到 128 维。
            nn.GELU(), # 使用 GELU 激活函数。
            nn.Linear(128, 128), # 进一步映射到 128 维，作为最终的时间嵌入向量。
        )
        
        # 下采样路径：逐步降低特征图的分辨率，同时增加通道数，提取高层次的特征。
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1) # 将输入的 3 通道图像通过一个 3x3 卷积层，映射到 64 通道。
        self.down1 = DownBlock(64, 128, time_emb_dim=128) # 三个下采样块，每个块将输入的特征图分辨率减半，同时将通道数翻倍（64 -> 128 -> 256 -> 512）。
        self.down2 = DownBlock(128, 256, time_emb_dim=128)
        self.down3 = DownBlock(256, 512, time_emb_dim=128)
        
        # 中间层（带时间嵌入）：在下采样和上采样之间引入一个中间层，进一步提取特征。
        self.mid_conv1 = nn.Conv2d(512, 512, 3, padding=1) # 对 512 通道的特征图进行 3x3 卷积。
        self.mid_norm = nn.GroupNorm(8, 512) # 对特征图进行组归一化（GroupNorm）。
        self.mid_time_proj = nn.Linear(128, 512 * 2) # 将时间嵌入向量映射到 1024 维（512 * 2），并将其拆分为 scale 和 shift 两部分。
        self.mid_act = nn.GELU() # 使用 GELU 激活函数。
        
        # 上采样路径：逐步恢复特征图的分辨率，同时减少通道数，生成与输入图像尺寸相同的输出。
        self.up1 = UpBlock(512, 256, time_emb_dim=128) # 三个上采样块，每个块将输入的特征图分辨率加倍，同时将通道数减半（512 -> 256 -> 128 -> 64）。
        self.up2 = UpBlock(256, 128, time_emb_dim=128)
        self.up3 = UpBlock(128, 64, time_emb_dim=128)
        
        # 最终输出层：将上采样路径输出的 64 通道特征图映射回 3 通道（RGB 图像）。
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), # 对特征图进行 3x3 卷积。
            nn.GroupNorm(8, 64), # 对特征图进行组归一化（GroupNorm）。
            nn.GELU(), # 使用 GELU 激活函数。
            nn.Conv2d(64, 3, 1), # 使用 1x1 卷积将 64 通道映射到 3 通道。
        )

    def forward(self, x, t):
        x = x.to(next(self.parameters()).dtype)
        t = t.to(next(self.parameters()).dtype)
        t = t.to(next(self.parameters()).dtype)
        t_embed = self.time_embed(t)  
        
        # 下采样路径
        x1 = F.gelu(self.conv1(x))  # [B,64,64,64]
        x2 = self.down1(x1, t_embed)
        x3 = self.down2(x2, t_embed)
        x = self.down3(x3, t_embed)
        
        # 中间处理
        x = self.mid_conv1(x)
        # 应用时间嵌入
        scale, shift = self.mid_time_proj(t_embed).chunk(2, dim=1)
        x = x * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        x = self.mid_norm(x)
        x = self.mid_act(x)
        
        # 上采样路径
        x = self.up1(x, x3, t_embed)
        x = self.up2(x, x2, t_embed)
        x = self.up3(x, x1, t_embed)
        
        return self.final_conv(x)