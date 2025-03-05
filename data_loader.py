# data_loader.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler

def load_data(config, local_rank):
    """加载并预处理数据集
    Args:
        config (Config): 配置参数
        local_rank (int): 当前GPU编号
    Returns:
        DataLoader: 训练数据加载器
    """
    # 图像预处理流水线
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 正确归一化到[-1,1]
        transforms.Lambda(lambda x: x.half())  # 转换为FP16
    ])
    dataset = load_dataset(config.dataset_name, split='train')
    dataset = dataset.train_test_split(test_size=0.1)
    
    def transform_fn(examples):
        examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
        return examples
    
    dataset = dataset.map(transform_fn, batched=True)
    dataset.set_format(type='torch', columns=['image'])
    
    # 确保只在分布式环境中使用DistributedSampler
    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(
            dataset["train"],
            shuffle=True,
            rank=local_rank,
            num_replicas=torch.distributed.get_world_size()
        )
    else:
        train_sampler = None
    
    return dataset["train"]