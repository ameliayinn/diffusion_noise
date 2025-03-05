from torchvision import transforms
from datasets import load_dataset
from torch.utils.data.distributed import DistributedSampler
import os

def load_data(config, local_rank):
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 正确归一化到[-1,1]
        transforms.Lambda(lambda x: x.half())  # 转换为FP16
    ])
    
    dataset = load_dataset(config.dataset_name, split='train').train_test_split(test_size=0.1)
    dataset = dataset.map(lambda ex: {"image": [transform(img.convert("RGB")) for img in ex["image"]]}, batched=True)
    dataset.set_format(type='torch', columns=['image'])
    
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(
            dataset["train"],
            shuffle=True,
            rank=local_rank,
            num_replicas=torch.distributed.get_world_size()
        )
    else:
        sampler = None
    
    return dataset["train"]