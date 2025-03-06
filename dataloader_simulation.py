import torch
from datasets import Dataset
from torch.utils.data.distributed import DistributedSampler

def load_data(config, local_rank, seed=42):
    """加载并生成模拟数据集
    Args:
        config (Config): 配置参数
        local_rank (int): 当前GPU编号
        seed (int): 随机种子，用于控制数据生成的可重复性
    Returns:
        Dataset: Hugging Face Dataset 对象（train 部分）
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)

    # 定义高维正态分布的参数
    image_size = config.image_size
    dim = 3 * image_size * image_size  # 确保维度可以被 reshape 为图像格式
    num_samples = 100  # 每个数据集的样本数量

    # 第一个数据集的均值和协方差矩阵
    mu1 = torch.ones(dim) * 5  # 均值
    sigma1 = torch.eye(dim) * 2  # 协方差矩阵

    # 第二个数据集的均值和协方差矩阵
    mu2 = torch.ones(dim) * 10  # 均值
    sigma2 = torch.eye(dim) * 4  # 协方差矩阵

    # 从高维正态分布中生成数据
    data1 = torch.distributions.MultivariateNormal(mu1, sigma1).sample((num_samples,))
    data2 = torch.distributions.MultivariateNormal(mu2, sigma2).sample((num_samples,))

    # 将两个数据集合并
    data = torch.cat([data1, data2], dim=0)
    labels = torch.cat([torch.zeros(num_samples), torch.ones(num_samples)])

    # 将数据 reshape 为图像格式
    data = data.view(-1, 3, image_size, image_size)

    # 创建 Hugging Face Dataset
    dataset = Dataset.from_dict({
        "image": [img for img in data],  # 将数据转换为列表形式
        "label": labels.tolist()  # 将标签转换为列表形式
    })

    # 划分训练集和测试集
    dataset = dataset.train_test_split(test_size=0.1)

    # 保存初始模拟数据到文件
    with open("simulation.txt", "w") as f:
        f.write("Data1:\n")
        f.write(str(data1.numpy()))  # 将数据转换为 NumPy 数组并保存
        f.write("\n\nData2:\n")
        f.write(str(data2.numpy()))
        f.write("\n\nData:\n")
        f.write(str(data.numpy()))

    # 返回训练集部分
    return dataset["train"]