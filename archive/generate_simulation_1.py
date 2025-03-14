import torch
from datasets import Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt

seed = 42

# 设置随机种子以确保可重复性
torch.manual_seed(seed)

# 定义高维正态分布的参数
image_size = 8
# dim = 3 * image_size * image_size  # 确保维度可以被 reshape 为图像格式
dim = image_size * image_size
num_samples_1 = 10000  # 数据集的样本数量
num_samples_2 = 100

# 第一个数据集的均值和协方差矩阵
mu1 = torch.ones(dim) * 4  # 均值
sigma1 = torch.eye(dim) * 1  # 协方差矩阵

# 第二个数据集的均值和协方差矩阵
mu2 = torch.ones(dim) * 10  # 均值
sigma2 = torch.eye(dim) * 4  # 协方差矩阵

# 从高维正态分布中生成数据
data1 = torch.distributions.MultivariateNormal(mu1, sigma1).sample((num_samples_1,))
data2 = torch.distributions.MultivariateNormal(mu2, sigma2).sample((num_samples_2,))

# 将 data1 和 data2 reshape 成 10x10 的矩阵
# data1_reshaped = data1.view(num_samples_1, image_size, image_size)
# data2_reshaped = data2.view(num_samples_2, image_size, image_size)

# 如果你想保持原始数据的连续性，可以使用 reshape 方法
data1_reshaped = data1.reshape(num_samples_1, image_size, image_size)
data2_reshaped = data2.reshape(num_samples_2, image_size, image_size)


# 将两个数据集合并
data = torch.cat([data1, data2], dim=0)
labels = torch.cat([torch.zeros(num_samples_1), torch.ones(num_samples_2)])

# 将数据 reshape 为图像格式
# data = data.view(-1, 3, image_size, image_size)

# 创建 Hugging Face Dataset
dataset = Dataset.from_dict({
    "image": [img for img in data],  # 将数据转换为列表形式
    "label": labels.tolist()  # 将标签转换为列表形式
})

# 划分训练集和测试集
dataset = dataset.train_test_split(test_size=0.1)
dataset.set_format(type='torch', columns=['image'])

# 保存初始模拟数据到文件
with open("simulation.txt", "w") as f:
    f.write("Data1:\n")
    f.write(str(data1_reshaped[0].numpy()))  # 将数据转换为 NumPy 数组并保存
    f.write("\n\nData2:\n")
    f.write(str(data2_reshaped[0].numpy()))

res_list = []
for row in data:
    mean = np.mean(row.numpy())
    res_list.append(mean)

# 确定横坐标范围
min_value = min(res_list)
max_value = max(res_list)
with open("range.txt", "w") as f:
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
plt.savefig("samples.png")