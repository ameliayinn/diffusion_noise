from datasets import get_dataset_config_names

# 获取某个数据集的所有配置
configs = get_dataset_config_names("your_dataset_name")
print(configs)