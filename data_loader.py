import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(data_path='../Data', batch_size=128, num_workers=4):
    """
    创建 MNIST 的训练集与测试集 DataLoader。

    参数:
        data_path: MNIST 数据集的根目录（PyTorch 会在该目录下查找 MNIST 子目录）
        batch_size: 训练/测试的批大小
        num_workers: 数据加载使用的工作进程数

    返回:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root=data_path,
        train=True,
        download=False,  # 使用已有数据，不重新下载
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        download=False,  # 使用已有数据，不重新下载
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # 将张量固定到页锁内存以加速主机到 GPU 的传输
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # 将张量固定到页锁内存以加速主机到 GPU 的传输
    )

    return train_loader, test_loader
