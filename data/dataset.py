from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(data_root, batch_size, image_size=32, mean=(0.1307,), std=(0.3081,)):
    """
    获取MNIST训练集和测试集的DataLoader
    首次运行会自动下载数据集到data_root目录
    """
    # 数据预处理流水线
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 统一缩放到32×32
        transforms.ToTensor(),                        # 转为张量，像素值归一化到[0,1]
        transforms.Normalize(mean, std)               # 标准化
    ])

    # 加载训练集（自动下载）
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    # 加载测试集（自动下载）
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,   # 训练集打乱顺序
        num_workers=0   # Windows环境设为0，避免多进程报错
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集无需打乱
        num_workers=0
    )

    return train_loader, test_loader