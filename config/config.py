import torch

class Config:
    # 运行设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据集参数
    data_root = './data'        # MNIST数据集自动下载存放路径
    num_classes = 10            # 数字0-9共10类
    image_size = 32             # LeNet-5标准输入尺寸
    mean = (0.1307,)            # MNIST数据集均值
    std = (0.3081,)             # MNIST数据集标准差
    
    # 训练超参数
    batch_size = 64             # 批次大小
    epochs = 20                 # 训练总轮次
    learning_rate = 0.001       # 学习率
    
    # 输出路径
    save_model_path = './output/lenet5_mnist.pth'
    save_figure_dir = './output'