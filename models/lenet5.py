import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    LeNet-5 经典卷积神经网络
    输入尺寸：1 × 32 × 32（单通道灰度图）
    输出尺寸：10（10个数字类别的概率）
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # C1：卷积层 + ReLU激活
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # S2：最大池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # C3：卷积层 + ReLU激活
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        # S4：最大池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # C5：卷积层 + ReLU激活
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        # F6：全连接层 + ReLU激活
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        # 输出层
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # 第一层卷积池化
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # 第二层卷积池化
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # 第三层卷积
        x = self.conv3(x)
        x = self.relu3(x)
        # 展平为一维向量
        x = x.view(-1, 120)
        # 全连接层
        x = self.fc1(x)
        x = self.relu4(x)
        # 输出分类结果
        x = self.fc2(x)
        return x

# 单独运行此文件可验证模型维度是否正确
if __name__ == '__main__':
    model = LeNet5()
    test_input = torch.randn(1, 1, 32, 32)
    output = model(test_input)
    print(f"输入张量形状: {test_input.shape}")
    print(f"输出张量形状: {output.shape}")