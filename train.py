import os
import torch
import torch.nn as nn
import torch.optim as optim

from config.config import Config
from models.lenet5 import LeNet5
from data.dataset import get_mnist_dataloaders
from utils.train_utils import train_one_epoch, evaluate
from utils.visualization import plot_training_curves, plot_confusion_matrix, print_metrics_report

def main():
    opt = Config()
    # 创建输出文件夹
    os.makedirs(opt.save_figure_dir, exist_ok=True)

    print("="*55)
    print("LeNet-5 手写数字识别训练开始")
    print("="*55)
    print(f"运行设备: {opt.device}")
    print(f"训练轮次: {opt.epochs}")
    print(f"批次大小: {opt.batch_size}")
    print(f"初始学习率: {opt.learning_rate}")
    print("-"*55)

    # 1. 加载MNIST数据集（首次运行自动下载）
    print("正在加载MNIST数据集...")
    train_loader, test_loader = get_mnist_dataloaders(
        opt.data_root, opt.batch_size, opt.image_size, opt.mean, opt.std
    )
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print("-"*55)

    # 2. 初始化模型、损失函数、优化器
    model = LeNet5(opt.num_classes).to(opt.device)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    print("-"*55)

    # 3. 训练循环
    print("开始训练...\n")
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(opt.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, opt.device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, opt.device)

        # 记录历史数据
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        # 打印每轮结果
        print(f"Epoch [{epoch+1:2d}/{opt.epochs}]  "
              f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}  |  "
              f"测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f}")

    # 4. 最终评估与结果可视化
    print("\n" + "-"*55)
    print("训练完成，正在生成评估结果...")
    _, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, opt.device)
    print(f"最终测试准确率: {final_acc:.4f}")

    # 打印详细指标
    print_metrics_report(all_labels, all_preds)

    # 生成结果图
    plot_training_curves(
        train_loss_history, test_loss_history,
        train_acc_history, test_acc_history,
        opt.save_figure_dir
    )
    plot_confusion_matrix(all_labels, all_preds, opt.num_classes, opt.save_figure_dir)

    # 5. 保存模型权重
    torch.save(model.state_dict(), opt.save_model_path)
    print(f"\n模型权重已保存至: {opt.save_model_path}")
    print(f"训练曲线图、混淆矩阵已保存至: {opt.save_figure_dir} 文件夹")
    print("="*55)

if __name__ == '__main__':
    main()