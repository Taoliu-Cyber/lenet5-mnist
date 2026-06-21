import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# 解决Matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_curves(train_loss_list, test_loss_list, train_acc_list, test_acc_list, save_dir):
    """绘制训练损失、准确率双曲线图"""
    epoch_num = len(train_loss_list)
    plt.figure(figsize=(12, 5))

    # 损失子图
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch_num+1), train_loss_list, 'b-o', label="训练损失", markersize=3)
    plt.plot(range(1, epoch_num+1), test_loss_list, 'r-o', label="测试损失", markersize=3)
    plt.xlabel("训练轮次 Epoch")
    plt.ylabel("Loss 损失值")
    plt.title("损失变化曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    # 准确率子图
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch_num+1), train_acc_list, 'b-o', label="训练准确率", markersize=3)
    plt.plot(range(1, epoch_num+1), test_acc_list, 'r-o', label="测试准确率", markersize=3)
    plt.xlabel("训练轮次 Epoch")
    plt.ylabel("Accuracy 准确率")
    plt.title("准确率变化曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(true_labels, pred_labels, num_classes, save_dir):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("测试集混淆矩阵")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

def print_metrics_report(true_labels, pred_labels):
    """打印分类评估指标：精确率、召回率、F1分数"""
    print("\n" + "="*60)
    print("MNIST测试集分类详细评估报告")
    print("="*60)
    print(classification_report(true_labels, pred_labels, digits=4))