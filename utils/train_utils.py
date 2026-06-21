import torch

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch，返回平均损失和准确率
    """
    model.train()  # 设置为训练模式
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in dataloader:
        # 数据迁移到对应设备
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, dim=1)
        total_samples += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """
    在测试集上验证模型，返回损失、准确率、所有预测和真实标签
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # 关闭梯度计算，加速推理
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, dim=1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 保存所有结果用于后续指标计算
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy, all_predictions, all_labels