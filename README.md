# LeNet-5 MNIST 手写数字识别（PyTorch复现）

## 📌 项目简介
本项目基于 PyTorch 深度学习框架复现经典 LeNet-5 卷积神经网络，在 MNIST 公开手写数字数据集上完成 0~9 十类手写数字的分类识别任务。
项目采用模块化工程结构，代码注释完整，支持 CPU/GPU 自动切换，Windows 环境全适配，可作为卷积神经网络入门学习与课程实验参考。

- 模型总参数量：61,706
- 测试集最高准确率：**99.11%**
- 默认训练轮次：20 Epoch

## 🛠️ 环境依赖
- Python 3.8 ~ 3.12
- PyTorch >= 1.10.0
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn

一键安装命令（国内清华镜像加速）：
```bash
pip install torch torchvision torchaudio numpy matplotlib seaborn scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
