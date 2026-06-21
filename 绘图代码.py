import matplotlib.pyplot as plt

# 设置图片清晰度、字体适配论文
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 每层信息：层名称、输出通道@尺寸、横坐标位置
layers_info = [
    {"name": "输入层\n1×32×32", "pos": 0},
    {"name": "C1 卷积层\n6@28×28", "pos": 1},
    {"name": "S2 最大池化\n6@14×14", "pos": 2},
    {"name": "C3 卷积层\n16@10×10", "pos": 3},
    {"name": "S4 最大池化\n16@5×5", "pos": 4},
    {"name": "C5 卷积层\n120@1×1", "pos": 5},
    {"name": "F6 全连接层\n84维特征", "pos": 6},
    {"name": "输出层\n10类数字", "pos": 7},
]

fig, ax = plt.subplots(figsize=(16, 4))
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(0, 1)
ax.axis("off")  # 隐藏坐标轴

# 绘制每层矩形框
box_w = 0.6
box_h = 0.6
for layer in layers_info:
    x = layer["pos"]
    # 绘制模块方框
    rect = plt.Rectangle((x - box_w/2, 0.2), box_w, box_h,
                         edgecolor="#225599", facecolor="#e6f0ff", linewidth=1.5)
    ax.add_patch(rect)
    # 文字标注
    ax.text(x, 0.5, layer["name"], ha="center", va="center", fontsize=10)

# 绘制层之间连接箭头
for i in range(len(layers_info)-1):
    x_start = layers_info[i]["pos"] + box_w/2
    x_end = layers_info[i+1]["pos"] - box_w/2
    ax.annotate("", xy=(x_end, 0.5), xytext=(x_start, 0.5),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))

# 总标题
ax.set_title("LeNet-5 完整网络结构数据流图", fontsize=13, pad=15)

plt.tight_layout()
plt.savefig("lenet5_structure.png", bbox_inches="tight")
plt.show()