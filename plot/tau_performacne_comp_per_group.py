import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置 Seaborn 风格
sns.set(style="whitegrid") #, font_scale=1.3

# 数据
taus = [0, 0.5, 1.0, 1.5, 2.0]
head = [88.5, 87, 83.9, 77.1, 66.3]
med = [74.7, 79.4, 81, 79.5, 74.9]
tail = [46.3, 63, 74.3, 80.8, 83.6]

labels = ['Head', 'Medium', 'Tail']
#colors = sns.color_palette("Set3")
colors = ['#EA8379', '#7DAEE0', '#B395BD']

# 全局字体设置
plt.rcParams.update({
    'font.size': 24,            # 所有字体基础大小
    'axes.titlesize': 26,       # 图标题字体
    'axes.labelsize': 24,       # 坐标轴标题字体
    'xtick.labelsize': 24,      # x轴刻度字体
    'ytick.labelsize': 24,      # y轴刻度字体
    'legend.fontsize': 24,      # 图例字体
    'figure.titlesize': 24,     # 整体图标题
})

# x轴位置设置
x = np.arange(len(taus))
bar_width = 0.2

fig, ax = plt.subplots(figsize=(15, 5))

# 三组柱子
rects1 = ax.bar(x - bar_width, head, width=bar_width, label='Head', color=colors[0])
rects2 = ax.bar(x, med, width=bar_width, label='Medium', color=colors[1])
rects3 = ax.bar(x + bar_width, tail, width=bar_width, label='Tail', color=colors[2])

# 添加柱子顶部标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + 1, f'{height:.1f}',
                ha='center', va='bottom', fontsize=20)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)


# 坐标轴设置
ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in taus], fontsize=24)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(30, 100)
#ax.set_title(r'Per-Group Accuracy Comparison Across $\tau$', fontsize=26)

# 图例放在内部右上角
ax.legend(loc='upper right', bbox_to_anchor=(1., 1.01), ncol=3, frameon=True)


# 网格线
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.xaxis.grid(False)

# 保存图像
plt.tight_layout()
plt.savefig("1b_tau_comparison.pdf", bbox_inches='tight')
plt.show()
