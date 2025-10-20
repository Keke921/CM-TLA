import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os



# 设置 Seaborn 风格
sns.set(style="whitegrid") #, font_scale=1.3

# 方法和数据
methods = ['CE', 'Focal', 'LDAM', 'CB', 'LADE', 'LA']
head_acc = [86.1, 85.5, 86.4, 82.3, 81.2, 81.3]
med_acc = [68.7, 69.1, 66.5, 76.3, 76.7, 77.4]
tail_acc = [42.1, 44.7, 33.4, 63.5, 73.4, 73.4]

x = np.arange(len(methods))
bar_width = 0.25

# 配色
#palette = sns.color_palette("Set2")
palette = ['#EA8379', '#7DAEE0', '#B395BD']
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

# 创建图
fig, ax = plt.subplots(figsize=(15, 5))

bars1 = ax.bar(x - bar_width, head_acc, width=bar_width, label='Head', color=palette[0])
bars2 = ax.bar(x, med_acc, width=bar_width, label='Medium', color=palette[1])
bars3 = ax.bar(x + bar_width, tail_acc, width=bar_width, label='Tail', color=palette[2])

# 添加数值标注
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=20)

annotate_bars(bars1)
annotate_bars(bars2)
annotate_bars(bars3)

# 坐标轴设置
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(28, 100)
#ax.set_title('Per-Group Accuracy Comparison Across Methods', fontsize=26)

# 图例放在内部右上角
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.01), ncol=3, frameon=True)

# 网格线
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.xaxis.grid(False)
# 紧凑布局
fig.tight_layout()

# 保存 PDF 文件到当前目录
pdf_path = os.path.abspath("1a_per_group_accuracy.pdf")
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

# 确认保存成功
print("PDF saved to:", pdf_path)

# 显示图
plt.show()

# 可选：关闭图防止资源泄露
plt.close(fig)