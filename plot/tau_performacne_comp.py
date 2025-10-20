import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# 设置全局字体大小和Seaborn风格
plt.rcParams.update({
    'font.size': 24, 
    'axes.titlesize': 24, 
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 30
})
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})

# Seaborn调色板定义
sns_colors = sns.color_palette("colorblind")
palette = {
    0: sns_colors[0],   # 蓝色
    0.5: sns_colors[1], # 橙色
    1: sns_colors[2],   # 绿色
    1.5: sns_colors[3], # 红色
    2: sns_colors[4],   # 紫色
}
head_color = sns.light_palette("crimson", 3)[1]  # 浅红色头类区域
tail_color = sns.light_palette("dodgerblue", 3)[1]  # 浅蓝色尾类区域

# 数据
data_lines = [
    "100 98 98 92 89 96 97 84 95 98 80 81 95 95 97 93 93 95 80 92 93 91 91 90 85 70 88 74 83 81 91 88 78 63 85 61 95 84 85 90 83 94 80 89 71 64 72 60 97 92 64 83 81 92 95 24 94 71 89 59 81 37 89 49 49 67 71 47 94 63 82 63 24 43 14 57 62 53 70 44 42 20 86 25 69 71 77 56 62 69 36 26 39 42 16 4 37 60 9 32",
    "98  98 95 89 84 94 95 80 94 98 77 79 94 91 97 93 92 92 79 90 94 91 91 89 91 74 84 73 87 80 89 90 81 62 88 73 95 87 85 90 85 93 80 89 74 72 81 59 98 93 68 86 77 95 97 40 98 78 94 62 84 66 87 61 58 73 72 55 97 76 87 82 40 56 30 77 80 73 76 68 57 49 91 72 40 76 80 90 71 81 81 69 46 58 69 44 11 60 21 55",
    "94  95 91 80 62 92 91 77 93 97 64 70 92 88 97 92 90 89 77 91 93 94 88 79 91 76 74 69 87 81 77 92 71 57 85 80 93 87 79 89 83 89 77 88 78 78 88 53 98 92 69 85 76 95 93 46 97 83 95 66 85 83 87 68 60 79 79 59 98 81 88 88 57 58 47 86 90 87 82 79 62 70 93 80 68 82 84 94 79 88 90 84 60 65 84 72 30 77 40 64",
    "87  83 89 63 33 88 83 70 90 95 45 70 85 69 95 91 90 85 70 88 90 95 87 69 91 77 66 61 87 75 53 91 59 49 81 84 89 86 72 87 81 85 60 90 75 77 91 48 98 92 64 84 77 92 89 36 94 84 95 68 85 90 79 68 63 80 78 60 99 82 86 96 61 55 60 86 92 90 83 85 70 79 94 88 84 86 87 96 86 92 94 91 71 73 91 85 55 81 62 64",
    "51  65 79 55 15 79 72 65 85 95 32 63 70 41 84 89 78 80 67 84 88 87 80 60 87 70 52 36 87 67 20 88 45 31 72 84 80 75 62 81 76 78 50 76 71 73 89 40 97 88 55 84 80 84 83 31 92 83 95 57 83 94 69 58 62 82 78 53 99 81 83 92 66 58 63 87 90 92 88 87 71 79 96 92 90 88 87 96 92 96 94 94 83 72 88 88 70 84 83 58"
]

# 正确的τ值列表
tau_values = [0, 0.5, 1, 1.5, 2]
data_dict = {}

# 解析数据（每行直接是100个数据点）
for i, line in enumerate(data_lines):
    # 处理数据中的中文"极"字符（原意为"very"）
    cleaned_line = line.replace('极', '')
    values = list(map(float, cleaned_line.split()))
    data_dict[tau_values[i]] = values

# 创建图表
fig, ax = plt.subplots(figsize=(20, 10))

# 自定义τ=1.5的线型 - 更加明显的点划线样式
line_styles = {
    0: '-',     # 实线
    0.5: '--',  # 虚线
    1: '-.',    # 点划线
    1.5: (0, (5, 2, 1, 2)),  # 自定义线型：长虚线+点
    2: '-'      # 实线
}

line_widths = {
    0: 3.5,
    0.5: 3.5,
    1: 3.5,
    1.5: 4.0,  # 加粗τ=1.5的线条
    2: 3.5
}

# 绘制折线
for tau in tau_values:
    ax.plot(range(100), data_dict[tau], 
            linewidth=line_widths[tau],
            linestyle=line_styles[tau],
            color=palette[tau],
            label=f'τ = {tau}')

# 关键区域标记
head_start, head_end = 0, 36
tail_start, tail_end = 71, 99
ax.axvspan(head_start, head_end, alpha=0.1, color=head_color, label='Head Classes (0-36)')
ax.axvspan(tail_start, tail_end, alpha=0.1, color=tail_color, label='Tail Classes (71-99)')

# 设置图表样式
ax.set_title('Performance Variation Across τ Values', fontsize=26, fontweight='bold', pad=25)
ax.set_xlabel('Class Index', fontsize=22)
ax.set_ylabel('Performance (%)', fontsize=22)
ax.grid(True)

# 设置坐标轴
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.set_ylim(0, 105)
ax.set_xlim(-1, 100)

# 添加关键统计量
head_mean_tau0 = np.mean(data_dict[0][head_start:head_end+1])
tail_mean_tau0 = np.mean(data_dict[0][tail_start:tail_end+1])
head_mean_tau1_5 = np.mean(data_dict[1.5][head_start:head_end+1])
tail_mean_tau1_5 = np.mean(data_dict[1.5][tail_start:tail_end+1])
head_mean_tau2 = np.mean(data_dict[2][head_start:head_end+1])
tail_mean_tau2 = np.mean(data_dict[2][tail_start:tail_end+1])

'''
# 添加统计框
head_delta = head_mean_tau2 - head_mean_tau0
tail_delta = tail_mean_tau2 - tail_mean_tau0
balance_point = "τ=1.5 Head: %.1f%%\nTail: %.1f%%" % (head_mean_tau1_5, tail_mean_tau1_5)

stats_text = (f'Head Classes (0-36):\n'
              f'τ=0: {head_mean_tau0:.1f}%   τ=2: {head_mean_tau2:.1f}%   Δ: {head_delta:+.1f}%\n\n'
              f'Tail Classes (71-99):\n'
              f'τ=0: {tail_mean_tau0:.1f}%   τ=2: {tail_mean_tau2:.1f}%   Δ: {tail_delta:+.1f}%\n\n'
              f'{balance_point}')

ax.text(0.98, 0.20, stats_text, 
        transform=ax.transAxes, 
        fontsize=20,
        bbox=dict(boxstyle="round,pad=0.8", facecolor='white', edgecolor='gray', alpha=0.9),
        verticalalignment='top',
        horizontalalignment='right')
'''

# 添加分区界线
ax.axvline(head_end, color=head_color, linestyle='--', alpha=0.8, linewidth=2)
ax.axvline(tail_start, color=tail_color, linestyle='--', alpha=0.8, linewidth=2)
#ax.text(head_end+2, 102, f'Head End\n(Class {head_end})', color=head_color, fontsize=18)
#ax.text(tail_start-15, 102, f'Tail Start\n(Class {tail_start})', color=tail_color, fontsize=18)

# 添加图例 - 单排水平放置
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.12), 
          ncol=5, fancybox=True, shadow=True, fontsize=20)

'''
# 特别标记τ=1.5线型说明（图例中）
ax.text(0.02, 0.98, 'τ=1.5: Custom dash-dot line',
        transform=ax.transAxes, fontsize=16, color=palette[1.5],
        bbox=dict(facecolor='white', alpha=0.8, edgecolor=palette[1.5]))
'''

# 优化布局并保存
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # 为顶部图例留出空间
plt.savefig('tau_performance_final.pdf', dpi=300, bbox_inches='tight')
plt.show()