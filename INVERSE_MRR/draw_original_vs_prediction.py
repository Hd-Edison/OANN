import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件并跳过第一列
df = pd.read_excel('original_vs_prediction.xls', header=None).iloc[:, 1:]

# 读取original和prediction行数据，并颠倒顺序
original = df.iloc[0].values[::-1][:20]  # 取前20个数据点
prediction = df.iloc[1].values[::-1][:20]  # 取前20个数据点

# 生成 x 轴从 1500 到 1600 的 20 个均匀分布的点
x_values = np.linspace(1500, 1600, 20)

# 设置字体为 Times New Roman，文本大小为14
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18

# 绘图
plt.plot(x_values, original, label='Original', linewidth=2.5)
plt.plot(x_values, prediction, label='Inverse result', color='orange', linestyle='--', linewidth=2.5)
# 设置坐标轴的粗细
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
# 设置x轴标签
plt.xlabel('Wavelength (um)')

# 设置x轴范围
plt.xlim(1500, 1600)

# 隐藏y轴标签
plt.ylabel('k')

# 不显示图表标题
plt.title('')

# 显示图例
plt.legend(prop={'size': 18, 'weight': 'bold'}, loc='best')
plt.tight_layout()
# 保存图像到本地
plt.savefig('original_vs_prediction.png')

# 显示图表
plt.show()
