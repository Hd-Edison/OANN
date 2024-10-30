import pandas as pd
import matplotlib.pyplot as plt

# 读取损失数据文件
loss_data = pd.read_csv('loss_data.csv')

# 绘制训练损失和验证损失曲9
# 设置字体和字号
plt.rc('font', family='Times New Roman', size=18)

# 绘制损失曲线
plt.plot(loss_data['Epoch'], loss_data['Train Loss'], label='Train Loss')
plt.plot(loss_data['Epoch'], loss_data['Validation Loss'], label='Validation Loss')
# 设置坐标轴的粗细
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# 设置坐标轴刻度的粗细
plt.tick_params(axis='both', which='major', width=2)

# 增大坐标值的字体大小
plt.tick_params(axis='both', labelsize=18)  # 增加刻度标签的字体大小
# 添加图例和标签
plt.xlabel('Epoch', fontsize=18, fontweight='bold')
plt.ylabel('Loss', fontsize=18, fontweight='bold')
# 添加图例并加粗标签
plt.legend(prop={'size': 18, 'weight': 'bold'}, loc='best')
plt.grid(True)
plt.tight_layout()
# 保存图像
plt.savefig('loss_plot.png')  # 保存为 loss_plot.png 文件

# 显示图像
plt.show()
