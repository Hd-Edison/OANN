import pandas as pd
import matplotlib.pyplot as plt

# 读取损失数据文件
loss_data = pd.read_csv('loss_data.csv')

# 绘制训练损失和验证损失曲9
# 设置字体和字号
plt.rc('font', family='Times New Roman', size=14)

# 绘制损失曲线
plt.plot(loss_data['Epoch'], loss_data['Train Loss'], label='Train Loss')
plt.plot(loss_data['Epoch'], loss_data['Validation Loss'], label='Validation Loss')

# 添加图例和标签
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
# 保存图像
plt.savefig('loss_plot.png')  # 保存为 loss_plot.png 文件

# 显示图像
plt.show()
