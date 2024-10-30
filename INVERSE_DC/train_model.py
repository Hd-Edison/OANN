import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv("DC_dataset_Lc_gap.csv", header=None)

# 划分输入和输出
X = data.iloc[:, 2:].values  # 后20列作为输入
y = data.iloc[:, :2].values  # 前两列作为输出

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = MLP()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00017)

# 转换数据为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 保存训练过程中的loss和val_loss
train_loss_history = []
val_loss_history = []

# 训练过程
num_epochs = 20000
import time
start_time = time.time()

for epoch in range(num_epochs):
    # 训练模式
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # 保存训练损失
    train_loss_history.append(loss.item())

    # 验证模式
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        val_loss_history.append(val_loss.item())

    # 打印训练信息
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

end_time = time.time()
execution_time = end_time - start_time

print(f"程序运行时间为: {execution_time} 秒")

# 保存模型
torch.save(model.state_dict(), 'mlp_model.pth')

# 保存损失数据到 CSV 文件
loss_data = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Train Loss': train_loss_history,
    'Validation Loss': val_loss_history
})

loss_data.to_csv('loss_data.csv', index=False)

# 绘制损失曲线并保存图片
plt.figure(figsize=(10, 6))
plt.plot(loss_data['Epoch'], loss_data['Train Loss'], label='Train Loss')
plt.plot(loss_data['Epoch'], loss_data['Validation Loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()

import subprocess

subprocess.run(["python", "load_model.py"])