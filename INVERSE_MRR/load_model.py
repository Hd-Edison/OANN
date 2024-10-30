import torch
import torch.nn as nn
import numpy as np
import pandas as pd


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


def load_model(input_value, model_path=None):
    if model_path is None:
        import os
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp_model.pth')
    # 加载模型
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 归一化输入
    input_value = np.array([input_value])
    input_tensor = torch.tensor(input_value, dtype=torch.float32)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 反归一化输出
    output_value = output_tensor.numpy()

    return output_value


if __name__ == '__main__':
    # 读取CSV文件
    data = pd.read_csv("dataset_Lc_gap.csv", header=None)

    # 提取第三列及其之后的20列数据
    input_values = data.iloc[:, 2:22].values
    import time

    start_time = time.time()

    # 预测并生成两列数据
    predictions = []
    for input_value in input_values:
        prediction = load_model(input_value)
        predictions.append(prediction.flatten())

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"程序运行时间为: {execution_time} 秒")

    # 将预测生成的两列数据转换为DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_gap (scaled)', 'Predicted_Lc (scaled)'])
    predictions_df.to_csv("predicted_dataset.csv", index=False)

    print("Predictions saved to predicted_dataset.csv")
