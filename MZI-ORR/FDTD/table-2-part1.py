import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import keras
from keras.models import model_from_json
import tensorflow as tf


@keras.saving.register_keras_serializable()
def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the true and predicted values.

    Parameters:
        y_true (tensor-like): True target values.
        y_pred (tensor-like): Predicted target values.

    Returns:
        float: Root Mean Squared Error (RMSE) between y_true and y_pred.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def load_file(json_file, weights):
    # Load Model architecture from JSON file
    with open(json_file, "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)
    return loaded_model


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


# 加载模型
model = MLP()
model.load_state_dict(torch.load('mlp_model.pth'))
model.eval()


def predict(input_value):
    # 归一化输入
    input_value = np.array([input_value])
    input_tensor = torch.tensor(input_value, dtype=torch.float32)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    # 反归一化输出
    output_value = output_tensor.numpy()

    return output_value


# 读取CSV文件
data = pd.read_csv("E:\\jiatao\\OANN\\MZI-ORR\\FDTD\\dataset_Lc_gap.csv", header=None)

# 随机选取20行数据
sampled_data = data.sample(n=20, random_state=42)  # random_state可以确保结果可复现

# 记录选取的行号
sampled_indices = sampled_data.index.tolist()

# 提取第三列及其之后的20列数据
input_values = sampled_data.iloc[:, 2:22].values

# 预测并生成两列数据
predictions = []
for input_value in input_values:
    prediction = predict(input_value)
    predictions.append(prediction.flatten())

# 将预测生成的两列数据转换为DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Column1', 'Predicted_Column2'])

# 保存到新的CSV文件
predictions_df.to_csv("table-2.csv", index=False)

### load Model
loaded_model = load_file("forward model\\model2.json", "forward model\\model2.weights.h5")

y_pred = loaded_model.predict(predictions)
print("Predictions saved to table-2.csv")
