import numpy as np
import tensorflow as tf
from main import OANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import csv


# 定义 H(a, b, c, d) 函数
def H(kr, phi, a, b, k1, k2, z_squared):
    oann = OANN(lam=1.55)
    a_new = a * 10
    b_new = b * 10
    return abs(oann.transfer_function_MZI_ORR(HR=oann.transfer_function_ORR(k=kr, gamma=0.8, neff=2.384556,
                                                                            phi=oann.calculate_phi(a_new * 2 * np.pi, b_new,
                                                                                                   z_squared=z_squared),
                                                                            L=5e-6 * 2 * np.pi),
                                              k1=k1, k2=k2, phi=phi * 2 * np.pi)) ** 2 * z_squared


# 定义目标函数 f(z)
def f(z):
    # Calculate m
    if True:
        beta = 0.15
        m = 1 / (1 - beta)
        if z is None:
            z = tf.range(0, 1, 1 / 200)
        result = m * (z - beta)

        return tf.where(result < 0, tf.zeros_like(result), result)

    else:
        beta = 0.6
        alpha = 0.65
        if z is None:
            z = tf.range(0, 1, 1 / 100)
        result = beta * z
        return tf.where(result > alpha, alpha, result)


def plot_prediction_curve(z_squared, function, predicted_function, name):
    plt.plot(z_squared, function, linewidth=2, label="True")
    plt.plot(z_squared, predicted_function, linewidth=2, linestyle='--', label="Prediction")
    # plt.title(name, fontname='Times New Roman', fontsize=18, loc='center')
    # plt.xlabel('input', fontname='Times New Roman', fontsize=18)
    # plt.ylabel('output', fontname='Times New Roman', fontsize=18)
    plt.xticks(fontfamily='Times New Roman', fontsize=14)
    plt.yticks(fontfamily='Times New Roman', fontsize=14)
    plt.legend()
    plt.savefig('%s.png' % name)
    plt.show()
    plt.close()


def custom_loss(y_true, y_pred):
    a, b, c, d, k1, k2 = tf.split(y_pred, num_or_size_splits=6, axis=1)
    # tf.gather 不能大于batch size
    a, b, c, d, k1, k2 = tf.gather(a, [7]), tf.gather(b, [7]), tf.gather(c, [7]), tf.gather(d, [7]), tf.gather(k1, [
        7]), tf.gather(k2, [7])
    z = tf.reshape(y_true, (-1, 1))
    H_pred = H(a, b, c, d, k1, k2, z)
    print(H_pred)
    return tf.reduce_mean(tf.square(f(z) - H_pred)) * 1e7


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


# 生成样本数据

z_train = np.linspace(0, 1, 200)
f_train = f(z_train)

# 构建神经网络模型
model = Sequential([
    Input(shape=(1,)),
    Dense(64, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(64, activation='sigmoid'),
    Dense(6, activation="sigmoid")
])

# 自定义损失函数


# 编译模型
model.compile(optimizer=Adam(learning_rate=0.00004), loss=custom_loss)

# 训练模型
history = model.fit(z_train, z_train, epochs=5000, batch_size=100)
train_loss = history.history['loss']

# save the loss values in csv file
with open('history_OANN_model.pkl', 'wb') as f:
    pickle.dump(history.history, f)
fieldnames = ['Epoch', 'Training Loss', 'Validation Loss']
with open('loss_OANN_model.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for epoch, train_loss_value in zip(range(1, len(train_loss) + 1), train_loss):
        writer.writerow({'Epoch': epoch, 'Training Loss': train_loss_value})

# save the forward model and its weights
model_json = model.to_json()
json_file = open("OANN_model.json", "w")
json_file.write(model_json)
model.save_weights("OANN_model.weights.h5")
json_file.close()

### plot the loss graph
# plt.plot(history.history['val_loss'], linewidth=1)
plt.plot(history.history['loss'], linewidth=2, linestyle='--')
plt.title('The loss of training model', fontname='Times New Roman', fontsize=18, loc='center')
plt.xlabel('epochs', fontname='Times New Roman', fontsize=18)
plt.ylabel('loss', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=14)
plt.yticks(fontfamily='Times New Roman', fontsize=14)
plt.legend(['train'])
plt.savefig("loss.png")
plt.show()

# 预测
predicted_params = model.predict(z_train)
a_pred, b_pred, c_pred, d_pred, k1_pred, k2_pred = np.split(predicted_params, 6, axis=1)

# 输出结果
print("Predicted parameters:")
# print("kr:", np.mean(a_pred))
# print("phi:", np.mean(b_pred))
# print("a:", np.mean(c_pred))
# print("b:", np.mean(d_pred))
# print("k1:", np.mean(k1_pred))
# print("k2:", np.mean(k2_pred))
print("kr: %f,\t phi: %f,\t a: %f,\t b: %f,\t k1: %f,\t k2: %f" % (
a_pred[7], b_pred[7], c_pred[7], d_pred[7], k1_pred[7], k2_pred[7]))
plot_prediction_curve(z_train, function=f_train,
                      predicted_function=H(a_pred[7], b_pred[7], c_pred[7], d_pred[7],
                                           k1_pred[7], k2_pred[7], z_train),
                      name="predicted_function")
print("kr: %f,\n phi: %f,\n a: %f,\n b: %f,\n k1: %f,\n k2: %f" % (
    a_pred[7], b_pred[7], c_pred[7], d_pred[7], k1_pred[7], k2_pred[7]))
# print("rmse = ", rmse(y_true=f_train, y_pred=H(a_pred[7], b_pred[7], c_pred[7], d_pred[7],
#                                                k1_pred[7], k2_pred[7], z_train).numpy()))
print("rmse = %f dB" % (
        10 * np.log10(rmse(y_true=f_train, y_pred=H(a_pred[7], b_pred[7], c_pred[7], d_pred[7],
                                                    k1_pred[7], k2_pred[7], z_train).numpy()))))
