"""
The code for the research presented in the paper titled "A deep learning method for empirical spectral prediction and inverse design of all-optical nonlinear plasmonic ring resonator switches

@authors: Ehsan Adibnia, Majid Ghadrdan and Mohammad Ali Mansouri-Birjandi
Corresponding author: mansouri@ece.usb.ac.ir

This code is corresponding to the Forward Deep Neural Network (DNN) section of the article.
Please cite the paper in any publication using this code.
"""
import keras.optimizers
# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import ReLU
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta
from keras.saving import register_keras_serializable
import tensorflow as tf

# from main import OANN

### Load the data from CSV file (the results of FDTD solver)
result = pd.read_csv("RELU_MZI_training_data.csv", header=None)
result = result.to_numpy()
from numpy import float16

result = result.astype(float16)

# 控制k1 = 0.5: [405000: 432000,]
x = result[405000:432000, 4:]
y = result[405000:432000, 1:4]
# x = result[0:result.shape[0], 4:]
# y = result[0:result.shape[0], 0:4]

# 先标准化再分割数据集
sc = StandardScaler()
x = sc.fit_transform(x.T).T

# Allocation of 70% of the total data to the training data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.60, shuffle=True)
# Allocation of 50% of the remaining data to the validateion data and 50% to the test data (15% validation and 15% test of total)
x_test, x_val, y_test, y_val = train_test_split(x_val, y_val, test_size=0.25, shuffle=True)

### Feature Scaling
# 这段代码对特征进行了标准化处理。使用 StandardScaler 对训练集、验证集和测试集的特征进行了标准化处理，以使其均值为0，方差为1。
# It seems that the original data set was scaled along colunms but not rows, which was weird as the plot in te paper didn't go wrong.
# 看起来源代码是按列标准化的而不是按行，it's weird
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train.T).T
# x_val = sc.transform(x_val.T).T
# x_test = sc.transform(x_test.T).T


@register_keras_serializable()
def custom_rmse(y_true, y_pred):
    # lower_bound = [0., 0., 0., 0.]
    # upper_bound = [1., 1., 1., 20.]
    lower_bound = [0., 0., 0.]
    upper_bound = [1., 1., 20.]
    clipped_pred = tf.clip_by_value(y_pred, lower_bound, upper_bound)
    return tf.sqrt(tf.reduce_mean(3 * tf.square(y_true - clipped_pred), axis=-1))


@register_keras_serializable()
def mean_squared_logarithmic_error_with_panalty(y_true, y_pred):
    from numpy import pi
    # lower_bound = [0., 0., 0., 0.]
    # upper_bound = [1., 1., 1., 20.]
    lower_bound = [0., 0., 0.]
    upper_bound = [1., 1., 20.]
    clipped_pred = tf.clip_by_value(y_pred, lower_bound, upper_bound)
    msle = tf.keras.losses.mean_squared_logarithmic_error(y_true, clipped_pred)
    # panalty = 0
    # if tf.reduce_min(y_pred).numpy() < 0:
    #     panalty = abs(tf.reduce_min(y_pred))
    return tf.reduce_mean(msle, axis=-1)


# To prevent the RAM from being filled up, we delete the result.
del result
### Defining the Layers of the deep neural network (DNN)
# 6 hidden layer and 60 neurons in each layer

Model = Sequential()
Model.add(Dense(60, input_dim=200))
Model.add((ReLU()))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((ReLU()))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((ReLU()))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((ReLU()))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((ReLU()))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((ReLU()))
Model.add(Dense(3))
Model.summary()

# from keras.utils import plot_model
# plot_model(Model, to_file='model.png', show_shapes=True, show_layer_names=True)

# def RELU(beta, z=None):
#     # Calculate m
#     m = 1 / (1 - beta)
#     if not z:
#         z = tf.range(0, 1, 0.05)
#     # Calculate result with element-wise multiplication
#     result = m * (z ** 2 - beta)
#
#     # Apply threshold using tf.where
#     result = tf.where(result < 0, tf.zeros_like(result), result)
#     return result
#
# def f1(oann, y_pred):
#     z = tf.range(0, 1, 0.05)
#     return (tf.abs(oann.transfer_function_MZI(k1=y_pred[:, 0], k2=y_pred[:, 1],
#                                        phi=oann.calculate_phi(y_pred[:, 2], y_pred[:, 3], z))) * z) ** 2 + 1e-7
# def Loss_function_RELU(y_true, y_pred):
#     # from keras import backend as K
#     # z = tf.linspace(0, 1, 20)
#
#     relu = RELU(0.15)
#     oann = OANN(lam=1.55)
#     f = f1(oann, y_pred)
#     # f = (tf.abs(oann.transfer_function_MZI(k1=y_pred[:, 0], k2=y_pred[:, 1],
#     #                                        phi=oann.calculate_phi(y_pred[:, 2], y_pred[:, 3], z))) * z) ** 2 + 1e-7
#     squared_difference = (f - relu) ** 2
#     loss = tf.reduce_mean(squared_difference, axis=-1)
#     print("the loss is", loss)
#     return loss


# useing EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=5000)

### Configuring the settings of the training procedure 
# Mean Squared Logarithmic Error (MSLE) function has been used for loss estimation
# AdaDelta Optimizer has been used and learning rate of 0.1 has been set
# 'mean_squared_logarithmic_error'
# "root_mean_squared_error"
# "root_mean_squared_error"
# Model.compile(loss=keras.losses.MeanSquaredError(),
#               optimizer=keras.optimizers.SGD(learning_rate=0.001))
Model.compile(loss=keras.losses.MeanSquaredError(),
              optimizer=keras.optimizers.Adam(learning_rate=0.00005))
# Model.compile(loss=keras.losses.MeanSquaredError(),
#               optimizer=keras.optimizers.Adadelta(learning_rate=0.001))
# learning_rate=0.1
# Model.compile(loss=custom_MSE,
#               optimizer=Adadelta())
### Training Model 
# batch size of 80 was set and 5000 epochs were performed
# TODO: seems 5000 epochs are not enough
history = Model.fit(x_train, y_train, epochs=5000,
                    batch_size=200, validation_data=(x_val, y_val), callbacks=[es])

### plot the loss graph
plt.plot(history.history['val_loss'], linewidth=1)
plt.plot(history.history['loss'], linewidth=2, linestyle='--')
plt.title('The loss of training model', fontname='Times New Roman', fontsize=18, loc='center')
plt.xlabel('epochs', fontname='Times New Roman', fontsize=18)
plt.ylabel('loss', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=14)
plt.yticks(fontfamily='Times New Roman', fontsize=14)
plt.legend(['train', 'Validation', 'test'])
plt.savefig("loss.png")
plt.show()

### loss value of train and validation data
train_loss = history.history['loss']
val_loss = history.history['val_loss']

### Testing Model 
predictions = Model.predict(x_test)
Loss = Model.evaluate(x_test, y_test)
print(Loss)

# save the loss values in csv file
with open('history_OANN_model.pkl', 'wb') as f:
    pickle.dump(history.history, f)
fieldnames = ['Epoch', 'Training Loss', 'Validation Loss']
with open('loss_OANN_model.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for epoch, train_loss_value, val_loss_value in zip(range(1, len(train_loss) + 1), train_loss, val_loss):
        writer.writerow({'Epoch': epoch, 'Training Loss': train_loss_value, 'Validation Loss': val_loss_value})

# save the forward model and its weights
model_json = Model.to_json()
json_file = open("OANN_model.json", "w")
json_file.write(model_json)
Model.save_weights("OANN_model.weights.h5")
json_file.close()
