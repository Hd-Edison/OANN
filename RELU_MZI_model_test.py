"""
The code for the research presented in the paper titled "A deep learning method for empirical spectral prediction and inverse design of all-optical nonlinear plasmonic ring resonator switches

@authors: Ehsan Adibnia, Majid Ghadrdan and Mohammad Ali Mansouri-Birjandi
Corresponding author: mansouri@ece.usb.ac.ir

This code is corresponding to the Forward Deep Neural Network (DNN) section of the article.
Please cite the paper in any publication using this code.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta

### Load the data from CSV file (the results of FDTD solver)
result = pd.read_csv("RELU_MZI_training_data.csv", header=None, skiprows=1)
result = result.to_numpy()

x = result[0:result.shape[0], 4:]
y = result[0:result.shape[0], 0:4]

# Allocation of 70% of the total data to the training data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.30, shuffle='true')
# Allocation of 50% of the remaining data to the validateion data and 50% to the test data (15% validation and 15% test of total)
x_test, x_val, y_test, y_val = train_test_split(x_val, y_val, test_size=0.50, shuffle='true')

### Feature Scaling
# 这段代码对特征进行了标准化处理。使用 StandardScaler 对训练集、验证集和测试集的特征进行了标准化处理，以使其均值为0，方差为1。
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_val = sc.transform(x_val)
x_test = sc.transform(x_test)

# To prevent the RAM from being filled up, we delete the result.
del result
### Defining the Layers of the deep neural network (DNN)
# 6 hidden layer and 60 neurons in each layer
# Slope of 0.2 has been set for the Leaky ReLU
# Input consist of 5 geometric parameter of all-optical plasmonic switch (AOPS) and the wavelength
# Output is the transmission value at the wavelength (800 point for through port and 800 point for drop port)
Model = Sequential()
Model.add(Dense(60, input_dim=20))
Model.add((LeakyReLU(alpha=0.2)))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((LeakyReLU(alpha=0.2)))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((LeakyReLU(alpha=0.2)))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((LeakyReLU(alpha=0.2)))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((LeakyReLU(alpha=0.2)))
Model.add(Dropout(0.1))
Model.add(Dense(60))
Model.add((LeakyReLU(alpha=0.2)))
Model.add(Dense(4))
Model.summary()


def RELU(beta, z):
    m = 1 / (1 - beta)
    result = m * (z - beta)
    result[result < 0] = 0
    return result


def Loss_function_RELU(y_true, y_pred):
    from .main import OANN
    z = np.arange(0, 1.05, 0.05)
    relu = RELU(0.15, z)
    oann = OANN(lam=1.55)
    f = oann.transfer_function_MZI(k1=y_pred[0], k2=y_pred[1], phi=oann.calculate_phi(y_pred[2], y_pred[3], z))
    squared_difference = (f - relu) ** 2
    return np.mean(squared_difference, axis=-1)


# useing EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=2, )

### Configuring the settings of the training procedure 
# Mean Squared Logarithmic Error (MSLE) function has been used for loss estimation
# AdaDelta Optimizer has been used and learning rate of 0.1 has been set 
# Model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=Adadelta(learning_rate=0.1))
Model.compile(loss=Loss_function_RELU,
              optimizer=Adadelta(learning_rate=0.1))
### Training Model 
# batch size of 80 was set and 5000 epochs were performed
history = Model.fit(x_train, y_train, epochs=5000,
                    batch_size=80, validation_data=(x_val, y_val), callbacks=[es])

### plot the loss graph
plt.plot(history.history['val_loss'], linewidth=1)
plt.plot(history.history['loss'], linewidth=2, linestyle='--')
plt.title('The loss of training model', fontname='Times New Roman', fontsize=18, loc='center')
plt.xlabel('epochs', fontname='Times New Roman', fontsize=18)
plt.ylabel('loss', fontname='Times New Roman', fontsize=18)
plt.xticks(fontfamily='Times New Roman', fontsize=14)
plt.yticks(fontfamily='Times New Roman', fontsize=14)
plt.legend(['train', 'Validation', 'test'])
plt.show()

### loss value of train and validation data
train_loss = history.history['loss']
val_loss = history.history['val_loss']

### Testing Model 
predictions = Model.predict(x_test)
Loss = Model.evaluate(x_test, y_test)
print(Loss)

# save the loss values in csv file
with open('history_Forward_model.pkl', 'wb') as f:
    pickle.dump(history.history, f)
fieldnames = ['Epoch', 'Training Loss', 'Validation Loss']
with open('loss_Forward_model.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for epoch, train_loss_value, val_loss_value in zip(range(1, len(train_loss) + 1), train_loss, val_loss):
        writer.writerow({'Epoch': epoch, 'Training Loss': train_loss_value, 'Validation Loss': val_loss_value})

# save the forward model and its weights
model_json = Model.to_json()
json_file = open("T-shaped switch_Nozhat_model.json", "w")
json_file.write(model_json)
Model.save_weights("T-shaped switch_Nozhat_model_weights.h5")
json_file.close()

### output
# Model: "sequential"
# ┌─────────────────────────────────┬────────────────────────┬───────────────┐
# │ Layer (type)                    │ Output Shape           │       Param # │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense (Dense)                   │ (None, 60)             │           420 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ leaky_re_lu (LeakyReLU)         │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dropout (Dropout)               │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_1 (Dense)                 │ (None, 60)             │         3,660 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ leaky_re_lu_1 (LeakyReLU)       │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dropout_1 (Dropout)             │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_2 (Dense)                 │ (None, 60)             │         3,660 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ leaky_re_lu_2 (LeakyReLU)       │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dropout_2 (Dropout)             │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_3 (Dense)                 │ (None, 60)             │         3,660 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ leaky_re_lu_3 (LeakyReLU)       │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dropout_3 (Dropout)             │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_4 (Dense)                 │ (None, 60)             │         3,660 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ leaky_re_lu_4 (LeakyReLU)       │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dropout_4 (Dropout)             │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_5 (Dense)                 │ (None, 60)             │         3,660 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ leaky_re_lu_5 (LeakyReLU)       │ (None, 60)             │             0 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_6 (Dense)                 │ (None, 2)              │           122 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 18,842 (73.60 KB)
#  Trainable params: 18,842 (73.60 KB)
#  Non-trainable params: 0 (0.00 B)
# Epoch 1/5000
# 150863/150863 ━━━━━━━━━━━━━━━━━━━━ 126s 827us/step - loss: 0.0051 - val_loss: 0.0023
# Epoch 2/5000
# 150863/150863 ━━━━━━━━━━━━━━━━━━━━ 124s 822us/step - loss: 0.0023 - val_loss: 0.0015
# Epoch 3/5000
# 150863/150863 ━━━━━━━━━━━━━━━━━━━━ 126s 832us/step - loss: 0.0018 - val_loss: 0.0013
# Epoch 4/5000
# 150863/150863 ━━━━━━━━━━━━━━━━━━━━ 125s 826us/step - loss: 0.0016 - val_loss: 0.0013
# Epoch 5/5000
# 150863/150863 ━━━━━━━━━━━━━━━━━━━━ 127s 838us/step - loss: 0.0015 - val_loss: 0.0014
# Epoch 6/5000
# 150863/150863 ━━━━━━━━━━━━━━━━━━━━ 128s 849us/step - loss: 0.0014 - val_loss: 0.0013
# Epoch 6: early stopping
# 80820/80820 ━━━━━━━━━━━━━━━━━━━━ 30s 365us/step
# 80820/80820 ━━━━━━━━━━━━━━━━━━━━ 28s 346us/step - loss: 0.0013
# 0.0013152621686458588
