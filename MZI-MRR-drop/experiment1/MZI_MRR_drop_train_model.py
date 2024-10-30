import numpy as np
import tensorflow as tf
from main import OANN
from activation_functions import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
import csv
import os
from tensorflow.keras.losses import Loss


class CustomLoss(Loss):
    def __init__(self, true_function):
        self.true_function = true_function
        super(CustomLoss, self).__init__()

    def call(self, y_true, y_pred):
        k1, k2, kr1, kr2, phi, a, b, phir2 = tf.split(y_pred, num_or_size_splits=8, axis=1)
        # tf.gather 不能大于batch size
        k1, k2, kr1, kr2, phi, a, b, phir2 = tf.gather(k1, [0]), tf.gather(k2, [0]), tf.gather(kr1, [0]), \
            tf.gather(kr2, [0]), tf.gather(phi, [0]), tf.gather(a, [0]), tf.gather(b, [0]), tf.gather(phir2, [0])
        z = tf.reshape(y_true, (-1, 1))
        temp = self.true_function.function(z)
        H_pred = H(k1, k2, kr1, kr2, phi, a, b, phir2, z)
        print(H_pred)
        return tf.reduce_mean(tf.square(temp - H_pred))


def H(k1, k2, kr1, kr2, phi, a, b, phir2, z_squared):
    oann = OANN(lam=1.55)
    return abs(transfer_function_MZI_ORR(HR=oann.transfer_function_MRR(k=kr1, gamma=0.86, neff=2.389,
                                                                       phi=oann.calculate_phi(a * 2 * np.pi,
                                                                                              b * 10,
                                                                                              z_squared=z_squared),
                                                                       L=100e-6 * 2 * np.pi),
                                         HR2=oann.transfer_function_MRR(k=kr2, gamma=0.86, neff=2.389,
                                                                        phi=phir2 * 2 * np.pi, L=100e-6 * 2 * np.pi),
                                         k1=k1, k2=k2, phi=phi * 2 * np.pi)) ** 2 * z_squared


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def plot_loss(loss, name=None):
    plt.plot(loss, linewidth=2, linestyle='--')
    plt.title('The loss of training Model', fontname='Times New Roman', fontsize=18, loc='center')
    plt.xlabel('epochs', fontname='Times New Roman', fontsize=18)
    plt.ylabel('loss', fontname='Times New Roman', fontsize=18)
    plt.xticks(fontfamily='Times New Roman', fontsize=14)
    plt.yticks(fontfamily='Times New Roman', fontsize=14)
    plt.legend(['train'])
    plt.savefig("./%s/loss.png" % name)
    # plt.show()
    plt.close()


def plot_prediction_curve(z_squared, function, predicted_function, name):
    plt.plot(z_squared, function, linewidth=2, label="True")
    plt.plot(z_squared, predicted_function, linewidth=2, linestyle='--', label="Prediction")
    # plt.title(name, fontname='Times New Roman', fontsize=18, loc='center')
    plt.xlabel('z^2', fontname='Times New Roman', fontsize=18)
    plt.ylabel('f(z^2)', fontname='Times New Roman', fontsize=18)
    plt.xticks(fontfamily='Times New Roman', fontsize=14)
    plt.yticks(fontfamily='Times New Roman', fontsize=14)
    plt.legend()
    plt.savefig('./%s/prediction.png' % name)
    # plt.show()
    plt.close()


def train(true_function, name, z_train=None):
    if not z_train:
        z_train = np.linspace(0, 1, 200)

    model = Sequential([
        Input(shape=(1,)),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(8, activation="sigmoid")
    ])
    custom_loss = CustomLoss(true_function)
    model.compile(optimizer=Adam(learning_rate=0.00004), loss=custom_loss)

    history = model.fit(z_train, z_train, epochs=5000, batch_size=200)
    train_loss = history.history['loss']

    current_directory = os.path.dirname(__file__)
    target_folder = os.path.join(current_directory, name)
    os.makedirs(target_folder, exist_ok=True)

    with open('./%s/history_OANN_model.pkl' % name, 'wb') as f:
        pickle.dump(history.history, f)
    fieldnames = ['Epoch', 'Training Loss', 'Validation Loss']
    with open('./%s/loss_OANN_model.csv' % name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, train_loss_value in zip(range(1, len(train_loss) + 1), train_loss):
            writer.writerow({'Epoch': epoch, 'Training Loss': train_loss_value})

    # save the forward Model and its weights
    model_json = model.to_json()
    json_file = open("./%s/model2.json" % name, "w")
    json_file.write(model_json)
    model.save_weights("./%s/model2.weights.h5" % name)
    json_file.close()

    ### plot the loss graph
    plot_loss(loss=history.history['loss'], name=name)

    # 预测
    predicted_params = model.predict(z_train)
    k1, k2, kr1, kr2, phi, a, b, phir2 = np.split(predicted_params, 8, axis=1)

    # 输出结果
    print("Predicted parameters:")
    f = true_function.function(z_train)
    plot_prediction_curve(z_train, function=f,
                          predicted_function=H(k1[0], k2[0], kr1[0], kr2[0],
                                               phi[0], a[0], b[0], phir2[0], z_train), name=name)

    rmse_dB = 10 * np.log10(rmse(y_true=f, y_pred=H(k1[0], k2[0], kr1[0], kr2[0],
                                                    phi[0], a[0], b[0], phir2[0], z_train).numpy()))
    with open('./%s/output.txt' % name, 'w') as f:
        f.write("k1: %f,\n k2: %f,\n kr1: %f,\n kr2: %f,\n phi: %f,\n a: %f,\n b: %f,\n phir2: %f,\n" % (
            k1[0], k2[0], kr1[0], kr2[0], phi[0] * 2 * np.pi, a[0] * 2 * np.pi, b[0] * 10, phir2[0] * 2 * np.pi))
        f.write("rmse = " + str(rmse_dB) + " dB")


if __name__ == '__main__':
    names = ["ReLU", "Clipped_ReLU", "ELU", "GeLU", "Parametric_ReLU", "SiLU", "Gaussian", "Quadratic", "Sigmoid",
             "Sine", "Softplus", "Tanh", "Softsign", "Exponential"]
    for name in names:
        if name == names[0]:
            for beta in (0.15, 0.25, 0.35, 0.55):
                f = ReLU(beta=beta)
                train(f, name=name + str(beta))
        if name == names[1]:
            for alpha in (0.6, 0.8, 1.0):
                for beta in (0.55, 0.65, 0.75, 0.85):
                    f = Cliped_ReLU(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[2]:
            for alpha in (0.1, 0.2, 0.3):
                for beta in (0.15, 0.25, 0.35, 0.55):
                    f = ELu(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[3]:
            for alpha in (6., 8., 10.):
                for beta in (0.25, 0.35, 0.45, 0.55):
                    f = GeLu(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[4]:
            for alpha in (0.1, 0.15, 0.2):
                for beta in (0.15, 0.25, 0.35, 0.55):
                    f = Parametric_ReLu(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[5]:
            for alpha in (10., 15., 20.):
                for beta in (0.25, 0.35, 0.45, 0.55):
                    f = SiLU(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[6]:
            for alpha in (0.1, 0.2, 0.3):
                for beta in (0.5, 0.6, 0.7):
                    f = Gaussian(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[7]:
            f = Quadratic(alpha=alpha, beta=beta)
            train(f, name=name)
        if name == names[8]:
            for alpha in (10., 20.):
                for beta in (0.5, 0.7):
                    f = Sigmoid(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[9]:
            for alpha in (1., np.pi, 2 * np.pi):
                beta = (- np.pi / 2)
                f = Sine(alpha=alpha, beta=beta)
                train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[10]:
            for alpha in (10., 15., 20.):
                for beta in (0.2, 0.3, 0.4):
                    f = Softplus(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[11]:
            for alpha in (5., 10., 15.):
                for beta in (0.5, 0.7):
                    f = Tanh(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[12]:
            for alpha in (10., 20.):
                for beta in (0.5, 0.7):
                    f = Softsign(alpha=alpha, beta=beta)
                    train(f, name=name + "_alpha=" + str(alpha) + "_beta=" + str(beta))
        if name == names[13]:
            for beta in (3., 6., 9.):
                f = Exponential(beta=beta)
                train(f, name=name + "_beta=" + str(beta))