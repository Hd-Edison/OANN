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

oann = OANN(lam=1550e-9)


class CustomLoss(Loss):
    def __init__(self, true_function):
        self.true_function = true_function
        super(CustomLoss, self).__init__()

    def call(self, y_true, y_pred):
        p1, p2, p3, p4, p5, p6, p7, p8 = tf.split(y_pred, num_or_size_splits=8, axis=1)
        # tf.gather 不能大于batch size
        p1, p2, p3, p4, p5, p6, p7, p8 = tf.gather(p1, [0]), tf.gather(p2, [0]), tf.gather(p3, [0]), \
            tf.gather(p4, [0]), tf.gather(p5, [0]), tf.gather(p6, [0]), tf.gather(p7, [0]), tf.gather(p8, [0])
        temp = self.true_function.function()
        H_pred = H(p1, p2, p3, p4, p5, p6, p7, p8)
        print(H_pred)
        return tf.reduce_mean(tf.square(temp - H_pred))


def H(a_k1, b_k1, k2, kr1, kr2, phi, a, b, z_squared=None):
    if z_squared is None:
        z_squared = tf.range(0, 1, 1 / 200)
    k1 = oann.calculate_k(a_k1 * 2 * np.pi, b_k1 * 10, z_squared=z_squared)
    phir = oann.calculate_phi(a=a * 2 * np.pi, b=b * 10, z_squared=z_squared)
    HR_MRR_drop = oann.transfer_function_MRR_drop(kr1=kr1, kr2=kr2, phi_r=phir, neff=2.389, r=100e-6,
                                                  phi=phi * 2 * np.pi, l=500e-6)
    return abs(oann.transfer_function_MZI(phi=0.0, k1=k1, k2=k2, HR2=HR_MRR_drop)) ** 2 * z_squared


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


def train(true_function, name, z_squared=None):
    if not z_squared:
        z_squared = tf.range(0, 1, 1 / 200)
    input = output = tf.fill((1, 1), 1)

    model = Sequential([
        Input(shape=(1,)),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(8, activation="sigmoid")
    ])
    custom_loss = CustomLoss(true_function)
    model.compile(optimizer=Adam(learning_rate=0.00004), loss=custom_loss)

    history = model.fit(input, output, epochs=5000, batch_size=200)
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
    json_file = open("./%s/model.json" % name, "w")
    json_file.write(model_json)
    model.save_weights("./%s/model.weights.h5" % name)
    json_file.close()

    ### plot the loss graph
    plot_loss(loss=history.history['loss'], name=name)

    # 预测
    predicted_params = model.predict(z_squared)
    a_k1, b_k1, k2, kr1, kr2, phi, a, b = np.split(predicted_params, 8, axis=1)

    # 输出结果
    print("Predicted parameters:")
    f = true_function.function()
    plot_prediction_curve(z_squared, function=f,
                          predicted_function=H(a_k1[0], b_k1[0], k2[0], kr1[0], kr2[0],
                                               phi[0], a[0], b[0], z_squared), name=name)

    rmse_dB = 10 * np.log10(rmse(y_true=f, y_pred=H(a_k1[0], b_k1[0], k2[0], kr1[0], kr2[0],
                                                    phi[0], a[0], b[0], z_squared).numpy()))
    with open('./%s/output.txt' % name, 'w') as f:
        f.write("a_k1: %f,\n b_k1: %f,\n k2: %f,\n kr1: %f,\n kr2: %f,\n phi: %f,\n a: %f,\n b: %f,\n" % (
            a_k1[0] * 2 * np.pi, b_k1[0] * 10, k2[0], kr1[0], kr2[0], phi[0] * 2 * np.pi, a[0] * 2 * np.pi, b[0] * 10))
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
    os.system("save_result_to_csv.py")
