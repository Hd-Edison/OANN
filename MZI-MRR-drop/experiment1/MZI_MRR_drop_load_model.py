from keras.models import model_from_json
from keras.saving import register_keras_serializable
import numpy as np
import tensorflow as tf
from main import OANN
import matplotlib.pyplot as plt


def H(kr, phi, a, b, k1, k2, z_squared):
    oann = OANN(lam=1.55)
    return abs(transfer_function_MZI_ORR(HR=oann.transfer_function_MRR(k=kr, gamma=0.8, neff=2.384556,
                                                                       phi=oann.calculate_phi(a * 10 * 2 * np.pi,
                                                                                              b * 10,
                                                                                              z_squared=z_squared),
                                                                       L=5e-6 * 2 * np.pi),
                                         k1=k1, k2=k2, phi=phi * 2 * np.pi)) ** 2 * z_squared


def H_new(kr, phi, a, b, k1, k2, z_squared):
    oann = OANN(lam=1.55)
    H = np.empty(200)
    for i in range(200):
        m = abs(transfer_function_MZI_ORR(HR=oann.transfer_function_MRR(k=kr[i, 0], gamma=0.8, neff=2.384556,
                                                                        phi=oann.calculate_phi(a[i] * 2 * np.pi,
                                                                                               b[i],
                                                                                               z_squared=z_squared[
                                                                                                   i]),
                                                                        L=5e-6 * 2 * np.pi),
                                          k1=k1[i], k2=k2[i], phi=phi[i] * 2 * np.pi)) ** 2 * z_squared[i]
        H[i] = m.numpy()
    return H


# 定义目标函数 f(z)
def f(z):
    # Calculate m
    if False:
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


@register_keras_serializable()
def custom_loss(y_true, y_pred):
    a, b, c, d, k1, k2 = tf.split(y_pred, num_or_size_splits=6, axis=1)
    z = tf.reshape(y_true, (-1, 1))
    H_pred = H(a, b, c, d, k1, k2, z)
    return tf.reduce_mean(tf.square(f(z) - H_pred)) * 100000


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


def plot_variables(z_squared, variables, name="variables"):
    for i in range(len(variables)):
        plt.plot(z_squared, variables[i])
    # plt.title(name, fontname='Times New Roman', fontsize=18, loc='center')
    # plt.xlabel('input', fontname='Times New Roman', fontsize=18)
    # plt.ylabel('output', fontname='Times New Roman', fontsize=18)
    plt.xticks(fontfamily='Times New Roman', fontsize=14)
    plt.yticks(fontfamily='Times New Roman', fontsize=14)
    plt.legend(["kr", "phi", "a", "b", "k1", "k2"])
    plt.savefig('%s.png' % name)
    plt.show()
    plt.close()


def load_file(json_file, weights):
    # Load Model architecture from JSON file
    json_file = open(json_file, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)

    return loaded_model


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


if __name__ == '__main__':
    ### load Model
    model = load_file("model2.json", "model2.weights.h5")
    z_train = np.linspace(0, 1, 200)

    # 预测
    predicted_params = model.predict(z_train)
    a_pred, b_pred, c_pred, d_pred, k1_pred, k2_pred = np.split(predicted_params, 6, axis=1)

    # 输出结果
    print("Predicted parameters:")
    print("kr:", np.mean(a_pred))
    print("phi:", np.mean(b_pred))
    print("a:", np.mean(c_pred))
    print("b:", np.mean(d_pred))
    print("k1:", np.mean(k1_pred))
    print("k2:", np.mean(k2_pred))
    print(a_pred[7], b_pred[7], c_pred[7], d_pred[7], k1_pred[7], k2_pred[7])

    # print(custom_loss(f(z_train), [[a_pred, b_pred, c_pred, d_pred, k1_pred, k2_pred]]))
    print(model.evaluate(f(z_train), z_train))

    # plot_prediction_curve(z_train, function=f(z_train),
    #                       predicted_function=H(np.mean(a_pred), np.mean(b_pred), np.mean(c_pred), np.mean(d_pred),
    #                                            np.mean(k1_pred), np.mean(k2_pred), z_train),
    #                       name="relu")
    plot_prediction_curve(z_train, function=f(z_train),
                          predicted_function=H(a_pred[7], b_pred[7], c_pred[7], d_pred[7],
                                               k1_pred[7], k2_pred[7], z_train),
                          name="predicted_function")
    # plot_prediction_curve(z_train, function=f(z_train),
    #                       predicted_function=H_new(a_pred, b_pred, c_pred, d_pred,
    #                                            k1_pred, k2_pred, z_train),
    #                       name="relu")
    plot_variables(z_train, variables=[a_pred, b_pred, c_pred, d_pred,
                                       k1_pred, k2_pred])
    print("kr: %f,\n phi: %f,\n a: %f,\n b: %f,\n k1: %f,\n k2: %f" % (
        a_pred[7], b_pred[7], c_pred[7], d_pred[7], k1_pred[7], k2_pred[7]))
    # print("rmse = ", rmse(y_true=f(z_train), y_pred=H(a_pred[7], b_pred[7], c_pred[7], d_pred[7],
    #                                                   k1_pred[7], k2_pred[7], z_train)).numpy())
    print("rmse = %f dB" % (
                10 * np.log10(rmse(y_true=f(z_train), y_pred=H(a_pred[7], b_pred[7], c_pred[7], d_pred[7],
                                                               k1_pred[7], k2_pred[7], z_train).numpy()))))
