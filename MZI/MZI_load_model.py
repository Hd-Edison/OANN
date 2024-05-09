from keras.models import model_from_json
from keras.saving import register_keras_serializable
import numpy as np
import tensorflow as tf
from main import OANN
import matplotlib.pyplot as plt


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
    lower_bound = [0., 0., 0., 0.]
    upper_bound = [1.0, 1., 2 * pi, 20.]
    clipped_pred = tf.clip_by_value(y_pred, lower_bound, upper_bound)
    msle = tf.keras.losses.mean_squared_logarithmic_error(y_true, clipped_pred)
    # panalty = 0
    # if tf.reduce_min(y_pred).numpy() < 0:
    #     panalty = abs(tf.reduce_min(y_pred))
    return tf.reduce_mean(msle, axis=-1)


def RELU(beta, z=None):
    # Calculate m
    m = 1 / (1 - beta)
    if z is None:
        z = tf.range(0, 1, 1 / 200)
    result = m * (z - beta)

    return tf.where(result < 0, tf.zeros_like(result), result)
def Cliped_RELU(alpha, beta, z=None):

    if z is None:
        z = tf.range(0, 1, 1 / 200)
    result = beta * z
    return tf.where(result > alpha, alpha, result)


def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the true and predicted values.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def plot_prediction_curve(z_squared, function, predicted_function, name):
    plt.plot(z_squared, function[0, :], linewidth=2, label="True")
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


def load_file(json_file, weights):
    # Load model architecture from JSON file
    json_file = open(json_file, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)

    return loaded_model


if __name__ == '__main__':
    ### load model
    loaded_model = load_file("OANN_model.json", "OANN_model.weights.h5")
    z_squared = tf.range(0, 1, 1 / 200)

    ### predict
    use_calculation = True
    ## use calculation
    if use_calculation:
        true_function = RELU(0, z_squared).numpy()  # Your new data as a NumPy array
        # true_function = Cliped_RELU(0.6, 0.85,  z_squared).numpy()
    else:
        ## use data from data set
        import pandas as pd
        true_function = pd.read_csv("RELU_MZI_training_data.csv", header=None, skiprows=431000, nrows=1).to_numpy()[0, 4:]
    true_function = np.expand_dims(true_function, axis=0)
    scaled_true_function = true_function

    ## StandardScalar, essential if used before training
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    scaled_true_function = sc.fit_transform(true_function.T).T
    data_restored = sc.inverse_transform(scaled_true_function.T).T

    predictions = loaded_model.predict(scaled_true_function)[0, :]
    print(predictions)

    ### predicted function
    oann = OANN(1.55)
    H = abs(oann.transfer_function_MZI(k1=0.5, k2=predictions[0],
                                       phi=oann.calculate_phi(predictions[1] * 2 * np.pi, predictions[2],
                                                              z_squared=z_squared)))
    # H = abs(oann.transfer_function_MZI(k1=predictions[0], k2=predictions[1],
    #                                    phi=oann.calculate_phi(predictions[2] * 2 * np.pi, predictions[3],
    #                                                           z_squared=z_squared)))
    predicted_function = H ** 2 * z_squared

    ### draw comparation curve
    plot_prediction_curve(z_squared, true_function, predicted_function, "RELU")
    print(predictions)
    print("rmse = ", rmse(y_true=true_function, y_pred=predicted_function).numpy())
    print("rmse in dB = %f dB" % (10 * np.log10(rmse(y_true=true_function, y_pred=predicted_function).numpy())))
