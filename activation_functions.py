import tensorflow as tf


class Activation_Functions():

    def __init__(self):
        self.Output()

    def Output(self):
        return ValueError("this should be overridden")


class ReLU(Activation_Functions):
    def __init__(self, beta, z):
        self.beta = beta
        self.z = z
        self.Output()

    def Output(self):
        self.beta = 0.15
        m = 1 / (1 - self.beta)
        if self.z is None:
            self.z = tf.range(0, 1, 1 / 200)
        result = m * (self.z - self.beta)

        return tf.where(result < 0, tf.zeros_like(result), result)


class Cliped_ReLU(Activation_Functions):
    def __init__(self, alpha, beta, z):
        self.alpha = alpha
        self.beta = beta
        self.z = z
        self.Output()

    def Output(self):
        if self.z is None:
            self.z = tf.range(0, 1, 1 / 200)
        result = self.beta * self.z
        return tf.where(result > self.alpha, self.alpha, result)


class ELu(Activation_Functions):
    def __init__(self, alpha, beta, z, m, c):
        self.alpha = alpha
        self.beta = beta
        self.z = z
        self.m = m
        self.c = c
        self.Output()

    def Output(self):
        if self.z is None:
            self.z = tf.range(0, 1, 1 / 200)

        return tf.where(self.z > self.beta, self.alpha * (tf.exp(self.m * (self.z - self.beta) - 1) + self.c),
                        self.m * (self.z - self.beta) + self.c)


class GeLu(Activation_Functions):
    def __init__(self, alpha, beta, z, c, scale: float = 1.):
        self.alpha = alpha
        self.beta = beta
        self.z = z
        self.scale = scale
        self.c = c
        self.Output()

    def Output(self):
        if self.z is None:
            self.z = tf.range(0, 1, 1 / 200)

        return 0.5 * self.alpha * (self.z - self.beta) * (
                    1 + tf.math.erf(self.alpha * (self.z - self.beta) / tf.sqrt(2)) + self.c) / self.scale


class Parametric_ReLu(Activation_Functions):
    def __init__(self, alpha, beta, z, c, scale: float = 1.):
        self.alpha = alpha
        self.beta = beta
        self.z = z
        self.scale = scale
        self.c = c
        self.Output()

    def Output(self):
        if self.z is None:
            self.z = tf.range(0, 1, 1 / 200)

        return (tf.maximum(self.alpha * (self.z - self.beta), (self.z - self.beta)) + self.c) /self.scale


class SiLU(Activation_Functions):
    def __init__(self, alpha, beta, z, c, scale: float = 1.):
        self.alpha = alpha
        self.beta = beta
        self.z = z
        self.scale = scale
        self.c = c
        self.Output()

    def Output(self):
        if self.z is None:
            self.z = tf.range(0, 1, 1 / 200)

        return (tf.maximum(self.alpha * (self.z - self.beta), (self.z - self.beta)) + self.c) /self.scale