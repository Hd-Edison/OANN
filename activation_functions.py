import tensorflow as tf


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


class Activation_Functions():
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.c = c
        self.scale = scale

    # def __new__(cls):
    #     pass
    #     raise ValueError("this should be overridden")

    def name(self):
        return self.__class__.__name__

    def function(self, z=None):
        return ValueError("this should be overridden")

class ReLU(Activation_Functions):

    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)

    def function(self, z=None):
        m = 1 / (1 - self.beta)
        if z is None:
            z = tf.range(0, 1, 1 / 200)
        result = m * (z - self.beta)

        return tf.where(result < 0, tf.zeros_like(result), result)
    # def __new__(cls, beta, z=None):
    #     m = 1 / (1 - beta)
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #     result = m * (z - beta)
    #
    #     return tf.where(result < 0, tf.zeros_like(result), result)


class Cliped_ReLU(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)
        result = self.beta * z
        return tf.where(result > self.alpha, self.alpha, result)

    # def __new__(cls, alpha, beta, z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #     result = beta * z
    #     return tf.where(result > alpha, alpha, result)


class ELu(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return tf.where(z > self.beta, self.alpha * (tf.exp(self.m * (z - self.beta) - 1) + self.c),
                        self.m * (z - self.beta) + self.c)
    # def __new__(cls, alpha, beta, m, c, z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return tf.where(z > beta, alpha * (tf.exp(m * (z - beta) - 1) + c),
    #                     m * (z - beta) + c)


class GeLu(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return 0.5 * self.alpha * (z - self.beta) * (
                1 + tf.math.erf(self.alpha * (z - self.beta) / tf.sqrt(2)) + self.c) / self.scale
    # def __new__(cls, alpha, beta, c, scale: float = 1., z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return 0.5 * alpha * (z - beta) * (
    #             1 + tf.math.erf(alpha * (z - beta) / tf.sqrt(2)) + c) / scale


class Parametric_ReLu(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return (tf.maximum(self.alpha * (z - self.beta), (z - self.beta)) + self.c) / self.scale
    # def __new__(cls, alpha, beta, c, scale: float = 1., z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return (tf.maximum(alpha * (z - beta), (z - beta)) + c) / scale


class SiLU(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return (self.alpha * (z - self.beta) / (
                1 + tf.exp(-self.alpha * (z - self.beta))) + self.c) / self.scale
    # def __new__(cls, alpha, beta, c, scale: float = 1., z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return (alpha * (z - beta) / (
    #             1 + tf.exp(-alpha * (z - beta))) + c) / scale


class Gaussian(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return tf.exp(-tf.square(z - self.beta) / (2 * tf.square(self.alpha))) / self.scale
    # def __new__(cls, alpha, beta, scale: float = 1., z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return tf.exp(-tf.square(z - beta) / (2 * tf.square(alpha))) / scale


class Quadratic(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return tf.square(z)
    # def __new__(cls, z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return tf.square(z)


class Sigmoid(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return self.scale / (1 + tf.exp(-self.alpha * (z - self.beta)))
    # def __new__(cls, alpha, beta, scale: float = 1., z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return scale / (1 + tf.exp(-alpha * (z - beta)))


class Sine(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return (tf.sin(self.alpha * z + self.beta) + z) / 2
    # def __new__(cls, alpha, beta, z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return (tf.sin(alpha * z + beta) + z) / 2


class Softplus(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)
        ### 文章中没说清楚是ln还是log10
        return tf.math.log(1 + tf.exp(self.alpha * (z - self.beta))) / self.scale
    # def __new__(cls, alpha, beta, scale: float = 1., z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #     ### 文章中没说清楚是ln还是log10
    #     return tf.math.log(1 + tf.exp(alpha * (z - beta))) / scale


class Tanh(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return (tf.math.tanh(self.alpha * (z - self.beta)) + self.c) * self.scale
    # def __new__(cls, alpha, beta, c, scale: float = 1., z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return (tf.math.tanh(alpha * (z - beta)) + c) * scale


class Softsign(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return (self.alpha * (z - self.beta) / (1 + tf.abs(self.alpha * (z - self.beta))) + self.c) / self.scale
    # def __new__(cls, alpha, beta, c, scale: float = 1., z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return (alpha * (z - beta) / (1 + tf.abs(alpha * (z - beta))) + c) / scale


class Exponential(Activation_Functions):
    def __init__(self, alpha: float = 1., beta: float = 1., m: float = 1., c: float = 1., scale: float = 1.):
        super().__init__(alpha, beta, m, c, scale)
    def function(self, z=None):
        if z is None:
            z = tf.range(0, 1, 1 / 200)

        return tf.exp(self.beta(z - 1))
    # def __new__(cls, beta, z=None):
    #     if z is None:
    #         z = tf.range(0, 1, 1 / 200)
    #
    #     return tf.exp(beta(z - 1))


if __name__ == '__main__':
    q = Quadratic()
    print(q)
