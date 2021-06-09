import math
import os
import random
from typing import List

import numpy as np
import phe
from keras.backend import random_uniform, cast
from phe.util import powmod, invert

import flwr as fl
import tensorflow as tf

# Make TensorFlow log less verbose
from flwr.client.paillier_client import PaillierClient

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


class MobileClient(PaillierClient):

    def get_parameters(self):  # type: ignore
        return model.get_weights()

    def fit(self, parameters, config=None):  # type: ignore
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        encrypted_weights = self.encrypt_weights(model.get_weights())
        return encrypted_weights, len(x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

    def encrypt_weights(self, w):
        w = np.array(w)
        gm = 1 + (self.public_key.n * w) % self.public_key.nsquare
        # randomness = random_uniform(shape=weights.shape, maxval=pub.n)
        # randomness = np.random.uniform(high=pub.n, size=weights.size)
        r = random.SystemRandom().randrange(1, self.public_key.n)
        rn = powmod(r, self.public_key.n, self.public_key.nsquare)
        c = gm * rn % self.public_key.nsquare
        return c


if __name__ == "__main__":
    # Start Flower client
    fl.client.start_paillier_client("localhost:8080", client=MobileClient())

    # c = np.array(model.get_weights())
    #
    # pub, pri = phe.generate_paillier_keypair(n_length=10)
    #
    # p = pri.p
    # q = pri.q
    # n = p * q
    # nsquare = n * n
    # d1 = (p - 1) * (q - 1)
    #
    # gxd = np.power(c, d1) % nsquare
    # xd = (gxd - 1) // n
    # d2 = invert(d1, n)
    # x = (xd * d2) % n
    # res = cast(List[np.ndarray], x)
    # print(res)
