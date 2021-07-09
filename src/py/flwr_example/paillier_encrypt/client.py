import math
import os
import random
import sys

import numpy as np
import phe
from phe import EncodedNumber
from phe.util import powmod, invert
from keras.backend import random_uniform, cast
import time
import multiprocessing
from joblib import Parallel, delayed
import tensorflow as tf
import flwr as fl

# Make TensorFlow log less verbose
from flwr.client.paillier_client import PaillierClient

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

N_JOBS = multiprocessing.cpu_count()


class MobileClient(PaillierClient):

    def get_parameters(self):  # type: ignore
        return model.get_weights()

    def fit(self, parameters, config=None):  # type: ignore
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        time_start = time.time()
        encrypted_weights = self.encrypt_weights(model.get_weights())
        time_end = time.time()
        print('encrypt time cost', time_end - time_start, 's')
        return encrypted_weights, len(x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

    def encrypt_weights(self, weights):
        en = []
        time_start = time.time()
        for matrix in weights:
            origin_shape = matrix.shape

            if len(matrix.shape) == 1:
                matrix = np.expand_dims(matrix, axis=0)

            print('encrypting matrix shaped ' + str(origin_shape))
            matrix = np.squeeze(np.reshape(matrix, (1, -1)))
            encrypt_matrix = Parallel(n_jobs=N_JOBS)(delayed(self.public_key.encrypt)(num.item()) for num in matrix)
            encrypt_matrix = np.expand_dims(encrypt_matrix, axis=0)
            encrypt_matrix = np.reshape(encrypt_matrix, origin_shape)
            en.append(encrypt_matrix)
        time_end = time.time()
        print('encrypt time cost', time_end - time_start, 's')

        return en


if __name__ == "__main__":
    # Start Flower client
    # fl.client.start_paillier_client("localhost:8080", client=MobileClient())

    pub, pri = phe.generate_paillier_keypair(n_length=100)

    weights = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])


    def encode(data):
        bin_flt_exponent = math.frexp(data)[1]
        bin_lsb_exponent = bin_flt_exponent - sys.float_info.mant_dig
        prec_exponent = math.floor(bin_lsb_exponent / math.log(16, 2))

        int_rep = int(round(data * pow(16, -prec_exponent)))

        if abs(int_rep) > pub.max_int:
            raise ValueError('Integer needs to be within +/- %d but got %d'
                             % (pub.max_int, int_rep))

        encoding = int_rep % pub.n
        obfuscator = 1
        ciphertext = pub.raw_encrypt(encoding, r_value=obfuscator)
        return ciphertext

    # def decode(data):
    #     encoded = pri.raw_decrypt(data)
    #     if self.encoding >= self.public_key.n:
    #         # Should be mod n
    #         raise ValueError('Attempted to decode corrupted number')
    #     elif self.encoding <= self.public_key.max_int:
    #         # Positive
    #         mantissa = self.encoding
    #     elif self.encoding >= self.public_key.n - self.public_key.max_int:
    #         # Negative
    #         mantissa = self.encoding - self.public_key.n
    #     else:
    #         raise OverflowError('Overflow detected in decrypted number')
    #
    #     return mantissa * pow(self.BASE, self.exponent)

    # en = []
    # for matrix in weights:
    #     origin_shape = matrix.shape
    #
    #     if len(matrix.shape) == 1:
    #         matrix = np.expand_dims(matrix, axis=0)
    #
    #     print('encrypting matrix shaped ' + str(origin_shape))
    #     matrix = np.squeeze(np.reshape(matrix, (1, -1)))
    #     encrypt_matrix = Parallel(n_jobs=N_JOBS)(delayed(encode)(num.item()) for num in matrix)
    #     encrypt_matrix = np.expand_dims(encrypt_matrix, axis=0)
    #     encrypt_matrix = np.reshape(encrypt_matrix, origin_shape)
    #     en.append(encrypt_matrix)
    #
    # de = []
    # time_start = time.time()
    # for encrypt_matrix in en:
    #     origin_shape = encrypt_matrix.shape
    #     print('decrypting matrix shaped ' + str(origin_shape))
    #     encrypt_matrix = np.squeeze(np.reshape(encrypt_matrix, (1, -1)))
    #     decrypt_matrix = Parallel(n_jobs=N_JOBS)(delayed(pri.raw_decrypt)(num) for num in encrypt_matrix)
    #     decrypt_matrix = np.expand_dims(decrypt_matrix, axis=0)
    #     decrypt_matrix = np.reshape(decrypt_matrix, origin_shape)
    #     de.append(decrypt_matrix)
    # time_end = time.time()
    # print('decrypt time cost', time_end - time_start, 's')
    #
    # print(de)
