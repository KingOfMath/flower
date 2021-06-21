# import math
# import os
# import random
# from typing import List
#
# import numpy as np
# import phe
# from phe.util import powmod, invert
# from keras.backend import random_uniform, cast
#
# import flwr as fl
import tensorflow as tf
#
# # Make TensorFlow log less verbose
# from flwr.client.paillier_client import PaillierClient
#
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
#
# # Load CIFAR-10 dataset
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#
#
# class MobileClient(PaillierClient):
#
#     def get_parameters(self):  # type: ignore
#         return model.get_weights()
#
#     def fit(self, parameters, config=None):  # type: ignore
#         model.set_weights(parameters)
#         model.fit(x_train, y_train, epochs=1, batch_size=32)
#         encrypted_weights = self.encrypt_weights(model.get_weights())
#         return encrypted_weights, len(x_train), {}
#
#     def evaluate(self, parameters, config):  # type: ignore
#         model.set_weights(parameters)
#         loss, accuracy = model.evaluate(x_test, y_test)
#         return loss, len(x_test), {"accuracy": accuracy}
#
#     def encrypt_weights(self, w):
#         w = np.array(w)
#         gm = (1 + self.public_key.n * w) % self.public_key.nsquare
#         # r = np.random.uniform(high=pub.n, size=weights.size)
#
#         r = random.SystemRandom().randrange(1, self.public_key.n)
#         rn = powmod(r, self.public_key.n, self.public_key.nsquare)
#         c = gm % self.public_key.nsquare
#         return c


if __name__ == "__main__":
    # Start Flower client
    # fl.client.start_paillier_client("localhost:8080", client=MobileClient())

    # raw = np.array(model.get_weights())

    import numpy as np
    import phe
    from phe.util import powmod, invert
    import random

    pub, pri = phe.generate_paillier_keypair()

    weights = model.get_weights()

    # w = 0.8
    # w = np.int64(3)
    # w = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.longdouble)
    w = 3.141592653

    # ssss = pub.encrypt(w)
    # sss = pri.decrypt(ssss)

    int_rep = int(round(w * pow(16, 14))) % pub.n
    neg_plaintext = pub.n - int_rep
    neg_ciphertext = (pub.n * neg_plaintext + 1) % pub.nsquare
    nude_ciphertext = invert(neg_ciphertext, pub.nsquare)

    # gm = invert(((pub.n - w) * pub.n + 1) % pub.nsquare, pub.nsquare)
    gm = (1 + pub.n * int_rep) % pub.nsquare

    # r = np.random.uniform(high=pub.n, size=weights.size)

    r = random.SystemRandom().randrange(1, pub.n)
    rn = powmod(r, pub.n, pub.nsquare)
    c = (gm * rn) % pub.nsquare

    p = pri.p
    q = pri.q
    n = p * q
    nsquare = n * n
    ps = pri.psquare
    qs = pri.qsquare
    # d1 = (p - 1) * (q - 1)
    #
    # gxd = np.power(c, d1) % nsquare
    # xd = (gxd - 1) // n
    # d2 = invert(d1, n)
    # x = (xd * d2) % n
    # print(x)

    # t1 = np.power(c, p - 1) % ps
    # t2 = np.power(c, q - 1) % qs
    q1 = powmod(c, p - 1, ps)
    q2 = powmod(c, q - 1, qs)

    # dp = (np.power(c, p - 1) % ps - 1) // p * pri.hp % p
    # dq = (np.power(c, q - 1) % qs - 1) // q * pri.hq % q
    gp = (powmod(c, p - 1, ps) - 1) // p * pri.hp % p
    gq = (powmod(c, q - 1, qs) - 1) // q * pri.hq % q

    u = (gq - gp) * pri.p_inverse % q
    res = gp + (u * p)

    print(res)
    # decrypt_to_p = powmod(ciphertext, self.p-1, self.psquare), self.p) * self.hp % self.p
    # decrypt_to_q = powmod(ciphertext, self.q-1, self.qsquare), self.q) * self.hq % self.q
    # return self.crt(decrypt_to_p, decrypt_to_q)
