import copy

import tensorflow as tf
import numpy as np

# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
from flwr_example.pytorch_cifar import cifar

model = cifar.Net

def to_ndarray(w_locals):
    for user_idx in range(len(w_locals)):
        for key in w_locals[user_idx]:
            w_locals[user_idx][key] = w_locals[user_idx][key].cpu().numpy()
    return w_locals


w_locals = to_ndarray(model.get_weights())
#
# user_one_d = []
# for user_idx in range(len(w_locals)):
#     tmp = np.array([])
#     for key in w_locals[user_idx]:
#         data_idx_key = np.array(w_locals[user_idx][key]).flatten()
#         tmp = copy.deepcopy( np.hstack((tmp, data_idx_key)) )
#     user_one_d.append(tmp)


