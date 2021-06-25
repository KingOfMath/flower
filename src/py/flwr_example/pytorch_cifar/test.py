import math

import torch
import cifar
import numpy as np

model_x = cifar.load_model()
weights_X = np.array(model_x.get_weights())

model_y = cifar.load_model()
weights_Y = np.array(model_y.get_weights())

gradients = []
gradients.append(weights_Y - weights_X)
gradients.append((weights_Y - weights_X) + 0.1)


def pariwise(data):
    n = len(data)
    for i in range(n):
        for j in range(i + 1, n):
            yield (data[i], data[j])


def krum_aggregate(gradients, f, m=None):
    n = len(gradients)
    if m is None:
        m = n

    distances = np.array([0] * (n * (n - 1) // 2))
    for i, (x, y) in enumerate(pariwise(tuple(range(n)))):
        dist = gradients[x] - gradients[y]
        for j, item in enumerate(dist):
            dist[j] = np.linalg.norm(item)
        dist = np.linalg.norm(dist)
        if not math.isfinite(dist):
            dist = math.inf
        distances[i] = dist

    scores = list()
    for i in range(n):
        # Collect the distances
        grad_dists = list()
        for j in range(i):
            grad_dists.append(distances[(2 * n - j - 3) * j // 2 + i - 1])
        for j in range(i + 1, n):
            grad_dists.append(distances[(2 * n - i - 3) * i // 2 + j - 1])
        # Select the n - f - 1 smallest distances
        grad_dists.sort()
        scores.append((np.sum(grad_dists[:n - f - 1]), gradients[i], i))

    # Compute the average of the selected gradients
    scores.sort(key=lambda x: x[0])
    accepted_nums = [accepted_num for _, _, accepted_num in scores[:m]]

    return [sum(grad for _, grad, _ in scores[:m]) / m, accepted_nums]


[krum_weight, krum_accepted] = krum_aggregate(gradients=gradients, f=1)

print(krum_weight, krum_accepted)
