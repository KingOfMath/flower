import tensorflow as tf
import flwr as fl

# Load and compile Keras model
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# Define Flower client
class PaillierClient(fl.client.NumPyClient):

    def __init__(self):
        self.public_key = None

    def receive_public_keys(self, public_key):
        self.public_key = public_key

    def get_parameters(self):  # type: ignore
        return model.get_weights()

    def fit(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        # encrypted_weights = self.encrypt_vector(model.get_weights())
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

    def encrypt_vector(self, x):
        return [self.public_key.encrypt(i) for i in x]
