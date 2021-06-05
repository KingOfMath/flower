import os

import flwr as fl
from examples.paillier_homomorphic_encryption.paillier_client import PaillierClient

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == "__main__":
    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=PaillierClient())
