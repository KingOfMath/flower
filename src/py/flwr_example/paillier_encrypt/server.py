import flwr as fl
from flwr_example.paillier_encrypt.paillier_server import PaillerServer
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy.fedpaillier import FedPaillier

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.server.start_server("localhost:8080", config={"num_rounds": 3},
                           server=PaillerServer(client_manager=SimpleClientManager(), key_length=60),
                           strategy=FedPaillier())
