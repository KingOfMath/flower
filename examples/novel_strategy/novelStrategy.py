from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy


class novelStrategy(Strategy):

    def configure_fit(self, rnd, weights, client_manager: ClientManager):

        pass

    def aggregate_fit(self, rnd, results, failures):
        pass

    def configure_evaluate(self, rnd, weights, client_manager):
        pass

    def aggregate_evaluate(self, rnd, results, failures):
        pass

    def evaluate(self, weights):
        pass
