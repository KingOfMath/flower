import concurrent
import timeit
from logging import INFO, DEBUG, WARNING
from typing import Optional, Tuple, Dict, Union, cast, List

import concurrent.futures

from numpy import array
from phe.util import powmod, invert

from flwr.common import Parameters, Scalar, Weights, weights_to_parameters, GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.common.typing import SendPublicKey
from flwr.server import Server, History

import phe as paillier

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import FitResultsAndFailures, DEPRECATION_WARNING_FIT_ROUND, fit_clients
from flwr.server.strategy import FedPaillier
import numpy as np


class PaillerServer(Server):

    def __init__(self, key_length: int, client_manager: ClientManager, strategy: FedPaillier = None) -> None:
        super().__init__(client_manager, strategy)
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: FedPaillier = strategy if strategy is not None else FedPaillier()
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)

    def send_public_key(self):
        pass

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Getting initial parameters")
        self.parameters = self._get_initial_parameters()
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(rnd=current_round)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(rnd=current_round)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(rnd=current_round, loss=loss_fed)
                    history.add_metrics_distributed(
                        rnd=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def fit_round(
        self, rnd: int
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "fit_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # TODO: send public key
        public_key = SendPublicKey(self.public_key)
        client_public_key = [(client_instructions[0], public_key) for client_instructions in client_instructions]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(send_key_to_client, c, key) for c, key in client_public_key
            ]
            concurrent.futures.wait(futures)

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(client_instructions)
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result = self.strategy.aggregate_fit(rnd, results, failures)

        # TODO: decrypt weights
        aggregated_result = self.decrypt_aggregation(aggregated_result)

        metrics_aggregated: Dict[str, Scalar] = {}
        if aggregated_result is None:
            # Backward-compatibility, this will be removed in a future update
            log(WARNING, DEPRECATION_WARNING_FIT_ROUND)
            parameters_aggregated = None
        else:
            parameters_aggregated = weights_to_parameters(aggregated_result)

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def decrypt_aggregation(self, aggregated_result):
        c = np.array(aggregated_result)

        p = self.private_key.p
        q = self.private_key.q
        n = p * q
        nsquare = n * n
        d1 = (p - 1) * (q - 1)

        gxd = np.power(c, d1) % nsquare
        xd = (gxd - 1) // n
        d2 = invert(d1, n)
        x = (xd * d2) % n
        # res = cast(List[np.ndarray], x)
        return x


def send_key_to_client(client: ClientProxy, key: SendPublicKey) -> Tuple[ClientProxy, SendPublicKey]:
    key_res = client.receive_pk(key)
    return client, key_res
