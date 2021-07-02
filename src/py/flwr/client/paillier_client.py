import timeit
from abc import abstractmethod
from typing import List, Tuple, cast, Optional

import numpy as np

# Load and compile Keras model
import phe

from .client import Client
from .numpy_client import DEPRECATION_WARNING_FIT, DEPRECATION_WARNING_EVALUATE_0, \
    DEPRECATION_WARNING_EVALUATE_1
from ..common import ParametersRes, weights_to_parameters, FitIns, FitRes, parameters_to_weights, Metrics, EvaluateIns, \
    EvaluateRes

# Define Flower client
class PaillierClient:

    def __init__(self):
        self.public_key = None

    @abstractmethod
    def get_parameters(self):
        """"""

    @abstractmethod
    def fit(self, parameters, config):
        """"""

    @abstractmethod
    def evaluate(self, parameters, config):
        """"""

    @abstractmethod
    def receive_public_keys(self, public_key):
        """Send public key to client"""


class PaillierClientWrapper(Client):
    def __init__(self, paillier_client: PaillierClient) -> None:
        self.paillier_client = paillier_client

    def get_parameters(self) -> ParametersRes:
        """Return the current local model parameters."""
        parameters = self.paillier_client.get_parameters()
        parameters_proto = weights_to_parameters(parameters)
        return ParametersRes(parameters=parameters_proto)

    def fit(self, ins: FitIns) -> FitRes:
        """Refine the provided weights using the locally held dataset."""
        # Deconstruct FitIns
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)

        # Train
        fit_begin = timeit.default_timer()
        results = self.paillier_client.fit(parameters, ins.config)
        if len(results) == 2:
            print(DEPRECATION_WARNING_FIT)
            results = cast(Tuple[List[np.ndarray], int], results)
            parameters_prime, num_examples = results
            metrics: Optional[Metrics] = None
        elif len(results) == 3:
            results = cast(Tuple[List[np.ndarray], int, Metrics], results)
            parameters_prime, num_examples, metrics = results

        # Return FitRes
        fit_duration = timeit.default_timer() - fit_begin
        parameters_prime_proto = weights_to_parameters(parameters_prime)
        return FitRes(
            parameters=parameters_prime_proto,
            num_examples=num_examples,
            num_examples_ceil=num_examples,  # Deprecated
            fit_duration=fit_duration,  # Deprecated
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the provided parameters using the locally held dataset."""
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)

        results = self.paillier_client.evaluate(parameters, ins.config)
        if len(results) == 3:
            if (
                isinstance(results[0], float)
                and isinstance(results[1], int)
                and isinstance(results[2], dict)
            ):
                # Forward-compatible case: loss, num_examples, metrics
                results = cast(Tuple[float, int, Metrics], results)
                loss, num_examples, metrics = results
                evaluate_res = EvaluateRes(
                    loss=loss,
                    num_examples=num_examples,
                    metrics=metrics,
                )
            elif (
                isinstance(results[0], int)
                and isinstance(results[1], float)
                and isinstance(results[2], float)
            ):
                # Legacy case: num_examples, loss, accuracy
                # This will be removed in a future release
                print(DEPRECATION_WARNING_EVALUATE_0)
                results = cast(Tuple[int, float, float], results)
                num_examples, loss, accuracy = results
                evaluate_res = EvaluateRes(
                    loss=loss,
                    num_examples=num_examples,
                    accuracy=accuracy,  # Deprecated
                )
            else:
                raise Exception(
                    "Return value expected to be of type (float, int, dict)."
                )
        elif len(results) == 4:
            # Legacy case: num_examples, loss, accuracy, metrics
            # This will be removed in a future release
            print(DEPRECATION_WARNING_EVALUATE_1)
            results = cast(Tuple[int, float, float, Metrics], results)
            assert isinstance(results[0], int)
            assert isinstance(results[1], float)
            assert isinstance(results[2], float)
            assert isinstance(results[3], dict)
            num_examples, loss, accuracy, metrics = results
            evaluate_res = EvaluateRes(
                loss=loss,
                num_examples=num_examples,
                accuracy=accuracy,  # Deprecated
                metrics=metrics,
            )
        return evaluate_res

    def receive_public_keys(self, public_key):
        paillier_public_key = phe.PaillierPublicKey(n=public_key.n)
        self.paillier_client.public_key = paillier_public_key
