from typing import Dict, Callable
import flwr as fl
import src.py.flwr.server.strategy as strgy

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": str(0.01),
            "batch_size": str(32),
        }
        return config

    return fit_config


strategy = strgy.FedAvg(
    fraction_fit=0.1,
    min_fit_clients=10,
    min_available_clients=80,
    on_fit_config_fn=get_on_fit_config_fn(),
)

if __name__ == "__main__":
    fl.server.start_server("localhost:8888", config={"num_rounds": 6})
