"""Sacred configuration for NTRIL training."""

from sacred import Experiment

from imitation.scripts.ingredients import environment, expert, logging

train_ntril_ex = Experiment(
    "train_ntril",
    ingredients=[
        environment.environment_ingredient,
        expert.expert_ingredient,
        logging.logging_ingredient,
    ],
)
