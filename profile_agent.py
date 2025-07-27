import matplotlib.pyplot as plt
import numpy as np

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import pandas as pd
import os

from src.tictactoe import TicTacToe
from src.training_manager import TrainingManager

from src.agent_tabular_q import TabularQAgent
from src.agent_random import RandomAgent
from src.agent_minmax import MinMaxAgent
from src.agent_deep_q import DeepQAgent

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

# Force TensorFlow into deterministic op mode (TF 2.x)
tf.config.experimental.enable_op_determinism()


SEEDS   = 10
ACTIONS = 9
STATES = 3**9


game = TicTacToe(board_size  = 3, agent_count = 2)
training_manager = TrainingManager( game = game )

base_config = {
    "agent_types"         : ["RANDOM_AGENT", "RANDOM_AGENT"],
    "board_size"          : 3,
    "n_episode"           : 3000,   # Number of training episodes
    "n_eval"              : 100,    # Number of evaluation episodes every eval_freq training episodes
    "eval_freq"           : 100,
    "train_freq"          : 1,      # Train models every train_freq training episodes
    "grad_steps"          : 2,      # Number of gradient updates per training step
    "discount"            : 0.8,    # Discount in all Q learning algorithms
    "learning_rate_decay" : 1,
    "exploration"         : 1.0,    # Initial exploration rate
    "exploration_decay"   : 1e-2,   # Decrease of exploration rate for every action
    "exploration_min"     : 0.0,
    "learning_rate"       : 1e-2,
    "randomise_order"     : False,  # Randomise starting order of agents for every game
    "only_legal_actions"  : True,   # Have agents only take legal actions
    "debug"               : False,  # Print loss and evaluation information during training
    "plot_debug"          : False,  # Plot game outcomes
    "batch_size"          : 128,    # Batch size for DQN algorithm
    "replay_buffer_size"  : 10000,  # Replay buffer for DQN algorithm
    "replay_buffer_min"   : 1000,   # minimum size before we start training
    "target_update_tau"   : 0.1,    # Weight for update in dual DQN architecture target = (1 - tau) * target + tau * online
    "target_update_freq"  : 10,     # Update target network every n episodes
    "target_update_mode"  : "hard", # "hard": update every target_update freq or "soft": update using Polyakov rule with target_update_tau
    "initial_q"           : 0.6,    # Initial Q value for tabular Q learning
    "board_encoding"      : "encoding_tutorial"
}


def run_ensemble(make_agent_1, make_agent_2, SEEDS=10, base_config=base_config):

    rows = []
    for seed in range(SEEDS):
        random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
        config = dict(base_config)

        result = training_manager.run_training(config, [make_agent_1(config), make_agent_2(config)])

        agents = result["agents"]
        results = result["results"]
        metrics = results["metrics"]
        episodes = results["evaluation_episodes"]

        # Identify agent2 by ID or name (assuming agent1 is fixed MinMaxAgent)
        agent2 = next(a for a in agents if isinstance(a, DeepQAgent))

        for ep_idx, ep in enumerate(episodes):
            win_rate = metrics[f"victory_rate_{agent2.name}"][ep_idx]
            draw_rate = metrics["draw_rate"][ep_idx]
            loss = metrics[f"loss_{agent2.name}"][ep_idx]

            rows += [
                dict(seed=seed, episode=ep, metric="win",  val=win_rate),
                dict(seed=seed, episode=ep, metric="draw", val=draw_rate)
            ]

            if loss is not None:
                rows.append(dict(seed=seed, episode=ep, metric="loss", val=loss))

    df = pd.DataFrame(rows)
    return df


# Q-network factory
def build_simple_dqn_model(input_shape, num_actions, num_hidden_layers=1, hidden_layer_size=64):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_hidden_layers):
        x = layers.Dense(hidden_layer_size, activation='relu')(x)
    outputs = layers.Dense(num_actions, activation='linear')(x)
    return models.Model(inputs=inputs, outputs=outputs)

agent1 = MinMaxAgent(1, ACTIONS, 3**9, game, True)


def make_agent_1(cfg: dict):
    return agent1

def make_agent_2(cfg: dict):
    ag = DeepQAgent(2, ACTIONS, STATES, cfg)
    net = build_simple_dqn_model(ag.input_shape, ag.n_actions, hidden_layer_size=128)
    net.compile(optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]), loss="mse")
    ag.online_model = net
    ag.target_model = net          # 🔑 online == target
    return ag


cfg = dict(base_config)
cfg["batch_size"] = 512
cfg["eval_freq"] = 2999
df = run_ensemble(make_agent_1, make_agent_2, SEEDS=1, base_config=cfg)      # <== jouw trainings-loop
