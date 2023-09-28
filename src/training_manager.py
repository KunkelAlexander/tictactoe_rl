
import os
import errno
from datetime import datetime
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from .game_manager import GameManager

# Q-learning algorithm

class TrainingManager:

    def __init__(self, game, gui = None):
        """
        Args:
            game (object): The game object representing the game state.
            gui (function, optional): A function for displaying the game state visually.
        """

        self.game        = game
        self.gui         = gui

    def evaluate(self, outputs, agents, n_episode):

        draws              = np.zeros(n_episode)
        did_agent_win      = np.zeros((len(agents), n_episode))
        cum_rewards        = np.zeros((len(agents), n_episode))

        for episode, game in enumerate(outputs):
            for i, (agent_id, game_outcome, cumulative_reward) in enumerate(game):

                draws[episode]            = "DRAW"    == game_outcome
                did_agent_win[i, episode] = "VICTORY" == game_outcome
                cum_rewards  [i, episode] = cumulative_reward

        return draws, did_agent_win, cum_rewards

    def evaluate_agents(self, agents, n_eval, randomise_order, only_legal_actions, debug):
        eval_outputs = []
        for evaluation_episode in range(n_eval):
            game_manager = GameManager(game = self.game, agents = agents, gui = None)
            output       = game_manager.run_game(do_training=False, randomise_order = randomise_order, only_legal_actions=only_legal_actions)
            eval_outputs.append(output)

        draws, did_agent_win, cum_rewards = self.evaluate(eval_outputs, agents, n_eval)

        draw_rate       = np.sum(draws)/n_eval
        victory_rates   = [np.sum(did_agent_win[i, :])/n_eval for i in range(len(agents))]
        avg_cum_rewards = [np.sum(cum_rewards[i, :])/n_eval for i in range(len(agents))]

        if debug:
            print(f"Evaluation on {n_eval} episode: {draw_rate}", end = "")
            for i in range(len(agents)):
                print(f":{victory_rates[i]}", end="")
            print("")

        return draw_rate, victory_rates, avg_cum_rewards

    def run_training(self, config):
        """
        Run training episodes of a game with multiple agents and collect statistics.

        Returns:
            numpy.ndarray: An array of booleans indicating draws for each episode.
            numpy.ndarray: An array of booleans indicating if each agent won for each episode.
            numpy.ndarray: An array of cumulative rewards for each agent over episodes.
        """
        agent_types         = config["agent_types"]
        board_size          = config["board_size"]
        n_episode           = config["n_episode"]
        n_eval              = config["n_eval"]
        eval_freq           = config["eval_freq"]
        train_freq          = config["train_freq"]
        randomise_order     = config["randomise_order"]
        only_legal_actions  = config["only_legal_actions"]
        debug               = config["debug"]

        agents = []
        for i, agent_type in enumerate(agent_types):
            if agent_type == "RANDOM_AGENT":
                from .agent_random import RandomAgent
                agents.append(RandomAgent  (agent_id=i+1, n_actions=9))
            elif agent_type == "TABULAR_Q_AGENT":
                from .agent_tabular_q import TabularQAgent
                agents.append(TabularQAgent(agent_id=i+1, n_actions=9, n_states=3**9, config = config))
            elif agent_type == "SIMPLE_DEEP_Q_AGENT":
                from .agent_deep_q import SimpleDeepQAgent
                agents.append(SimpleDeepQAgent(agent_id=i+1, n_actions=9, n_states=3**9, config = config))
            elif agent_type == "CONVOLUTIONAL_DEEP_Q_AGENT":
                from .agent_deep_q import ConvolutionalDeepQAgent
                agents.append(ConvolutionalDeepQAgent(agent_id=i+1, n_actions=9, n_states=3**9, config = config))
            elif agent_type == "DUAL_DEEP_Q_AGENT":
                from .agent_deep_q import DualDeepQAgent
                agents.append(DualDeepQAgent(agent_id=i+1, n_actions=9, n_states=3**9, config = config))
            elif agent_type == "PRIORITISED_SIMPLE_DEEP_Q_AGENT":
                from .agent_deep_q import PrioritisedSimpleDeepQAgent
                agents.append(PrioritisedSimpleDeepQAgent(agent_id=i+1, n_actions=9, n_states=3**9, config = config))
            elif agent_type == "PRIORITISED_CONVOLUTIONAL_DEEP_Q_AGENT":
                from .agent_deep_q import PrioritisedConvolutionalDeepQAgent
                agents.append(PrioritisedConvolutionalDeepQAgent(agent_id=i+1, n_actions=9, n_states=3**9, config = config))
            elif agent_type == "DUELLING_DEEP_Q_AGENT":
                from .agent_deep_q import DuellingDeepQAgent
                agents.append(DuellingDeepQAgent(agent_id=i+1, n_actions=9, n_states=3**9, config = config))
            elif agent_type == "CONVOLUTIONAL_DUELLING_DEEP_Q_AGENT":
                from .agent_deep_q import ConvDuellingDeepQAgent
                agents.append(ConvDuellingDeepQAgent(agent_id=i+1, n_actions=9, n_states=3**9, config = config))
            elif agent_type == "MINMAX_AGENT":
                from .agent_minmax import MinMaxAgent
                agents.append(MinMaxAgent(agent_id=i+1, n_actions=9, n_states=3**9, game=self.game, act_randomly=False))
            elif agent_type == "RANDOM_MINMAX_AGENT":
                from .agent_minmax import MinMaxAgent
                agents.append(MinMaxAgent(agent_id=i+1, n_actions=9, n_states=3**9, game=self.game, act_randomly=True))
            else:
                raise ValueError(F"Unknown agent type: {agent_type}")

        outputs = []

        for episode in tqdm(range(n_episode)):
            game_manager = GameManager(game = self.game, agents = agents, gui = None)
            game_manager.run_game(do_training=True, randomise_order = randomise_order, only_legal_actions=only_legal_actions, debug=debug)

            if episode % train_freq == 0:
                for agent in agents:
                    agent.train()

            if episode % eval_freq == 0:
                output = self.evaluate_agents(agents = agents, n_eval = n_eval, randomise_order=randomise_order, only_legal_actions=only_legal_actions, debug=debug)
                outputs.append((episode, output))

        mydir = os.path.join(
                    os.getcwd(),
                    "runs",
                    datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                )
        try:
            os.makedirs(mydir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # This was not a "directory exist" error..


        with open(mydir + "/training_information.txt", 'w') as f:
            print(config, file=f)

        episodes        = np.zeros(len(outputs))
        draw_rates      = np.zeros(len(outputs))
        victory_rates   = np.zeros((len(outputs), len(agents)))
        avg_cum_rewards = np.zeros((len(outputs), len(agents)))
        for i, (episode, output) in enumerate(outputs):
            draw_rate, victory_rate, avg_cum_reward = output
            episodes[i]        = episode
            draw_rates[i]      = draw_rate
            victory_rates[i]   = victory_rate
            avg_cum_rewards[i] = avg_cum_reward


        plt.title("Game outcomes")
        plt.ylabel("Fraction of game outcomes")
        plt.xlabel("Number of episodes")

        plt.plot(episodes, draw_rates, label="draws")
        for i, agent in enumerate(agents):
            plt.plot(episodes, victory_rates[:, i], label=f"{agent.name} wins")
        plt.legend()
        plt.show()
        plt.savefig(mydir + "/outcomes.png")
        plt.close()

        plt.title("Cumulative rewards")
        plt.ylabel("Cumulative rewards per episode")
        plt.xlabel("Number of episodes")
        for i, agent in enumerate(agents):
            plt.plot(episodes, avg_cum_rewards[:, i], label=f"{agent.name} reward")
        plt.legend()
        plt.savefig(mydir + "/rewards.png")
        plt.close()

        return agents, draw_rates, victory_rates, avg_cum_rewards

