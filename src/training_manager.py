
import os
import errno
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from .game_manager import GameManager

# Q-learning algorithm

class TrainingManager:

    def __init__(self, game, gui=None):
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
            game_manager = GameManager(game=self.game, agents=agents, gui=None)
            output       = game_manager.run_game(do_training=False, randomise_order=randomise_order, only_legal_actions=only_legal_actions)
            eval_outputs.append(output)

        draws, did_agent_win, cum_rewards = self.evaluate(eval_outputs, agents, n_eval)

        draw_rate       = np.sum(draws)/n_eval
        victory_rates   = [np.sum(did_agent_win[i, :])/n_eval for i in range(len(agents))]
        avg_cum_rewards = [np.sum(cum_rewards[i, :])/n_eval for i in range(len(agents))]

        if debug:
            print(f"Evaluation on {n_eval} episode: {draw_rate}", end="")
            for i in range(len(agents)):
                print(f":{victory_rates[i]}", end="")
            print("")

        return draw_rate, victory_rates, avg_cum_rewards

    def run_training(self, config, agents=None, callback=None):
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
        plot_debug          = config["plot_debug"]
        do_training         = config.get("do_training", True)
        callback_freq       = config.get("callback_freq", 1e15)

        if agents is None:
            agents = []
            for i, agent_type in enumerate(agent_types):
                if agent_type == "RANDOM_AGENT":
                    from .agent_random import RandomAgent
                    agents.append(RandomAgent  (agent_id=i+1, n_actions=9))
                elif agent_type == "TABULAR_Q_AGENT":
                    from .agent_tabular_q import TabularQAgent
                    agents.append(TabularQAgent(agent_id=i+1, n_actions=9, n_states=3**9, config=config))
                elif agent_type == "SARSA_AGENT":
                    from .agent_sarsa import SARSAAgent
                    agents.append(SARSAAgent(agent_id=i+1, n_actions=9, n_states=3**9, config=config))
                elif agent_type == "MINMAX_AGENT":
                    from .agent_minmax import MinMaxAgent
                    agents.append(MinMaxAgent(agent_id=i+1, n_actions=9, n_states=3**9, game=self.game, act_randomly=False))
                elif agent_type == "RANDOM_MINMAX_AGENT":
                    from .agent_minmax import MinMaxAgent
                    agents.append(MinMaxAgent(agent_id=i+1, n_actions=9, n_states=3**9, game=self.game, act_randomly=True))
                else:
                    raise ValueError(F"Unknown agent type: {agent_type}")

        results = {
            "config": config,
            "timestamp": datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "evaluation_episodes": [],
            "metrics": defaultdict(list),  # keys: 'draw_rate', 'victory_rate_<agent>', 'reward_<agent>'
        }

        for episode in range(n_episode):
            game_manager = GameManager(game=self.game, agents=agents, gui=None)
            game_manager.run_game(
                do_training=do_training,
                randomise_order=randomise_order,
                only_legal_actions=only_legal_actions,
                debug=debug
            )

            if episode % train_freq == 0 and do_training:
                for agent in agents:
                    agent.train()

            if episode % eval_freq == 0:
                output = self.evaluate_agents(
                    agents=agents,
                    n_eval=n_eval,
                    randomise_order=randomise_order,
                    only_legal_actions=only_legal_actions,
                    debug=debug
                )
                draw_rate, victory_rate, avg_cum_reward = output
                results["evaluation_episodes"].append(episode)
                results["metrics"]["draw_rate"].append(draw_rate)
                for i, agent in enumerate(agents):
                    results["metrics"][f"victory_rate_{agent.name}"].append(victory_rate[i])
                    results["metrics"][f"reward_{agent.name}"].append(avg_cum_reward[i])
                    # Safe loss extraction
                    loss = None
                    if hasattr(agent, "training_log") and agent.training_log:
                        last_entry = agent.training_log[-1]
                        if "loss" in last_entry:
                            loss = last_entry["loss"]
                    results["metrics"][f"loss_{agent.name}"].append(loss)

            if episode % callback_freq == 0 and callback:
                callback(game_manager=game_manager)

        # Save config to a log directory
        mydir = os.path.join(os.getcwd(), "runs", results["timestamp"])
        os.makedirs(mydir, exist_ok=True)

        with open(os.path.join(mydir, "training_information.txt"), 'w') as f:
            print(config, file=f)

        # Optional plotting
        if plot_debug:
            import matplotlib.pyplot as plt

            x = results["evaluation_episodes"]
            plt.title("Game outcomes")
            plt.ylabel("Fraction of game outcomes")
            plt.xlabel("Number of episodes")
            plt.plot(x, results["metrics"]["draw_rate"], label="draws")
            for agent in agents:
                plt.plot(x, results["metrics"][f"victory_rate_{agent.name}"], label=f"{agent.name} wins")
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(mydir, "outcomes.png"))
            plt.close()

            plt.title("Cumulative rewards")
            plt.ylabel("Cumulative rewards per episode")
            plt.xlabel("Number of episodes")
            for agent in agents:
                plt.plot(x, results["metrics"][f"reward_{agent.name}"], label=f"{agent.name} reward")
            plt.legend()
            plt.savefig(os.path.join(mydir, "rewards.png"))
            plt.show()
            plt.close()

        return {
            "agents": agents,
            "results": results
        }