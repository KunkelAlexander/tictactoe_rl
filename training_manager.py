
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import os 
import errno
from datetime import datetime

from game_manager import GameManager

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
   
    def evaluate_agents(self, agents, n_eval, randomise_order, only_legal_actions):
        eval_outputs = []
        for evaluation_episode in range(n_eval):
            game_manager = GameManager(game = self.game, agents = agents, gui = None)
            output       = game_manager.run_game(do_training=False, randomise_order = randomise_order, only_legal_actions=only_legal_actions) 
            eval_outputs.append(output) 

        draws, did_agent_win, cum_rewards = self.evaluate(eval_outputs, agents, n_eval)
        print(f"Evaluation on {n_eval} episode: {np.sum(draws)/n_eval}", end = "") 
        for i, agent in enumerate(agents): 
            print(f":{np.sum(did_agent_win[i, :])/n_eval}", end="")
        print("")

        return draws, did_agent_win, cum_rewards

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
        window_size         = config["window_size"]   
        discount            = config["discount"]      
        learning_rate_decay = config["learning_rate_decay"]
        exploration_decay   = config["exploration_decay"]
        exploration         = config["exploration"]   
        learning_rate       = config["learning_rate"] 
        randomise_order     = config["randomise_order"]
        only_legal_actions  = config["only_legal_actions"]
        debug               = config["debug"]

        agents = []
        for i, agent_type in enumerate(agent_types): 
            if agent_type == "RANDOM_AGENT":
                from agents.random_agent import RandomAgent
                agents.append(RandomAgent  (agent_id=i+1, n_actions=9))
            elif agent_type == "TABULAR_Q_AGENT": 
                from agents.tabular_q_agent import TabularQAgent
                agents.append(TabularQAgent(agent_id=i+1, n_actions=9, n_states=3**9, learning_rate=learning_rate, discount=discount, exploration=exploration, learning_rate_decay=learning_rate_decay, exploration_decay=exploration_decay))
            elif agent_type == "DENSE_Q_AGENT": 
                from agents.dense_q_agent import DenseQAgent
                agents.append(DenseQAgent(agent_id=i+1, n_actions=9, n_states=3**9, learning_rate=learning_rate, discount=discount, exploration=exploration, learning_rate_decay=learning_rate_decay, exploration_decay=exploration_decay, batch_size=64, replay_buffer_size=10000))
            elif agent_type == "MINMAX_AGENT": 
                from agents.minmax_agent import MinMaxAgent
                agents.append(MinMaxAgent(agent_id=i+1, n_actions=9, n_states=3**9, game=self.game, act_randomly=False))            
            elif agent_type == "RANDOM_MINMAX_AGENT": 
                from agents.minmax_agent import MinMaxAgent
                agents.append(MinMaxAgent(agent_id=i+1, n_actions=9, n_states=3**9, game=self.game, act_randomly=True))
            else: 
                raise ValueError(F"Unknown agent type: {agent_type}")

        outputs = []
        print("episode")

        for episode in tqdm(range(n_episode)):
            game_manager = GameManager(game = self.game, agents = agents, gui = None)
            output       = game_manager.run_game(do_training=True, randomise_order = randomise_order, only_legal_actions=only_legal_actions, debug=debug) 
            outputs.append(output) 

            if episode % eval_freq == 0: 
                self.evaluate_agents(agents = agents, n_eval = n_eval, randomise_order=randomise_order, only_legal_actions=only_legal_actions)

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
            print(f'agent_types         = {agent_types}',         file=f)
            print(f'n_episode           = {n_episode}',           file=f)
            print(f'window_size         = {window_size}',         file=f)
            print(f'discount            = {discount}',            file=f)
            print(f'learning_rate       = {learning_rate}',       file=f)
            print(f'learning_rate_decay = {learning_rate_decay}', file=f)
            print(f'exploration         = {exploration}',         file=f)
            print(f'exploration_decay   = {exploration_decay}',   file=f)
            print(f'randmomise_order    = {randomise_order}',     file=f)

        draws, did_agent_win, cum_rewards = self.evaluate(outputs, agents, n_episode) 
        np.savetxt(mydir + "/draws.txt", draws)
        for i, agent in enumerate(agents): 
            np.savetxt(mydir + f"/did_agent_{agent.agent_id}_win.txt", did_agent_win[i])
            np.savetxt(mydir + "/cum_reward.txt",    cum_rewards[i])
        
        moving_avg = lambda data, window_size: np.convolve(data, np.ones(window_size), mode='valid') / window_size
        
        plt.title("Game outcomes")
        plt.ylabel("Fraction of game outcomes")
        plt.xlabel("Number of episodes")
        plt.plot(moving_avg(draws, window_size), label="draws")
        for i, agent in enumerate(agents): 
            plt.plot(moving_avg(did_agent_win[i], window_size), label=f"{agent.name} wins")
        plt.legend()
        plt.show() 
        plt.savefig(mydir + "/outcomes.png")
        plt.close()

        plt.title("Cumulative rewards")
        plt.ylabel("Cumulative rewards per episode")
        plt.xlabel("Number of episodes")
        for i, agent in enumerate(agents): 
            plt.plot(moving_avg(cum_rewards[i], window_size), label=f"{agent.name} reward")
        plt.legend()
        plt.show() 
        plt.savefig(mydir + "/rewards.png")
        plt.close()

        return agents, draws, did_agent_win, cum_rewards

