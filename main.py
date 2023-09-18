import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import os 
import errno
from datetime import datetime
import argparse 

from tictactoe import TicTacToe
from game_manager import GameManager
from agents.random_agent import RandomAgent
from agents.tabular_q_agent import TabularQAgent
from agents.dense_q_agent import DenseQAgent




def gui(tictactoe, agent_id, events, game_over):
    """
    Display the current game state in a text-based user interface.

    Args:
        tictactoe (TicTacToe): The TicTacToe game object.
        agent_id (int):        The ID of the current agent making a move.
        events (list):         A list of events that occurred during the move (e.g., "INVALID_MOVE", "VICTORY").
        game_over (bool):      True if the game is over, False otherwise.

    Displays:
        The current game board with player symbols ('X', 'O') and game events.
    """
    print(f"Agent {agent_id}; Events: {events}; Game over? {game_over}")

    ny, nx = tictactoe.board.shape
    shapes = {
        0: " ", 1: "X", 2: "O", 3: "△", 4: "▢",
    }
    for i in range(ny):
        print((1 + nx * 4) * "-")
        for j in range(nx):
            print(f"| {shapes[tictactoe.board[i, j]]} ", end="")
        print("|")
    print((1 + nx * 4) * "-")


# Q-learning algorithm

class TrainingManager:

    def __init__(self, game, agent_types, gui = None):
        """
        Args:
            game (object): The game object representing the game state.
            agents (list): A list of agent objects participating in the training.
            board_size (int, optional): Size of the game board. Default is 3.
            n_episodes (int, optional): Number of training episodes. Default is 50000.
            gui (function, optional): A function for displaying the game state visually.
        """

        self.game        = game 
        self.gui         = gui 
        self.agent_types = agent_types

    def run_training(self, board_size, n_episode, window_size, learning_rate_decay, exploration_decay, randomise_order = True, discount = 0.9, learning_rate = 0.1, exploration = 0.95): 
        """
        Run training episodes of a game with multiple agents and collect statistics.

        Returns:
            numpy.ndarray: An array of booleans indicating draws for each episode.
            numpy.ndarray: An array of booleans indicating if each agent won for each episode.
            numpy.ndarray: An array of cumulative rewards for each agent over episodes.
        """

        agents = []
        for i, agent_type in enumerate(self.agent_types): 
            if agent_type == "RANDOM_AGENT":
                agents.append(RandomAgent  (agent_id=i+1, n_actions=9))
            elif agent_type == "TABULAR_Q_AGENT": 
                agents.append(TabularQAgent(agent_id=i+1, n_actions=9, n_states=3**9, learning_rate=learning_rate, discount=discount, exploration=exploration,learning_rate_decay=learning_rate_decay, exploration_decay=exploration_decay))
            elif agent_type == "DENSE_Q_AGENT": 
                agents.append(DenseQAgent(agent_id=i+1, n_actions=9, n_states=3**9, learning_rate=learning_rate, discount=discount, exploration=exploration,learning_rate_decay=learning_rate_decay, exploration_decay=exploration_decay, batch_size=64, replay_buffer_size=10000))
            else: 
                raise ValueError(F"Unknown agent type: {agent_type}")

        draws         = np.zeros(n_episode)
        did_agent_win = np.zeros((len(agents), n_episode))
        cum_rewards   = np.zeros((len(agents), n_episode))

        for episode in tqdm(range(n_episode)):
            game_manager     = GameManager(game = self.game, agents = agents, gui = self.gui)
            agent_id, events = game_manager.run_game(do_training=True, randomise_order = randomise_order) 

            draws[episode] = "DRAW" in events

            for i, agent in enumerate(agents): 
                did_agent_win[i, episode] = ("VICTORY" in events) and (agent_id == agent.agent_id)
                cum_rewards  [i, episode] = agent.cumulative_reward


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

        np.savetxt(mydir + "/draws.txt", draws)
        for i, agent in enumerate(agents): 
            np.savetxt(mydir + "/did_agent_win.txt", did_agent_win[i])
            np.savetxt(mydir + "/cum_reward.txt",    cum_rewards[i])
        
        moving_avg = lambda data, window_size: np.convolve(data, np.ones(window_size), mode='valid') / window_size
        
        plt.title("Game outcomes")
        plt.plot(moving_avg(draws, window_size), label="draws")
        for i, agent in enumerate(agents): 
            plt.plot(moving_avg(did_agent_win[i], window_size), label=f"{agent.name} wins")
        plt.legend()
        plt.savefig(mydir + "/outcomes.png")
        plt.close()

        plt.title("Cumulative rewards")
        for i, agent in enumerate(agents): 
            plt.plot(moving_avg(cum_rewards[i], window_size), label=f"{agent.name} reward")
        plt.legend()
        plt.savefig(mydir + "/rewards.png")
        plt.close()

        return draws, did_agent_win, cum_rewards




# parse input parameters
argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argParser.add_argument("-n", "--n_episode",   type=int, help="Number of episodes for training.", default=50000)
argParser.add_argument("-w", "--window_size", type=int, help="Window size for plotting.", default=500)
argParser.add_argument("-s", "--board_size",  type=int, help="Size N of NxN board.", default=3)
argParser.add_argument("-a", "--agents",      type=str, help="Space separated list of agents", default="RANDOM_AGENT TABULAR_Q_AGENT")

args = argParser.parse_args()

agent_types = args.agents.split(sep=",")

training_manager = TrainingManager( game = TicTacToe(board_size  = args.board_size, agent_count = len(agent_types)), 
                                    agent_types = agent_types,
                                    gui  = None) 

training_manager.run_training(
                            board_size          = args.board_size, 
                            n_episode           = args.n_episode, 
                            window_size         = args.window_size, 
                            learning_rate_decay = 1 - 1e-5, 
                            exploration_decay   = 1 - 1e-2,
                            exploration         = 0.5,
                            learning_rate       = 0.1,
                            )


def hyperparameter_training(): 
    training_success = []
    for learning_rate_decay in np.logspace(-2, -6, 5):
        for exploration_decay in np.logspace(-2, -6, 5):
            draws, did_agent_win, cum_reward = training_manager.run_training(
                                        board_size          = args.board_size, 
                                        n_episode           = args.n_episode, 
                                        window_size         = args.window_size, 
                                        learning_rate_decay = 1 - learning_rate_decay, 
                                        exploration_decay   = 1 - exploration_decay,
                                        exploration         = 0.05,
                                        learning_rate       = 0.1,
                                        )
            training_success.append([learning_rate_decay, np.mean(cum_reward[1, -2000:]), exploration_decay])

    for t in training_success:
        print(t)