import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import os 
import errno
from datetime import datetime

class TicTacToe:
    FIELD_EMPTY = 0

    def __init__(self, board_size=3, agent_count=2):
        """
        Initialize the game

        :param board_size: Size of the board (e.g., 3 for a 3x3 board).
        :param agent_count: Number of players (agents) in the game.
        """
        self.board_size  = board_size
        self.agent_count = agent_count

    def start_game(self): 
        """
        Initialize the TicTacToe board
        """
        self.board       = self.FIELD_EMPTY * np.ones((self.board_size, self.board_size), dtype=int)

    def get_state(self):
        """
        Get the current state of the game board as a unique integer representation.

        :return: An integer representing the game state.
        """
        # Convert from base-(1 + self.agent_count) number to decimal number
        base = 1 + self.agent_count
        return np.sum(self.board.flatten() * (base ** np.arange(self.board.size)))

    def make_move(self, agent_id, action):
        """
        Make a move on the game board.

        :param agent_id: The ID of the player making the move.
        :param action: The action representing the move.
        :return: A list of events that happened during the move (e.g., "INVALID_MOVE", "VICTORY").
        """

        # Validate agent id
        if agent_id <= self.FIELD_EMPTY or agent_id > self.agent_count:
            raise ValueError(f"Invalid agent id")
                             
        x, y = action % self.board_size, action // self.board_size

        events = []

        if self.is_field_free(x, y):
            self.board[y, x] = agent_id
        else:
            events.append("INVALID_MOVE")

        if self.did_agent_win(agent_id):
            events.append("VICTORY")

        if self.is_tic_tac_toe_full():
            events.append("DRAW")

        return events

    def did_agent_win(self, agent_id):
        """
        Check if a player (agent) with the given ID has won the game.

        :param agent_id: The ID of the player to check.
        :return: True if the player has won, False otherwise.
        """

        # Validate agent id
        if agent_id <= self.FIELD_EMPTY or agent_id > self.agent_count:
            raise ValueError(f"Invalid agent id")
        
        # Check rows and columns for a win
        for i in range(self.board_size):
            if np.all(self.board[i, :] == agent_id) or np.all(self.board[:, i] == agent_id):
                return True

        # Check diagonals for a win
        if np.all(np.diag(self.board) == agent_id) or np.all(np.diag(np.fliplr(self.board)) == agent_id):
            return True

        return False

    def is_tic_tac_toe_full(self):
        """
        Check if the game board is full.

        :return: True if the board is full, False otherwise.
        """
        return np.all(self.board != self.FIELD_EMPTY)

    def is_field_free(self, x, y):
        """
        Check if a specific field on the board is free.

        :param x: The x-coordinate of the field.
        :param y: The y-coordinate of the field.
        :return: True if the field is free, False otherwise.
        """
        ny, nx = self.board.shape
        if x < 0 or x >= nx or y < 0 or y >= ny:
            return False
        return self.board[y, x] == self.FIELD_EMPTY
    
class Agent:
    def __init__(self, agent_id, n_actions): 
        """
        Initialize an agent.

        :param agent_id:  The ID of the agent.
        :param n_actions: The number of available actions.
        """
        self.agent_id           = agent_id 
        self.n_actions          = n_actions
        self.is_training        = False 
        self.name               = f"agent {agent_id}"
        self.cumulative_reward  = 0

    def start_game(self, do_training):
        """
        Set whether agent is in training mode and reset cumulative awards

        :param do_training: Set training mode of agent.
        """
        self.is_training       = do_training
        self.cumulative_reward = 0 

    def act(self, state):
        """
        Select an action based on the current state.

        :param state: The current state.
        :return:      The selected action.
        """
        raise NotImplementedError()
    
    def update(self, state, action, next_state, reward):
        """
        Update the agent's Q-values based on the observed transition.

        :param state:      The current state.
        :param action:     The selected action.
        :param next_state: The next state.
        :param reward:     The observed reward.
        """
        self.cumulative_reward += reward


class RandomAgent(Agent): 
    def __init__(self, agent_id, n_actions): 
        """
        Initialize a random agent.

        :param agent_id:  The ID of the agent.
        :param n_actions: The number of available actions.
        """
        super().__init__(agent_id, n_actions) 
        self.name      = f"random agent {agent_id}"

    def act(self, state): 
        """
        Select a random action.

        :param state: The current state (not used).
        :return: A randomly selected action.
        """
        return np.random.randint(self.n_actions)

class TableQAgent(Agent): 
    def __init__(self, agent_id, n_actions, n_states, learning_rate=0.1, discount=0.9, exploration=0.1, learning_rate_decay=1-1e-5, exploration_decay=1-1e-5): 
        """
        Initialize a Q-learning agent with a Q-table.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param learning_rate:       The learning rate (alpha).
        :param discount:            The discount factor (gamma).
        :param exploration:         The exploration rate (epsilon).
        :param learning_rate_decay: The learning rate decay factor.
        :param exploration_decay:   The exploration rate decay factor.
        """
        super().__init__(agent_id, n_actions) 
        self.n_states            = n_states
        # Initialise Q-values with 0.5
        self.q                   = 0.5 * np.ones((n_states, n_actions))
        self.learning_rate       = learning_rate
        self.discount            = discount
        self.exploration         = exploration
        self.learning_rate_decay = learning_rate_decay
        self.exploration_decay   = exploration_decay
        self.name                = f"table-q agent {agent_id}"


    def act(self, state): 
        """
        Select an action using an epsilon-greedy policy.

        :param state: The current state.
        :return:      The selected action.
        """
        if np.random.uniform(0, 1) < self.exploration and self.is_training:
            # explore
            return np.random.randint(self.n_actions)  
        else:
            # exploit
            return np.argmax(self.q[state])
    
    def update(self, state, action, next_state, reward):
        """
        Update the Q-values based on the Q-learning update rule.

        :param state:      The current state.
        :param action:     The selected action.
        :param next_state: The next state.
        :param reward:     The observed reward.
        """
        super().update(state, action, next_state, reward)

        if not self.is_training:
            return 
        
        # Validate state and action
        if state < 0 or state >= self.n_states:
            raise ValueError(f"Invalid state value: state = {state} and n_states = {self.n_states}")
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action value: action = {action} and n_actions = {self.n_actions}")

        # Q-learning update rule
        self.q[state][action] += self.learning_rate * (reward + 
                                                      self.discount * np.max(self.q[next_state]) - self.q[state][action])
        
        self.exploration        *= self.exploration_decay
        self.learning_rate      *= self.learning_rate_decay




class GameManager:
    """
    A class responsible for managing the execution of a game between agents.

    Attributes:
        game (object): The game object representing the game state.
        agents (list): A list of agents participating in the game.
        gui (function, optional): A function for displaying the game state visually.

    Methods:
        run_game(): Run the game until completion, managing agent actions and updates.

    Static Methods:
        events_to_reward(events): Convert game events into rewards.
        is_game_over(events): Check if the game is over based on events.
    """

    
    def __init__(self, game, agents, gui=None):
        """
        Initialize the game manager.

        Args:
            game (object): The game object representing the game state.
            agents (list): A list of agents participating in the game.
            gui (function, optional): A function for displaying the game state visually.
        """
        self.game   = game
        self.agents = agents
        self.gui    = gui

    def run_game(self, do_training):
        """
        Run the game until completion, managing agent actions and updates.
        """

        self.game.start_game()

        for agent in self.agents:
            agent.start_game(do_training=do_training)

        while True:
            for agent in self.agents:
                state      = self.game.get_state()
                action     = agent.act(state)
                next_state = self.game.get_state()
                events     = self.game.make_move(agent.agent_id, action)
                reward     = self.events_to_reward(events)

                agent.update(state, action, next_state, reward)

                if self.gui is not None:
                    self.gui(self.game, agent.agent_id, events, self.is_game_over(events))


                if self.is_game_over(events):
                    return agent.agent_id, events

    @staticmethod
    def events_to_reward(events):
        """
        Convert game events into rewards.

        Args:
            events (list): A list of events that occurred during the game.

        Returns:
            float: The total reward calculated from the events.
        """
        rewards = {
            "INVALID_MOVE": -1.0,
            "DRAW":          0.5,
            "VICTORY":       1.0
        }
        return np.sum([rewards[event] for event in events])

    @staticmethod
    def is_game_over(events):
        """
        Check if the game is over based on events.

        Args:
            events (list): A list of events that occurred during the game.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return ("DRAW" in events) or ("VICTORY" in events)
            

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

def run_training(game, agents, board_size = 3, n_episode = 50000, gui = None): 
    """
    Run training episodes of a game with multiple agents and collect statistics.

    Args:
        game (object): The game object representing the game state.
        agents (list): A list of agent objects participating in the training.
        board_size (int, optional): Size of the game board. Default is 3.
        n_episodes (int, optional): Number of training episodes. Default is 50000.
        gui (function, optional): A function for displaying the game state visually.

    Returns:
        numpy.ndarray: An array of booleans indicating draws for each episode.
        numpy.ndarray: An array of booleans indicating if each agent won for each episode.
        numpy.ndarray: An array of cumulative rewards for each agent over episodes.
    """
    draws         = np.zeros(n_episode, dtype=bool)
    did_agent_win = np.zeros((len(agents), n_episode), dtype=bool)
    cum_rewards   = np.zeros((len(agents), n_episode))

    for episode in tqdm(range(n_episode)):
        game_manager     = GameManager(game = game, agents = agents, gui=gui)
        agent_id, events = game_manager.run_game(do_training=True) 

        draws[episode] = "DRAW" in events

        for i, agent in enumerate(agents): 
            agent_won        = ("VICTORY" in events) and (agent_id == agent.agent_id)
            agent_reward     = agent.cumulative_reward
            print(i, agent_won, agent_reward)
            did_agent_win[i] = agent_won
            cum_rewards[i]   = agent_reward

        
    return draws, did_agent_win, cum_rewards

def evaluate_training(agents, draws, did_agent_win, cum_rewards, batch_size = 500):

    mydir = os.path.join(
        os.getcwd(), 
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    
    moving_avg = lambda data, batch_size: np.convolve(data, np.ones(batch_size), mode='valid') / batch_size
    
    plt.title("Game outcomes")
    plt.plot(moving_avg(draws, batch_size), label="draws")
    for i, agent in enumerate(agents): 
        plt.plot(did_agent_win[i], label=f"{agent.name} wins")
    plt.legend()
    plt.savefig(mydir + "/outcomes.png")
    plt.close()

    plt.title("Cumulative rewards")
    for i, agent in enumerate(agents): 
        plt.plot(cum_rewards[i], batch_size, label=f"{agent.name} reward")
    plt.legend()
    plt.savefig(mydir + "/rewards.png")





agr1         = RandomAgent(agent_id=1, n_actions=9)
agr2         = RandomAgent(agent_id=2, n_actions=9)
agq1         = TableQAgent(agent_id=1, n_actions=9, n_states=3**9, learning_rate_decay=1-1e-5, exploration_decay=1-1e-5)
agq2         = TableQAgent(agent_id=2, n_actions=9, n_states=3**9)


agents = [agr1, agq2]
draws, did_agent_win, cum_rewards = run_training(game = TicTacToe(board_size = 3, agent_count = 2), agents = agents, n_episode=10, gui=gui)
evaluate_training(agents, draws, did_agent_win, cum_rewards)