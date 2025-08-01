import numpy as np
from tqdm import tqdm

from .agent import Agent
from .util  import decimal_to_base

class MinMaxAgent(Agent):

    outcomes = {
        "VICTORY": 1,
        "DRAW": 0,
        "DEFEAT": -1,
        "ONGOING": 0,
    }

    def __init__(self, agent_id: int, n_actions: int, n_states: int, game, act_randomly: bool):
        """
        Initialize a MinMax agent.

        :param agent_id: The ID of the agent.
        :param n_actions: The number of available actions.
        :param n_states: The number of states in the environment.
        :param game: The game object representing the game state.
        :param act_randomly: Whether the agent should act randomly.
        """
        super().__init__(agent_id, n_actions)
        self.n_states = n_states
        self.best_moves = {}

        if act_randomly:
            self.name = f"random MinMax agent {agent_id}"
        else:
            self.name = f"MinMax agent {agent_id}"

        self.maximising_cache = {}
        self.minimising_cache = {}
        self.act_randomly = act_randomly

        print(f"Initializing {self.name}")
        # Iterate over all possible states
        for state in tqdm(range(n_states)):
            game.set_state(state)
            best_value, best_actions = self.minimax(game, depth=9, maximising_player=True)
            self.best_moves[state] = best_actions.flatten()


        # Return the Q-table constructed from the minmax values
        self.q = np.full((n_states, n_actions), -np.inf)

        for state in range(n_states):
            game.set_state(state)
            legal_actions = game.get_actions(only_legal_actions=True)
            for action in legal_actions:
                game_copy = game.copy()
                game_copy.make_move(self.agent_id, action)
                value, _ = self.minimax(game_copy, depth=8, maximising_player=False)
                self.q[state][action] = value


    def get_q(self):
        return self.q

    def minimax(self, game, depth, maximising_player):
        state = game.get_state()
        if maximising_player:
            if state in self.maximising_cache:
                return self.maximising_cache[state]
        else:
            if state in self.minimising_cache:
                return self.minimising_cache[state]

        # Calculate the heuristic value of the game state (e.g., +1 for a win, -1 for a loss, 0 for a draw)
        best_value = None
        best_actions = None

        game_state = game.evaluate_game_state(self.agent_id)

        # Check if the game is over
        if "ONGOING" not in game_state:
            best_value = self.outcomes[game_state]
            best_actions = np.array([-1])
            self.maximising_cache[state] = (best_value, best_actions)
            self.minimising_cache[state] = (best_value, best_actions)

        # Act for maximising player
        else:
            if maximising_player:
                id = self.agent_id
            else:
                id = 3 - self.agent_id  # Assuming 2 players

            actions = game.get_actions(only_legal_actions=True)
            values = np.zeros(actions.shape)
            for i, action in enumerate(actions):
                game_copy = game.copy()  # Make a copy of the game state
                game_copy.make_move(id, action)
                value, _ = self.minimax(game_copy, depth - 1, maximising_player=not maximising_player)
                values[i] = value

            if maximising_player:
                best_value = np.max(values)
                best_actions = actions[np.argwhere(values == best_value)]
                self.maximising_cache[state] = (best_value, best_actions)
            else:
                best_value = np.min(values)
                best_actions = actions[np.argwhere(values == best_value)]
                self.minimising_cache[state] = (best_value, best_actions)

        return best_value, best_actions

    def act(self, state, actions):
        """
        Select an action using an epsilon-greedy policy.

        :param state: The current state.
        :return: The selected action.
        """
        if self.act_randomly:
            return np.random.choice(self.best_moves[state])
        else:
            return self.best_moves[state][0]
