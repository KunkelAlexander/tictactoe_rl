import sys
sys.path.insert(1, '../')

import agent as agent
import numpy as np

class TabularQAgent(agent.Agent):
    def __init__(self, agent_id: int, n_actions: int, n_states: int, config: dict):
        """
        Initialize a Q-learning agent with a Q-table.

        :param agent_id: The ID of the agent.
        :param n_actions: The number of available actions.
        :param n_states: The number of states in the environment.
        :param config: A dictionary containing agent configuration parameters.
        """
        super().__init__(agent_id, n_actions)
        self.n_states = n_states
        self.initial_q = config["initial_q"]
        self.q = self.initial_q * np.ones((n_states, n_actions))
        self.discount = config["discount"]
        self.learning_rate = config["learning_rate"]
        self.learning_rate_decay = config["learning_rate_decay"]
        self.exploration = config["exploration"]
        self.exploration_decay = config["exploration_decay"]
        self.debug = config["debug"]
        self.name = f"table-q agent {agent_id}"
        self.training_data = []

    def start_game(self, do_training: bool):
        """
        Set whether the agent is in training mode and reset cumulative rewards.

        :param do_training: Set training mode of agent.
        """
        super().start_game(do_training)
        self.training_data = []

    def act(self, state, actions):
        """
        Select an action using an epsilon-greedy policy.

        :param state: The current state.
        :param actions: List of available actions.
        :return: The selected action.
        """
        # Explore
        if np.random.uniform(0, 1) < self.exploration and self.is_training:
            action = np.random.choice(actions)
        # Exploit
        else:
            # Disable q-values of illegal actions
            illegal_actions = np.setdiff1d(np.arange(self.n_actions), actions)
            self.q[state, illegal_actions] = -np.inf
            best_actions = np.argwhere(self.q[state] == np.max(self.q[state]))
            action = np.random.choice(best_actions.flatten())

        if self.debug:
            print(f"Pick action {action} in state {state} with q-values {self.q[state]}")
        return action

    def update(self, iteration, state, legal_actions, action, reward, done):
        """
        Update the Q-values based on the Q-learning update rule.

        :param iteration: The current iteration number.
        :param state: The current state.
        :param legal_actions: List of legal actions.
        :param action: The selected action.
        :param reward: The observed reward.
        :param done: True if the episode is done, False otherwise.
        """
        super().update(iteration, state, legal_actions, action, reward, done)
        if self.is_training:
            self.training_data.append([iteration, state, legal_actions, action, reward, done])

    def final_update(self, reward : float):
        """
        Update the Q-values at the end of an episode.

        :param reward: The observed reward.
        """
        super().final_update(reward)
        if self.is_training:
            self.training_data[-1][self.DONE] = True
            self.training_data[-1][self.REWARD] += reward

    def train(self):
        """
        Train the agent by updating Q-values based on the collected training data.
        """
        if not self.is_training:
            return

        next_max = None

        # Check integrity of training data
        if self.training_data[-1][self.DONE] is not True:
            raise ValueError("Last training datum not done")

        next_iteration = 0
        next_state = 0

        for i in range(len(self.training_data) - 1):
            # Validate iteration number
            i1 = self.training_data[i][self.ITERATION]
            i2 = self.training_data[i+1][self.ITERATION]
            if (i1 + 1 != i2):
                raise ValueError(f"Missing iteration between iterations {i1} and {i2} in training data")

        for iteration, state, legal_actions, action, reward, done in reversed(self.training_data):
            if self.debug:
                print(f"Iter {iteration}: q-value in state {state} before update: {self.q[state]} with reward {reward} and Game Over = {done}")

            # Q-learning update rule
            if done:
                self.q[state][action] = reward
            else:
                self.q[state][action] += self.learning_rate * (reward + self.discount * next_max - self.q[state][action])

            next_max = np.max(self.q[state])

            if self.debug:
                print(f"q-value in state {state} after update: {self.q[state]}")

        # Decrease learning and exploration rates
        self.exploration *= self.exploration_decay
        self.learning_rate *= self.learning_rate_decay
