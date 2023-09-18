# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import agent as agent
import numpy as np 


class TabularQAgent(agent.Agent): 
    def __init__(self, agent_id, n_actions, n_states, learning_rate, discount, exploration, learning_rate_decay, exploration_decay): 
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
        # explore
        if np.random.uniform(0, 1) < self.exploration and self.is_training:
            return np.random.randint(self.n_actions)  
        # exploit
        else:
            return np.argmax(self.q[state])
    
    def update(self, state, action, next_state, reward, done):
        """
        Update the Q-values based on the Q-learning update rule.

        :param state:      The current state.
        :param action:     The selected action.
        :param next_state: The next state.
        :param reward:     The observed reward.
        """
        super().update(state, action, next_state, reward, done)

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
        
        # Decrease learning and exploration rates
        self.exploration   *= self.exploration_decay
        self.learning_rate *= self.learning_rate_decay

