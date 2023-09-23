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
        self.q                   = 0.6 * np.ones((n_states, n_actions))
        self.learning_rate       = learning_rate
        self.discount            = discount
        self.exploration         = exploration
        self.learning_rate_decay = learning_rate_decay
        self.exploration_decay   = exploration_decay
        self.name                = f"table-q agent {agent_id}"
        self.training_data       = []
        self.debug               = False

    def start_game(self, do_training):
        """
        Set whether agent is in training mode and reset cumulative awards

        :param do_training: Set training mode of agent.
        """
        super().start_game(do_training) 
        self.training_data = []

    def act(self, state, actions): 
        """
        Select an action using an epsilon-greedy policy.

        :param state: The current state.
        :return:      The selected action.
        """
        # explore
        if np.random.uniform(0, 1) < self.exploration and self.is_training:
            action = np.random.choice(actions)
        # exploit
        else:
            # disable q-values of illegal actions
            illegal_actions = np.setdiff1d(np.arange(self.n_actions), actions) 
            self.q[state, illegal_actions] = -np.inf
            best_actions    = np.argwhere(self.q[state] == np.max(self.q[state]))
            action          = np.random.choice(best_actions.flatten())
    
        if self.debug:
            print(f"Pick action {action} in state {state} with q-values {self.q[state]}")
        return action 
    
    def update(self, iteration, state, legal_actions, action, reward, done):
        """
        Update the Q-values based on the Q-learning update rule.

        :param state:         The current state.
        :param legal_actions: List of legal actions. 
        :param action:        The selected action.
        :param next_state:    The next state.
        :param reward:        The observed reward.
        """
        super().update(iteration, state, legal_actions, action, reward, done)
        if self.is_training: 
            self.training_data.append([iteration, state, legal_actions, action, reward, done])
    

    def final_update(self, reward):
        """
        Update the Q-values based on the Q-learning update rule.

        :param state:      The current state.
        :param action:     The selected action.
        :param next_state: The next state.
        :param reward:     The observed reward.
        """
        super().final_update(reward)
        if self.is_training: 
            self.training_data[-1][self.DONE]    = True
            self.training_data[-1][self.REWARD] += reward

    def train(self):

        if not self.is_training: 
            return 
        
        next_max = None 

        # Check integrity of training data 
        if self.training_data[-1][self.DONE] is not True: 
                raise ValueError("Last training datum not done")
        
        next_iteration = 0 
        next_state     = 0
        
        for i in range(len(self.training_data) - 1): 
            # Validate iteration number
            i1 = self.training_data[i  ][self.ITERATION]
            i2 = self.training_data[i+1][self.ITERATION]
            if (i1 + 1 != i2):
                raise ValueError(f"Missing iteration between iterations {i1} and {i2} in training data")
            

        for iteration, state, legal_actions, action, reward, done in reversed(self.training_data):
            if self.debug: 
                print(f"Iter {iteration}: q-value in state {state} before update: {self.q[state]} with reward {reward} and Game Over = {done}")
            
            # Q-learning update rule
            if done: 
                self.q[state][action]  = reward
            else: 
                self.q[state][action] += self.learning_rate * (reward + self.discount * next_max - self.q[state][action])
            
            next_max = np.max(self.q[state]) 

            if self.debug: 
                print(f"q-value in state {state} after update: {self.q[state]}")

        # Decrease learning and exploration rates
        self.exploration   *= self.exploration_decay
        self.learning_rate *= self.learning_rate_decay
