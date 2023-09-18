# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import agent as agent
import numpy as np 

import tensorflow as tf
from tensorflow import keras
from collections import deque

class DenseQAgent(agent.Agent): 
    def __init__(self, agent_id, n_actions, n_states, learning_rate, discount, exploration, learning_rate_decay, exploration_decay, batch_size, deque_size): 
        """
        Initialize a Q-learning agent with a dense neural network

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
        self.learning_rate       = learning_rate
        self.discount            = discount
        self.exploration         = exploration
        self.learning_rate_decay = learning_rate_decay
        self.exploration_decay   = exploration_decay
        self.name                = f"deep-q agent {agent_id}"
        self.batch_size          = batch_size
        self.memory              = deque(maxlen=deque_size)


        # Define the Q-Network
        self.model = keras.Sequential([
            keras.layers.Dense(24, activation='relu', input_shape=(len(self.state2input(self.n_states -1)),)),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(n_actions, activation='linear')
        ])

        # Compile the model
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    def state2input(self, state): 
        binary_string = bin(state)[2:].zfill(len(bin(self.n_states - 1)[2:]))
        binary_array  = np.array([int(bit) for bit in binary_string])
        return binary_array

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
            s = self.state2input(state).reshape(1, -1)
            return np.argmax(self.model(s, training=False)) 
    
    def update(self, state, action, next_state, reward, done):
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

        # Store the experience in the replay memory
        self.memory.append((state, action, reward, next_state, done))

        # Sample a random minibatch from the replay memory
        if len(self.memory) >= self.batch_size:
            minibatch = np.random.sample(self.memory, self.batch_size)

            states, targets = [], []
            for s, a, r, s_, d in minibatch:
                target = self.model.predict(self.state2input(s))
                if d:
                    # If it's a terminal state, set the target to the immediate reward
                    target[0][a] = r  
                else:
                    # Update the target using the Bellman equation
                    # Note that Bellman equation here takes slightly different shape than for tabular_q_agent
                    # In tabular Q-learning, we compute delta Q and add it to Q(s). We therefore have an extra term where we subtract by lr * Q(s). 
                    # In Deep-Q-learning, the network represents Q(s). We compute the predicted Q-value given s and train the network using a loss function.
                    target[0][a] = r + self.discount * np.max(self.model.predict(self.state2input(s_), axis=1))

                states.append(s[0])
                targets.append(target[0])

            self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        # Decrease learning and exploration rates
        self.exploration   *= self.exploration_decay
        self.learning_rate *= self.learning_rate_decay

