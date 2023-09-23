# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import agent as agent
import numpy as np 

import tensorflow as tf
from tensorflow import keras
from collections import deque
import random 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint  # Import ModelCheckpoint callback
import os 
from util import decimal_to_base

# Define a directory to save checkpoints and logs
checkpoint_dir = 'checkpoints'
log_dir = 'logs'

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# View logs via tensorboard --logdir=./logs

class DeepQAgent(agent.Agent): 
    MODE_BINARY   = 0 
    MODE_TUTORIAL = 1

    def __init__(self, agent_id, n_actions, n_states, learning_rate, discount, exploration, learning_rate_decay, exploration_decay, batch_size, replay_buffer_size, n_eval): 
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
        self.replay_buffer       = deque(maxlen=replay_buffer_size)
        self.input_mode          = self.MODE_TUTORIAL
        if self.input_mode == self.MODE_BINARY:
            self.input_shape    = len(self.state_to_input(self.n_states -1))
        elif self.input_mode == self.MODE_TUTORIAL:
            self.input_shape     = 3 * n_actions 
        else: 
            raise ValueError("Unsupported input mode")
        
        # Define the Q-Network
        self.model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(self.input_shape,)),
            keras.layers.Dense(n_actions, activation='linear')
        ])

        # Compile the model
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

        # Create a ModelCheckpoint callback to save model checkpoints during training
        self.checkpoint_callback = ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model_weights_{epoch:02d}.h5'), 
            save_weights_only=True, 
            save_best_only=False, 
            save_freq=10  # Save every 10 epochs
        )

        # Create a TensorBoard callback for logging
        self.tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )

        # Training episode
        self.episode = 0
        self.debug   = False
        self.training_buffer = []
        self.n_eval = n_eval 


    def update(self, iteration, state, legal_actions, action, reward, done):
        """
        Update the Q-values based on the Q-learning update rule.

        :param state:      The current state.
        :param action:     The selected action.
        :param next_state: The next state.
        :param reward:     The observed reward.
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
            
    def state_to_input(self, state): 

        if self.input_mode == self.MODE_TUTORIAL: 
            n = 9 
            base_array = decimal_to_base(state, base = 3, padding = n) 
            representation = np.zeros(3 * n) 
            for i, b in enumerate(base_array):
                if b == 0:
                    representation[i]         = 1
                elif b == 1: 
                    representation[i + n]     = 1
                elif b == 2: 
                    representation[i + 2 * n] = 1
        elif self.input_mode == self.MODE_BINARY:
            representation = decimal_to_base(state, base = 2, padding = 15) 
        else:
            raise ValueError("Unsupported input mode")

        return representation

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
            illegal_actions    = np.setdiff1d(np.arange(self.n_actions), actions) 
            #best_actions      = np.argwhere(self.q[state] == np.max(self.q[state]))
            #action            = np.random.choice(best_actions.flatten())
            s                  = self.state_to_input(state).reshape(1, -1)
            q                  = self.model(s, training=False).numpy().flatten()
            #q[illegal_actions] = -np.inf
            action = np.argmax(q) 
    
            if self.debug:
                print(f"Pick action {action} in state {state} with q-values {q[action], q}")

        return action 
    
    def train(self):
        """
        Update the Q-values based on the Q-learning update rule.

        :param state:      The current state.
        :param action:     The selected action.
        :param next_state: The next state.
        :param reward:     The observed reward.
        """

        if not self.is_training:
            return 

        # Check integrity of training data 
        if self.training_data[-1][self.DONE] is not True: 
            raise ValueError("Last training datum not done")
        
        # Validate iteration number
        for i in range(len(self.training_data) - 1): 
            i1 = self.training_data[i  ][self.ITERATION]
            i2 = self.training_data[i+1][self.ITERATION]
            if (i1 + 1 != i2):
                raise ValueError(f"Missing iteration between iterations {i1} and {i2} in training data")
            
        # Connect state with next state and move to replay buffer
        for i in range(len(self.training_data)): 
            iteration, state, legal_actions, action, reward, done = self.training_data[i]
            if not done: 
                next_state = self.training_data[i+1][self.STATE]
            else: 
                next_state = 0
            self.replay_buffer.append([state, action, next_state, reward, done])

        self.is_training = []

        # Sample a random minibatch from the replay replay_buffer
        if len(self.replay_buffer) >= self.batch_size:

            mini_batch = random.sample(self.replay_buffer, self.batch_size)
            
            states       = np.zeros((len(mini_batch), self.input_shape), dtype=float)
            next_states  = np.zeros((len(mini_batch), self.input_shape), dtype=float)
            actions      = np.zeros(len(mini_batch), dtype=int)
            rewards      = np.zeros(len(mini_batch), dtype=float)
            not_terminal = np.zeros(len(mini_batch), dtype=int)
 
            for i, (s, a, s_, r, d) in enumerate(mini_batch):
                states      [i, :] = self.state_to_input(s).reshape(1, -1)
                actions     [i]    = a 
                next_states [i, :] = self.state_to_input(s_).reshape(1, -1)
                rewards     [i]    = r
                not_terminal[i]    = 1 - d



            # Update the target using the Bellman equation
            # If it's a terminal state, set the target to the immediate reward
            # Note that Bellman equation here takes slightly different shape than for tabular_q_agent
            # In tabular Q-learning, we compute delta Q and add it to Q(s). 
            # In order to compute the difference delta Q, we subtract by lr * Q(s). 
            # In Deep-Q-learning, the network represents Q(s). We compute the target Q-value given the Bellman equation
            # and train the network to mimise the mismatch between its current prediction and the Bellmann prediction
            # The difference betweeen the Q values is computed in the loss function.
            targets                                    = self.model.predict_on_batch(states)
            targets[np.arange(len(targets)), actions]  = rewards + not_terminal * self.discount * np.max(self.model.predict_on_batch(next_states), axis=1)

            if self.debug: 
                print(states, targets)
            history = self.model.fit(states, targets, epochs=1, verbose=0, callbacks=[self.checkpoint_callback, self.tensorboard])

            # Debugging: Print training progress
            if self.episode % self.n_eval == 0:
                print(f"Update: {self.episode}, Loss: {history.history['loss'][0]}")
                
        # Decrease learning and exploration rates
        self.exploration   *= self.exploration_decay
        self.episode       += 1 

