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

# Define a directory to save checkpoints and logs
checkpoint_dir = 'checkpoints'
log_dir = 'logs'

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# View logs via tensorboard --logdir=./logs

class DenseQAgent(agent.Agent): 
    MODE_BINARY   = 0 
    MODE_TUTORIAL = 1

    def __init__(self, agent_id, n_actions, n_states, learning_rate, discount, exploration, learning_rate_decay, exploration_decay, batch_size, replay_buffer_size): 
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
            self.input_shape    = len(self.state2input(self.n_states -1))
        elif self.input_mode == self.MODE_TUTORIAL:
            self.input_shape     = 3 * n_actions 
        else: 
            raise ValueError("Unsupported input mode")
        
        # Define the Q-Network
        self.model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(self.input_shape,)),
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


    def decimal_to_base(self, decimal_num, base = 3, padding = 9):    
        base_num = ""
        
        if decimal_num == 0:
            # Special case for 0 in base
            base_num = "0"
        else: 
            while decimal_num > 0:
                remainder = decimal_num % base
                base_num  = str(remainder) + base_num
                decimal_num //= base

        base_string = base_num.zfill(padding)[::-1]
        base_array  = np.array([int(digit) for digit in base_string])

        return base_array

    
    def state2input(self, state): 

        if self.input_mode == self.MODE_TUTORIAL: 
            n = 9 
            base_array = self.decimal_to_base(state, base = 3, padding = n) 
            representation = np.zeros(3 * n) 
            for i, b in enumerate(base_array):
                if b == 0:
                    representation[i]         = 1
                elif b == 1: 
                    representation[i + n]     = 1
                elif b == 2: 
                    representation[i + 2 * n] = 1
        elif self.input_mode == self.MODE_BINARY:
            representation = self.decimal_to_base(state, base = 2, padding = 15) 
        else:
            raise ValueError("Unsupported input mode")

        return representation

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
            #prediction = self.model(s, training=False)
            #print("State: ", s, "Prediction: ", prediction)
            return np.argmax(self.model(s, training=False)) 
    
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

        # Store the experience in the replay replay_buffer
        self.replay_buffer.append((self.state2input(state).reshape(1, -1), action, reward, self.state2input(next_state).reshape(1, -1), done))

        # Sample a random minibatch from the replay replay_buffer
        if len(self.replay_buffer) >= self.batch_size:
            #print("Hi")

            mini_batch = random.sample(self.replay_buffer, self.batch_size)
            
            states       = np.zeros((len(mini_batch), self.input_shape), dtype=float)
            next_states  = np.zeros((len(mini_batch), self.input_shape), dtype=float)
            actions      = np.zeros(len(mini_batch), dtype=int)
            rewards      = np.zeros(len(mini_batch), dtype=float)
            not_terminal = np.zeros(len(mini_batch), dtype=int)
 
            for i, (s, a, r, s_, d) in enumerate(mini_batch):
                states[i, :]      = s
                actions[i]        = a 
                rewards[i]        = r
                next_states[i, :] = s_
                not_terminal[i]   = 1 - d


            # Update the target using the Bellman equation
            # If it's a terminal state, set the target to the immediate reward
            # Note that Bellman equation here takes slightly different shape than for tabular_q_agent
            # In tabular Q-learning, we compute delta Q and add it to Q(s). 
            # In order to compute the difference delta Q, we subtract by lr * Q(s). 
            # In Deep-Q-learning, the network represents Q(s). We compute the target Q-value given the Bellman equation
            # and train the network to mimise the mismatch between its current prediction and the Bellmann prediction
            # The difference betweeen the Q values is computed in the loss function.
            targets                                    = self.model.predict_on_batch(states)
            targets[np.arange(len(targets)), actions]  = rewards + not_terminal * self.discount * np.amax(self.model.predict_on_batch(next_states), axis=1)

            history = self.model.fit(states, targets, epochs=1, verbose=0, callbacks=[self.checkpoint_callback, self.tensorboard])

            # Debugging: Print training progress
            if self.episode % 100 == 0:
                print(f"Update: {self.episode}, Loss: {history.history['loss'][0]}")
                
        # Decrease learning and exploration rates
        self.exploration   *= self.exploration_decay
        self.episode       += 1 

