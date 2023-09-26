
import os 
import random 
import numpy as np 
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint  # Import ModelCheckpoint callback
from tensorflow.keras import layers, models

from .agent import Agent
from .util  import decimal_to_base

# Define a directory to save checkpoints and logs
checkpoint_dir = 'checkpoints'
log_dir = 'logs'

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# View logs via tensorboard --logdir=./logs

# Define the simple DQN model 
def build_simple_dqn_model(input_shape, num_actions):
    """
    Build a simple DQN (Deep Q-Network) model.

    :param input_shape: The shape of the input state.
    :param num_actions: The number of available actions.
    :return: A Keras model for the simple DQN.
    """
    
    inputs  = tf.keras.layers.Input(shape=input_shape)
    layer1  = tf.keras.layers.Dense(256, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(num_actions, activation='linear')(layer1)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Define simple, convolutional model
def build_convolutional_dqn_model(input_shape, num_actions):
    """
    Build a simple DQN (Deep Q-Network) model using convolutional layers.

    :param input_shape: The shape of the input state.
    :param num_actions: The number of available actions.
    :return: A Keras model for the simple DQN.
    """

    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name='input_state')

    # Shared layers for both the value and advantage streams
    conv_layer1   = tf.keras.layers.Conv2D(128, (3, 3), data_format="channels_last", activation='relu', padding="SAME")(inputs)
    conv_layer2   = tf.keras.layers.Conv2D(128, (3, 3), data_format="channels_last", activation='relu', padding="SAME")(conv_layer1)
    conv_layer3   = tf.keras.layers.Conv2D(64, (3, 3),  data_format="channels_last", activation='relu', padding="SAME")(conv_layer2)
    flatten       = tf.keras.layers.Flatten()(conv_layer3) 
    
    layer1  = tf.keras.layers.Dense(256, activation='relu')(flatten)
    outputs = tf.keras.layers.Dense(num_actions, activation='linear')(layer1)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
    
# Define the Dueling DQN model
def build_dueling_dqn_model(input_shape, num_actions):
    """
    Build a simple DQN (Deep Q-Network) model.

    :param input_shape: The shape of the input state.
    :param num_actions: The number of available actions.
    :return: A Keras model for the simple DQN.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)


    # Shared layers for both the value and advantage streams
    dense = keras.layers.Dense(256, activation='relu')(inputs)

    # Value stream
    value_stream = keras.layers.Dense(32, activation='relu')(dense)
    value        = tf.keras.layers.Dense(1, activation='relu')(value_stream)

    # Advantage stream    
    advantage_stream = keras.layers.Dense(32, activation='relu')(dense)
    advantage        = tf.keras.layers.Dense(num_actions, activation='relu')(advantage_stream)

    # Combine value and advantage streams to get Q-values
    Q_values = value + (advantage - tf.math.reduce_mean(advantage, axis=1, keepdims=True))

    # Create the model
    model = models.Model(inputs=inputs, outputs=Q_values)
    return model

# Define the Dueling DQN model
def build_convolutional_dueling_dqn_model(input_shape, num_actions, reg_strength):
    """
    Build a Dueling DQN (Deep Q-Network) model with convolutional layers.

    :param input_shape: The shape of the input state.
    :param num_actions: The number of available actions.
    :return: A Keras model for the convolutional Dueling DQN.
    """

    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name='input_state')

    # Shared layers for both the value and advantage streams
    conv_layer1   = tf.keras.layers.Conv2D(128, (3, 3), data_format="channels_last", activation='relu', padding="SAME")(inputs)
    conv_layer2   = tf.keras.layers.Conv2D(128, (3, 3), data_format="channels_last", activation='relu', padding="SAME")(conv_layer1)
    conv_layer3   = tf.keras.layers.Conv2D(64, (3, 3),  data_format="channels_last", activation='relu', padding="SAME")(conv_layer2)
    flatten       = tf.keras.layers.Flatten()(conv_layer3) 
    shared_layer  = tf.keras.layers.Dense(256, 
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                          kernel_regularizer=tf.keras.regularizers.L1L2(l1=reg_strength, l2=reg_strength),
                                          activation='relu')(flatten) 


    # Value stream
    value_stream = keras.layers.Dense( 32, 
                                       kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=reg_strength, l2=reg_strength),
                                       activation='relu')(shared_layer)
    value = tf.keras.layers.Dense( 1, 
                                   kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=reg_strength, l2=reg_strength),
                                   activation='relu')(value_stream)

    # Advantage stream    
    advantage_stream = keras.layers.Dense( 32, 
                                           kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=reg_strength, l2=reg_strength),
                                           activation='relu')(shared_layer)
    advantage = tf.keras.layers.Dense( num_actions,
                                       kernel_initializer=tf.keras.initializers.VarianceScaling(),
                                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=reg_strength, l2=reg_strength),
                                       activation='relu')(advantage_stream)

    # Combine value and advantage streams to get Q-values
    Q_values = value + (advantage - tf.math.reduce_mean(advantage, axis=1, keepdims=True))

    # Create the model
    model = models.Model(inputs=inputs, outputs=Q_values)
    return model



class DeepQAgent(Agent): 
    MODE_BINARY   = 0 
    MODE_TUTORIAL = 1

    def __init__(self, agent_id, n_actions, n_states, config):         
        """
        Initialize a Deep Q-Network (DQN) agent for reinforcement learning.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing agent configuration parameters.
                                    - 'board_size': The size of the game board.
                                    - 'n_episode': The total number of episodes for training.
                                    - 'n_eval': The frequency of evaluation episodes.
                                    - 'eval_freq': The evaluation frequency in episodes.
                                    - 'discount': The discount factor (gamma).
                                    - 'learning_rate': The learning rate (alpha).
                                    - 'learning_rate_decay': The learning rate decay factor.
                                    - 'exploration': The exploration rate (epsilon).
                                    - 'exploration_decay': The exploration rate decay factor.
                                    - 'batch_size': The batch size for training.
                                    - 'replay_buffer_size': The size of the replay buffer.
                                    - 'target_update_freq': The frequency of updating the target network.
        """
        super().__init__(agent_id, n_actions) 
        self.n_states            = n_states

        self.name                = f"deep-q agent {agent_id}"
        self.board_size          = config["board_size"]    
        self.n_episode           = config["n_episode"]     
        self.n_eval              = config["n_eval"]        
        self.eval_freq           = config["eval_freq"]     
        self.discount            = config["discount"]      
        self.learning_rate       = config["learning_rate"] 
        self.learning_rate_decay = config["learning_rate_decay"]
        self.exploration         = config["exploration"]   
        self.exploration_decay   = config["exploration_decay"]
        self.batch_size          = config["batch_size"]
        self.replay_buffer_size  = config["replay_buffer_size"]
        self.n_eval              = config["n_eval"]
        self.target_update_freq  = config["target_update_freq"]
        self.update_counter      = 0
        self.episode             = 0
        self.debug               = False
        self.training_buffer     = []

        self.replay_buffer       = deque(maxlen=self.replay_buffer_size)
        self.input_mode          = self.MODE_TUTORIAL
        if self.input_mode == self.MODE_BINARY:
            self.input_shape     = (len(self.state_to_input(self.n_states -1)), )
        elif self.input_mode == self.MODE_TUTORIAL:
            self.input_shape     = (3 * n_actions, )
        else: 
            raise ValueError("Unsupported input mode")
        
        # Models need to be defined and compiled in derived classes 
        self.online_model        = None 
        self.target_model        = None 

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



    def update(self, iteration, state, legal_actions, action, reward, done):
        """
        Update the Q-values based on the Q-learning update rule.

        :param iteration:      The current iteration.
        :param state:          The current state.
        :param legal_actions:  List of legal actions.
        :param action:         The selected action.
        :param reward:         The observed reward.
        :param done:           Boolean indicating whether the episode is done.
        """
        super().update(iteration, state, legal_actions, action, reward, done)
        if self.is_training: 
            self.training_data.append([iteration, state, legal_actions, action, reward, done])
    

    def final_update(self, reward):
        """
        Update the training data after the final step of an episode.

        :param reward: The final observed reward.
        """
        super().final_update(reward)
        if self.is_training: 
            self.training_data[-1][self.DONE]    = True
            self.training_data[-1][self.REWARD] += reward
            
    def state_to_input(self, state): 
        """
        Convert the state into an input representation suitable for the neural network.

        :param state: The current state.
        :return:      The input representation of the state.
        """
        if self.input_mode == self.MODE_TUTORIAL: 
            n = 9 
            base_array     = decimal_to_base(state, base = 3, padding = n) 
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

        return tf.convert_to_tensor(representation.reshape(1, -1))
    
    def validate_training_data(self): 
        """
        Validate the integrity of training data, checking for missing iterations and incomplete episodes.
        """
        # Check integrity of training data 
        if self.training_data[-1][self.DONE] is not True: 
            raise ValueError("Last training datum not done")
        
        # Validate iteration number
        for i in range(len(self.training_data) - 1): 
            i1 = self.training_data[i  ][self.ITERATION]
            i2 = self.training_data[i+1][self.ITERATION]
            if (i1 + 1 != i2):
                raise ValueError(f"Missing iteration between iterations {i1} and {i2} in training data")
            
    def move_training_data_to_replay_buffer(self):
        """
        Move training data to the replay buffer, connecting states with their subsequent states.
        """
        self.validate_training_data() 

        # Connect state with next state and move to replay buffer
        for i in range(len(self.training_data)): 
            iteration, state, legal_actions, action, reward, done = self.training_data[i]
            if not done: 
                next_state = self.training_data[i+1][self.STATE]
            else: 
                next_state = 0
            self.replay_buffer.append([state, action, next_state, reward, done])

        self.is_training = []

    def minibatch_to_arrays(self, minibatch):
        """
        Convert a minibatch of experiences into arrays for training.

        :param minibatch: A minibatch of experiences.
        :return:          Tensors containing states, actions, next_states, rewards, and not_terminal flags.
        """
        states       = np.zeros((len(minibatch), *self.input_shape), dtype=np.float32)
        actions      = np.zeros(len(minibatch), dtype=np.int32)
        next_states  = np.zeros((len(minibatch), *self.input_shape), dtype=np.float32)
        rewards      = np.zeros(len(minibatch), dtype=np.float32)
        not_terminal = np.zeros(len(minibatch), dtype=np.float32)

        for i, (s, a, s_, r, d) in enumerate(minibatch):
            states      [i, :] = self.state_to_input(s)
            actions     [i]    = a 
            next_states [i, :] = self.state_to_input(s_)
            rewards     [i]    = r
            not_terminal[i]    = 1 - d

        return tf.convert_to_tensor(states), tf.convert_to_tensor(actions), tf.convert_to_tensor(next_states), tf.convert_to_tensor(rewards), tf.convert_to_tensor(not_terminal)

    def act(self, state, actions): 
        """
        Select an action using an epsilon-greedy policy.

        :param state:   The current state.
        :param actions: List of available actions.
        :return:        The selected action.
        """

        # explore
        if np.random.uniform(0, 1) < self.exploration and self.is_training:
            action = np.random.choice(actions)
        # exploit
        else:
            s = self.state_to_input(state)
            q = self.online_model(s, training=False)
            action = tf.argmax(q, axis=1) 
    
            if self.debug:
                print(f"Pick action {action} in state {state} with q-values {q[action], q}")

        # Decrease exploration rate
        self.exploration   *= self.exploration_decay

        return action 
    
    def train(self):
        """
        Train the agent's Q-network using experiences from the replay buffer.
        """

        if not self.is_training:
            return 

        self.move_training_data_to_replay_buffer() 

        # Sample a random minibatch from the replay replay_buffer
        if len(self.replay_buffer) >= self.batch_size:

            minibatch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, next_states, rewards, not_terminal = self.minibatch_to_arrays(minibatch)

            targets      = self.online_model.predict_on_batch(states)
            next_targets = self.target_model.predict_on_batch(next_states)
            targets[np.arange(len(targets)), actions]  = rewards + not_terminal * self.discount * np.max(next_targets, axis=1)

            if self.debug: 
                print(states, targets)

            history = self.online_model.fit(states, targets, epochs=1, verbose=0, callbacks=[self.checkpoint_callback, self.tensorboard])

            # Update the target network periodically
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_model.set_weights(self.online_model.get_weights())

            # Debugging: Print training progress
            if self.episode % self.n_eval == 0:
                print(f"Update: {self.episode}, Loss: {history.history['loss'][0]}")
                
        self.episode       += 1 

class SimpleDeepQAgent(DeepQAgent): 

    def __init__(self, agent_id, n_actions, n_states, config): 
        """
        A Q-learning agent with a simple dense neural network for Q-value approximation.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing configuration parameters.
        """
        super().__init__(agent_id, n_actions, n_states, config) 

        # Define the Q-Network
        self.online_model = build_simple_dqn_model(self.input_shape, self.n_actions)
        self.target_model = self.online_model 

        # Compile the model
        self.online_model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

class ConvolutionalDeepQAgent(DeepQAgent): 

    def __init__(self, agent_id, n_actions, n_states, config): 
        """
        A Q-learning agent with a simple dense neural network for Q-value approximation.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing configuration parameters.
        """
        super().__init__(agent_id, n_actions, n_states, config) 

        # Define the Q-Network
        self.online_model = build_convolutional_dqn_model(self.input_shape, self.n_actions)
        self.target_model = self.online_model 

        # Compile the model
        self.online_model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

class DualDeepQAgent(DeepQAgent): 

    def __init__(self, agent_id, n_actions, n_states, config): 
        """
        A Q-learning agent with a simple dense neural network for Q-value approximation.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing configuration parameters.
        """
        super().__init__(agent_id, n_actions, n_states, config) 

        # Define the Q-Network
        self.online_model = build_simple_dqn_model(self.input_shape, self.n_actions)
        self.target_model = build_simple_dqn_model(self.input_shape, self.n_actions)
        self.target_model.set_weights(self.online_model.get_weights())

        # Compile the model
        self.online_model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')


class DuellingDeepQAgent(DeepQAgent): 

    def __init__(self, agent_id, n_actions, n_states, config): 
        """
        A Q-learning agent with a dueling dense neural network architecture for Q-value approximation.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing configuration parameters.
        """
        super().__init__(agent_id, n_actions, n_states, config) 


        # Create the online Dueling DQN model
        self.online_model = build_dueling_dqn_model(self.input_shape, self.n_actions)

        # Create the target Dueling DQN model
        self.target_model = build_dueling_dqn_model(self.input_shape, self.n_actions)
        self.target_model.set_weights(self.online_model.get_weights())

        # Compile the online model
        self.online_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                                  loss='mean_squared_error')
        




class ConvDuellingDeepQAgent(DeepQAgent): 

    def __init__(self, agent_id, n_actions, n_states, config): 
        """
        A Q-learning agent with a convolutional neural network using a dueling architecture for Q-value approximation.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing configuration parameters.
        """
        
        super().__init__(agent_id, n_actions, n_states, config) 

        if self.input_mode != self.MODE_TUTORIAL:
            raise ValueError("Only input in MODE_TUTORIAL (one-hot) supported in convolutional network")
        
        self.input_shape = (3, 3, 3)

        reg_strength = 0.01 

        # Create the online Dueling DQN model
        self.online_model = build_convolutional_dueling_dqn_model(self.input_shape, self.n_actions, reg_strength=reg_strength)

        # Create the target Dueling DQN model
        self.target_model = build_convolutional_dueling_dqn_model(self.input_shape, self.n_actions, reg_strength=reg_strength)
        self.target_model.set_weights(self.online_model.get_weights())

        
        # Define a custom loss function that includes regularization
        def loss(y_true, y_pred):
            mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
            reg_loss = reg_strength * sum(self.online_model.losses)  # Sum of all regularization losses
            return mse_loss + reg_loss
    
        # Compile the online model
        self.online_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                                  loss=loss)
        

    def state_to_input(self, state):
        # Convert to NHWC (batch size, height, width, number of channels)
        input = super().state_to_input(state)
        input = tf.reshape(input, (1, 3, 3, 3))
        input = tf.transpose(input, [0,2,3,1])
        return input


class PrioritizedReplayBuffer:
    def __init__(self, maxlen, alpha=0.6, beta = 0.4, epsilon=1e-6):
        self.maxlen = maxlen
        self.alpha = alpha
        self.beta  = beta
        self.epsilon = epsilon
        self.tree = SumTree(maxlen)

    def add(self, experience, priority):
        self.tree.add(experience, priority)

    def sample(self, batch_size):
        batch = []
        indices = []
        priorities = []

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            (index, priority, experience) = self.tree.get(s)
            indices.append(index)
            priorities.append(priority)
            batch.append(experience)

        total_priority = np.max(priorities)
        weights = (len(self.tree) * np.array(priorities)) ** (-self.beta)
        weights /= max(weights)
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.tree.update(i, (priority + self.epsilon) ** self.alpha)

    def __len__(self):
        return len(self.tree)

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, data, priority):
        tree_index = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(tree_index, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                break

            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index

        data_index = parent_index - self.capacity + 1
        return parent_index, self.tree[parent_index], self.data[data_index]

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.n_entries


class PrioritisedDeepQAgent(DeepQAgent): 

    def __init__(self, agent_id, n_actions, n_states, config): 
        """
        A Q-learning agent with a simple dense neural network for Q-value approximation.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing configuration parameters.
        """
        super().__init__(agent_id, n_actions, n_states, config) 

        self.alpha = 0.6
        self.beta  = 0.4
        self.epsilon = 1e-6
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=self.replay_buffer_size, alpha=0.6, beta = 0.4, epsilon=1e-6)

    def move_training_data_to_replay_buffer(self):
        self.validate_training_data()

        for i in range(len(self.training_data)):
            iteration, state, legal_actions, action, reward, done = self.training_data[i]
            if not done:
                next_state = self.training_data[i + 1][self.STATE]
            else:
                next_state = 0

            # Compute the priority based on the TD error or loss
            current_q_values = self.online_model(self.state_to_input(state),      training=False)
            next_q_values    = self.target_model(self.state_to_input(next_state), training=False)

            target = reward + (1 - done) * self.discount * np.max(next_q_values, axis=1)

            # Compute mismatch between model prediction and value predicted by Bellmann equation
            td_error = tf.abs(target - current_q_values[0, int(action)])

            # Add small epsilon so that all samples have a chance of being revisited
            priority = (td_error + self.epsilon) ** self.alpha

            # Add the experience and priority to the prioritized replay buffer
            self.replay_buffer.add((state, action, next_state, reward, done), priority)

    def train(self):
        if not self.is_training:
            return
        
        self.move_training_data_to_replay_buffer() 

        if len(self.replay_buffer) >= self.batch_size:
            # Sample a minibatch from the prioritized replay buffer
            minibatch, batch_indices, weights = self.replay_buffer.sample(self.batch_size)

            states, actions, next_states, rewards, not_terminal = self.minibatch_to_arrays(minibatch)

            targets        = self.online_model.predict_on_batch(states)
            next_targets   = self.target_model.predict_on_batch(next_states)
            target_updates = rewards + not_terminal * self.discount * np.max(next_targets, axis=1)

            # Update priorities in the replay buffer based on the TD error
            td_errors = np.abs(targets[np.arange(self.batch_size), actions] - target_updates)
            
            priorities = (td_errors + self.epsilon) ** self.alpha
            self.replay_buffer.update_priorities(batch_indices, priorities)

            # Update targets
            targets[np.arange(len(targets)), actions] = target_updates

            # Perform the training step using the computed weights
            history = self.online_model.fit(states, targets, epochs=1, verbose=0, callbacks=[self.checkpoint_callback, self.tensorboard], sample_weight=weights)

            # Update the target network periodically
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_model.set_weights(self.online_model.get_weights())

            # Decrease learning and exploration rates
            self.exploration *= self.exploration_decay

        self.episode += 1


class PrioritisedSimpleDeepQAgent(PrioritisedDeepQAgent): 

    def __init__(self, agent_id, n_actions, n_states, config): 
        """
        A Q-learning agent with a simple dense neural network for Q-value approximation.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing configuration parameters.
        """
        super().__init__(agent_id, n_actions, n_states, config) 

        # Define the Q-Network
        self.online_model = build_simple_dqn_model(self.input_shape, self.n_actions)
        self.target_model = self.online_model 

        # Compile the model
        self.online_model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

class PrioritisedConvolutionalDeepQAgent(PrioritisedDeepQAgent): 

    def __init__(self, agent_id, n_actions, n_states, config): 
        """
        A Q-learning agent with a simple dense neural network for Q-value approximation.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing configuration parameters.
        """
        super().__init__(agent_id, n_actions, n_states, config) 

        # Define the Q-Network
        self.online_model = build_convolutional_dqn_model(self.input_shape, self.n_actions)
        self.target_model = self.online_model 

        # Compile the model
        self.online_model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
