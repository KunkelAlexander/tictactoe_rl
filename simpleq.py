import numpy as np
import random

# Define the environment (example: a simple gridworld)
# In this example, there are 6 states (S0 to S5) and 2 actions (A0 and A1).
num_states = 6
num_actions = 2

# Define the reward structure (R[state][action])
# This is a simple gridworld where the agent's goal is to reach the terminal state (S5).
# The agent receives a reward of +1 for reaching S5 and a reward of -1 for transitioning to S4 from any state.
R = np.array([
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [-1, -1],
    [0, 0],  # Terminal state with rewards for both actions
])

# Initialize the Q-table with zeros
Q = np.zeros((num_states, num_actions))

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Q-learning algorithm
for episode in range(num_episodes):
    state = 0  # Start in the initial state (S0)

    while state != num_states - 1:  # Continue until reaching the terminal state
        # Epsilon-greedy policy for action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions - 1)  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit

        # Take the selected action and observe the next state and reward
        next_state = state + 1
        reward = R[state][action]

        # Update the Q-value using the Q-learning update rule
        Q[state][action] += learning_rate * (reward + discount_factor * np.max(Q[next_state]) - Q[state][action])

        # Move to the next state
        state = next_state

# Print the learned Q-table
print("Learned Q-table:")
print(Q)
