# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')

import .agent as agent
import numpy as np 

class RandomAgent(agent.Agent): 
    def __init__(self, agent_id, n_actions): 
        """
        Initialize a random agent.

        :param agent_id:  The ID of the agent.
        :param n_actions: The number of available actions.
        """
        super().__init__(agent_id, n_actions) 
        self.name      = f"random agent {agent_id}"

    def act(self, state, actions): 
        """
        Select a random action.

        :param state:   The current state (not used).
        :param actions: List of possible actions. 
        :return: A randomly selected action.
        """
        return np.random.choice(actions)
