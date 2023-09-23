class Agent:
    ITERATION = 0 
    STATE     = 1 
    ACTION    = 2
    REWARD    = 3 
    DONE      = 4 

    def __init__(self, agent_id, n_actions): 
        """
        Initialize an agent.

        :param agent_id:  The ID of the agent.
        :param n_actions: The number of available actions.
        """
        self.agent_id           = agent_id 
        self.n_actions          = n_actions
        self.is_training        = False 
        self.name               = f"agent {agent_id}"
        self.cumulative_reward  = 0 

    def start_game(self, do_training):
        """
        Set whether agent is in training mode and reset cumulative awards

        :param do_training: Set training mode of agent.
        """
        self.is_training       = do_training
        self.cumulative_reward = 0 

    def act(self, state, actions):
        """
        Select an action based on the current state.

        :param state: The current state.
        :return:      The selected action.
        """
        raise NotImplementedError()
    
    def update(self, iteration, state, action, reward, done):
        """
        Update the agent's training data based on the observed transition.

        :param iteration:  The current iteration number. 
        :param state:      The current state.
        :param action:     The selected action.
        :param next_state: The next state.
        :param reward:     The observed reward.
        """
        self.cumulative_reward += reward 
        pass


    def final_update(self, reward):
        """
        Update the agent's training data based on the observed transition.

        :param iteration:  The current iteration number. 
        :param state:      The current state.
        :param action:     The selected action.
        :param next_state: The next state.
        :param reward:     The observed reward.
        """
        self.cumulative_reward += reward 
        pass

    def train(self):
        """
        Update the agent's Q-values based on the training data.
        """
        pass