
import numpy as np 

class GameManager:
    """
    A class responsible for managing the execution of a game between agents.

    Attributes:
        game (object): The game object representing the game state.
        agents (list): A list of agents participating in the game.
        gui (function, optional): A function for displaying the game state visually.

    Methods:
        run_game(): Run the game until completion, managing agent actions and updates.

    Static Methods:
        events_to_reward(events): Convert game events into rewards.
        is_game_over(events): Check if the game is over based on events.
    """

    
    def __init__(self, game, agents, gui=None):
        """
        Initialize the game manager.

        Args:
            game (object): The game object representing the game state.
            agents (list): A list of agents participating in the game.
            gui (function, optional): A function for displaying the game state visually.
        """
        self.game   = game
        self.agents = agents
        self.gui    = gui

    def run_game(self, do_training, randomise_order):
        """
        Run the game until completion, managing agent actions and updates.
        """

        self.game.start_game()

        for agent in self.agents:
            agent.start_game(do_training=do_training)

        # Randomise starting order in every game
        if randomise_order:
            agent_order = np.random.permutation(self.agents)
        else:
            agent_order = self.agents

        while True:
            for agent in agent_order:
                state      = self.game.get_state()
                action     = agent.act(state)
                next_state = self.game.get_state()
                events     = self.game.make_move(agent.agent_id, action)
                reward     = self.events_to_reward(events)
                done       = self.is_game_over(events)

                agent.update(state, action, next_state, reward, done)

                if self.gui is not None:
                    self.gui(self.game, agent.agent_id, events, done)

                if done:
                    return agent.agent_id, events

    @staticmethod
    def events_to_reward(events):
        """
        Convert game events into rewards.

        Args:
            events (list): A list of events that occurred during the game.

        Returns:
            float: The total reward calculated from the events.
        """
        rewards = {
            "INVALID_MOVE": -0.1,
            "DRAW":          0.5,
            "VICTORY":       1.0
        }
        return np.sum([rewards[event] for event in events])

    @staticmethod
    def is_game_over(events):
        """
        Check if the game is over based on events.

        Args:
            events (list): A list of events that occurred during the game.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return ("DRAW" in events) or ("VICTORY" in events)
            