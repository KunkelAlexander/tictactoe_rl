import numpy as np
from collections import deque

class GameManager:
    """
    A class responsible for managing the execution of a game between agents.

    Attributes:
        game (object): The game object representing the game state.
        agents (list): A list of agents participating in the game.
        gui (function, optional): A function for displaying the game state visually.

    Methods:
        run_game(do_training, randomise_order, only_legal_actions, debug=False):
            Run the game until completion, managing agent actions and updates.

    Static Methods:
        events_to_reward(events): Convert game events into rewards.
        is_game_over(events): Check if the game is over based on events.
    """

    MAX_ITERATIONS = int(1e5)

    def __init__(self, game, agents, gui=None):
        """
        Initialize the game manager.

        Args:
            game (object): The game object representing the game state.
            agents (list): A list of agents participating in the game.
            gui (function, optional): A function for displaying the game state visually.
        """
        self.game = game
        self.agents = agents
        self.gui = gui

    def run_game(self, do_training: bool, randomise_order: bool, only_legal_actions: bool, debug: bool = False):
        """
        Run the game until completion, managing agent actions and updates.

        Args:
            do_training (bool): Whether agents should perform training.
            randomise_order (bool): Whether to randomize the agent order in each game.
            only_legal_actions (bool): Whether to restrict agents to legal actions.
            debug (bool): Whether to enable debug mode.
        """
        self.game.start_game()

        for agent in self.agents:
            agent.start_game(do_training=do_training)
            agent.training_data = []

        # Randomize starting order in every game
        if randomise_order:
            agent_order = np.random.permutation(self.agents)
        else:
            agent_order = self.agents

        done = False

        for iteration in range(self.MAX_ITERATIONS):
            for agent in agent_order:
                state = self.game.get_state()
                legal_actions = self.game.get_actions(only_legal_actions=only_legal_actions)
                action = agent.act(state, legal_actions)
                events = self.game.make_move(agent.agent_id, action)
                game_events = [self.game.evaluate_game_state(agent.agent_id)]
                reward = self.events_to_reward(events)
                done = self.is_game_over(game_events)
                agent.update(iteration, state, legal_actions, action, reward, done)
                if self.gui is not None:
                    self.gui(self.game, agent.agent_id, events, done)

                if done:
                    break
            if done:
                break

        for agent in agent_order:
            game_events = [self.game.evaluate_game_state(agent.agent_id)]
            reward = self.events_to_reward(game_events)
            agent.final_update(reward)
            agent.train()

        return [(agent.agent_id, self.game.evaluate_game_state(agent.agent_id), agent.cumulative_reward) for agent in self.agents]

    @staticmethod
    def events_to_reward(events: list) -> float:
        """
        Convert game events into rewards.

        Args:
            events (list): A list of events that occurred during the game.

        Returns:
            float: The total reward calculated from the events.
        """
        rewards = {
            "INVALID_MOVE": -0.1,
            "ONGOING": 0.0,
            "DRAW": 0.5,
            "VICTORY": 1.0,
            "DEFEAT":  0.0
        }
        return np.sum([rewards[event] for event in events])

    @staticmethod
    def is_game_over(events: list) -> bool:
        """
        Check if the game is over based on events.

        Args:
            events (list): A list of events that occurred during the game.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return not ("ONGOING" in events)
