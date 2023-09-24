import numpy as np
from util import decimal_to_base

class TicTacToe:
    FIELD_EMPTY = 0

    def __init__(self, board_size: int, agent_count: int):
        """
        Initialize the TicTacToe game.

        :param board_size: Size of the board (e.g., 3 for a 3x3 board).
        :param agent_count: Number of players (agents) in the game.
        """
        self.board_size = board_size
        self.agent_count = agent_count

    def copy(self) -> 'TicTacToe':
        """
        Create a deep copy of the TicTacToe game.

        :return: A new TicTacToe instance with the same state.
        """
        copied_game = TicTacToe(board_size=self.board_size, agent_count=self.agent_count)
        copied_game.board = self.board.copy()
        return copied_game

    def start_game(self):
        """
        Initialize the TicTacToe board.
        """
        self.board = self.FIELD_EMPTY * np.ones((self.board_size, self.board_size), dtype=int)

    def get_state(self) -> int:
        """
        Get the current state of the game board as a unique integer representation.

        :return: An integer representing the game state.
        """
        base = 1 + self.agent_count
        return np.sum(self.board.flatten() * (base ** np.arange(self.board.size)), dtype=int)

    def set_state(self, state: int):
        """
        Set the current state of the game board via unique integer representation.

        :param state: Integer representation of the board.
        """
        self.board = decimal_to_base(state, base=1 + self.agent_count, padding=self.board_size**2).reshape(
            self.board_size, self.board_size
        )

    def make_move(self, agent_id: int, action: int) -> list:
        """
        Make a move on the game board.

        :param agent_id: The ID of the player making the move.
        :param action: The action representing the move.
        :return: A list of events that happened during the move (e.g., "INVALID_MOVE", "VICTORY").
        """
        events = []

        legal_actions = self.get_actions(only_legal_actions=True)

        if action in legal_actions:
            self.board[action // self.board_size, action % self.board_size] = agent_id
        else:
            events.append("INVALID_MOVE")

        return events

    def evaluate_game_state(self, agent_id: int) -> str:
        """
        Check the state of the game (victory, defeat, draw, or ongoing).

        :param agent_id: The ID of the player.
        :return: The game state ("VICTORY", "DEFEAT", "DRAW", or "ONGOING").
        """
        for id in range(1, self.agent_count + 1):
            if self.did_agent_win(id):
                if agent_id == id:
                    return "VICTORY"
                else:
                    return "DEFEAT"

        if self.is_tic_tac_toe_full():
            return "DRAW"

        return "ONGOING"

    def get_actions(self, only_legal_actions: bool) -> np.ndarray:
        """
        Get a list of available actions on the board.

        :param only_legal_actions: If True, return only legal (empty) actions.
        :return: Numpy array with indices of available actions.
        """
        if only_legal_actions:
            actions = np.argwhere(self.board.flatten() == self.FIELD_EMPTY).flatten()
        else:
            actions = np.arange(self.board_size**2)
        return actions

    def did_agent_win(self, agent_id: int, board: np.ndarray = None) -> bool:
        """
        Check if a player (agent) with the given ID has won the game.

        :param agent_id: The ID of the player to check.
        :param board: Optional game board for checking (useful for testing).
        :return: True if the player has won, False otherwise.
        """
        if board is None:
            board = self.board

        # Validate agent id
        if agent_id <= self.FIELD_EMPTY or agent_id > self.agent_count:
            raise ValueError(f"Invalid agent ID: {agent_id}")

        # Check rows and columns for a win
        for i in range(self.board_size):
            if np.all(board[i, :] == agent_id) or np.all(board[:, i] == agent_id):
                return True

        # Check diagonals for a win
        if np.all(np.diag(board) == agent_id) or np.all(np.diag(np.fliplr(board)) == agent_id):
            return True

        return False

    def is_tic_tac_toe_full(self) -> bool:
        """
        Check if the game board is full.

        :return: True if the board is full, False otherwise.
        """
        return np.all(self.board != self.FIELD_EMPTY)