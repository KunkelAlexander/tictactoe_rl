import numpy as np
from .util import decimal_to_base

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


        # Precompute the base‐powers for get_state
        base = 1 + agent_count
        self._powers = (base ** np.arange(board_size * board_size)).astype(np.int64)


        # indices of all winning lines (flattened index space)
        n = board_size
        rows    = [np.arange(i*n, i*n + n) for i in range(n)]
        cols    = [np.arange(i, n*n, n) for i in range(n)]
        diag1   = [np.arange(0, n*n, n+1)]
        diag2   = [np.arange(n-1, n*n-1, n-1)]
        self._lines = np.stack(rows + cols + diag1 + diag2)  # shape: (2n+2, n)

    def copy(self) -> 'TicTacToe':
        """
        Create a deep copy of the TicTacToe game.

        :return: A new TicTacToe instance with the same state.
        """
        copied_game = TicTacToe(board_size=self.board_size, agent_count=self.agent_count)
        copied_game.board = self.board.copy()
        return copied_game

    def start_game(self, start_state=0):
        """
        Initialize the TicTacToe board.
        """
        self.board = self.FIELD_EMPTY * np.ones((self.board_size, self.board_size), dtype=int)

        self.set_state(start_state)

    def get_state(self) -> int:
        """
        Get the current state of the game board as a unique integer representation.

        :return: An integer representing the game state.
        """
        flat = self.board.ravel()
        # a single dot‐product, no exponents every call
        return int(flat.dot(self._powers))

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

        board = self.board if board is None else board
        flat  = board.ravel()
        # compare once
        matches = (flat[self._lines] == agent_id)
        # a win if any line is all True
        return np.any(matches.all(axis=1))

    def is_tic_tac_toe_full(self) -> bool:
        """
        Check if the game board is full.

        :return: True if the board is full, False otherwise.
        """
        return np.all(self.board != self.FIELD_EMPTY)

    def is_valid_state(self, starting_agent_id: int, perspective_agent_id: int) -> bool:
        """
        Determine if the current board state is valid given who started the game and
        from whose perspective we are evaluating the validity.

        :param starting_agent_id: The agent who started the game (1 or 2).
        :param perspective_agent_id: The agent from whose perspective we are evaluating.
        :return: True if the state is valid, False otherwise.
        """
        if starting_agent_id not in [1, 2] or perspective_agent_id not in [1, 2]:
            raise ValueError("Only agent IDs 1 and 2 are supported.")

        flat = self.board.ravel()

        # Validate board values
        if np.any((flat < 0) | (flat > self.agent_count)):
            return False

        # Count number of moves made by each agent
        moves = {1: np.count_nonzero(flat == 1), 2: np.count_nonzero(flat == 2)}
        total_moves = moves[1] + moves[2]
        move_diff = moves[1] - moves[2]

        # Check valid move count difference based on starting agent
        if starting_agent_id == 1:
            if move_diff not in [0, 1]:
                return False
            current_turn = 1 if move_diff == 0 else 2
        else:
            if move_diff not in [-1, 0]:
                return False
            current_turn = 2 if move_diff == 0 else 1

        # If game is over (victory or draw), no further moves may be played
        state_eval = self.evaluate_game_state(perspective_agent_id)
        if state_eval in ["VICTORY", "DEFEAT", "DRAW"]:
            return False

        # Otherwise, it's valid only if it's this agent's turn
        return current_turn == perspective_agent_id

