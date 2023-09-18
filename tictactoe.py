import numpy as np 

class TicTacToe:
    FIELD_EMPTY = 0

    def __init__(self, board_size, agent_count):
        """
        Initialize the game

        :param board_size: Size of the board (e.g., 3 for a 3x3 board).
        :param agent_count: Number of players (agents) in the game.
        """
        self.board_size  = board_size
        self.agent_count = agent_count

    def start_game(self): 
        """
        Initialize the TicTacToe board
        """
        self.board       = self.FIELD_EMPTY * np.ones((self.board_size, self.board_size), dtype=int)

    def get_state(self):
        """
        Get the current state of the game board as a unique integer representation.

        :return: An integer representing the game state.
        """
        # Convert from base-(1 + self.agent_count) number to decimal number
        base = 1 + self.agent_count
        return np.sum(self.board.flatten() * (base ** np.arange(self.board.size)))

    def make_move(self, agent_id, action):
        """
        Make a move on the game board.

        :param agent_id: The ID of the player making the move.
        :param action: The action representing the move.
        :return: A list of events that happened during the move (e.g., "INVALID_MOVE", "VICTORY").
        """

        # Validate agent id
        if agent_id <= self.FIELD_EMPTY or agent_id > self.agent_count:
            raise ValueError(f"Invalid agent id")
        
        # Convert 1D action to 2D index on board                              
        x, y = action % self.board_size, action // self.board_size

        events = []

        if self.is_field_free(x, y):
            self.board[y, x] = agent_id
        else:
            events.append("INVALID_MOVE")
            return events

        if self.did_agent_win(agent_id):
            events.append("VICTORY")
            return events 

        if self.is_tic_tac_toe_full():
            events.append("DRAW")
            return events
        
        return events

    def did_agent_win(self, agent_id):
        """
        Check if a player (agent) with the given ID has won the game.

        :param agent_id: The ID of the player to check.
        :return: True if the player has won, False otherwise.
        """

        # Validate agent id
        if agent_id <= self.FIELD_EMPTY or agent_id > self.agent_count:
            raise ValueError(f"Invalid agent id")
        
        # Check rows and columns for a win
        for i in range(self.board_size):
            if np.all(self.board[i, :] == agent_id) or np.all(self.board[:, i] == agent_id):
                return True

        # Check diagonals for a win
        if np.all(np.diag(self.board) == agent_id) or np.all(np.diag(np.fliplr(self.board)) == agent_id):
            return True

        return False

    def is_tic_tac_toe_full(self):
        """
        Check if the game board is full.

        :return: True if the board is full, False otherwise.
        """
        return np.all(self.board != self.FIELD_EMPTY)

    def is_field_free(self, x, y):
        """
        Check if a specific field on the board is free.

        :param x: The x-coordinate of the field.
        :param y: The y-coordinate of the field.
        :return: True if the field is free, False otherwise.
        """
        ny, nx = self.board.shape

        if x < 0 or x >= nx or y < 0 or y >= ny:
            raise ValueError("Invalid position on board")
        
        return self.board[y, x] == self.FIELD_EMPTY
    