def ascii_gui(tictactoe, agent_id, events, game_over):
    """
    Display the current game state in a text-based user interface.

    Args:
        tictactoe (TicTacToe): The TicTacToe game object.
        agent_id (int):        The ID of the current agent making a move.
        events (list):         A list of events that occurred during the move (e.g., "INVALID_MOVE", "VICTORY").
        game_over (bool):      True if the game is over, False otherwise.

    Displays:
        The current game board with player symbols ('X', 'O') and game events.
    """
    print(f"Agent {agent_id}; Events: {events}; Game over? {game_over}")

    ny, nx = tictactoe.board.shape
    shapes = {
        0: " ", 1: "X", 2: "O", 3: "△", 4: "▢",
    }
    for i in range(ny):
        print((1 + nx * 4) * "-")
        for j in range(nx):
            print(f"| {shapes[tictactoe.board[i, j]]} ", end="")
        print("|")
    print((1 + nx * 4) * "-")
