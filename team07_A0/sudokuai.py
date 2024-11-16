import random
import time
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove
import competitive_sudoku.sudokuai


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    # Improved implementation that ensures only legal moves are made.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        # Check if a move is valid according to Sudoku rules
        def is_valid_move(i, j, value):
            # Cell must be empty
            if game_state.board.get((i, j)) != SudokuBoard.empty:
                return False

            # Move must not be taboo
            if TabooMove((i, j), value) in game_state.taboo_moves:
                return False

            # Move must be allowed for the player
            if (i, j) not in game_state.player_squares():
                return False

            # Sudoku rule: 'value' must not already exist in the same row, column, or grid
            for k in range(N):
                # Check row and column
                if game_state.board.get((i, k)) == value or game_state.board.get((k, j)) == value:
                    return False

            # Check the 3x3 subgrid (assuming standard 9x9 Sudoku)
            subgrid_size = int(N**0.5)
            start_row = (i // subgrid_size) * subgrid_size
            start_col = (j // subgrid_size) * subgrid_size

            for x in range(start_row, start_row + subgrid_size):
                for y in range(start_col, start_col + subgrid_size):
                    if game_state.board.get((x, y)) == value:
                        return False

            return True

        # Generate all valid moves
        all_moves = [
            Move((i, j), value)
            for i in range(N)
            for j in range(N)
            for value in range(1, N + 1)
            if is_valid_move(i, j, value)
        ]

        # If there are no valid moves, do nothing
        if not all_moves:
            print("No valid moves available.")
            return

        # Select a random valid move and propose it
        move = random.choice(all_moves)
        self.propose_move(move)

        # Keep proposing valid moves continuously
        while True:
            time.sleep(0.2)

            # Regenerate the list of valid moves to account for state changes
            all_moves = [
                Move((i, j), value)
                for i in range(N)
                for j in range(N)
                for value in range(1, N + 1)
                if is_valid_move(i, j, value)
            ]

            # Break if no moves are left (this might indicate game completion)
            if not all_moves:
                print("No valid moves remaining.")
                break

            # Propose another valid move
            self.propose_move(random.choice(all_moves))
