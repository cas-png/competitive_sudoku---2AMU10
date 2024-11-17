#  (C) Copyright Wieger Wesselink 2021. Distributed under the GPL-3.0-or-later
#  Software License, (See accompanying file LICENSE or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

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

    # N.B. This is a very naive implementation.
    def compute_best_move(self, game_state: GameState) -> None:
        N = game_state.board.N

        # Check whether a cell is empty, a value in that cell is not taboo, and that cell is allowed
        def possible(i, j, value):
            return game_state.board.get((i, j)) == SudokuBoard.empty \
                   and not TabooMove((i, j), value) in game_state.taboo_moves \
                       and (i, j) in game_state.player_squares()
        
        # Calculate the score of a move
        def score(i, j, value, game_state: GameState):
            # Put the move on the board
            game_state.board.put((i, j), value)

            N = game_state.board.N
            n = game_state.board.n      # Number of rows
            m = game_state.board.m      # Number of columns
            reg_completed = 0

            # Check the move completes a column
            full_col = True
            for k in range(N):
                if game_state.board.get((k, j)) == SudokuBoard.empty:
                    full_col = False
            if full_col:
                reg_completed += 1

            # Check the move completes a row
            full_row = True
            for l in range(N):
                if game_state.board.get((i, l)) == SudokuBoard.empty:
                    full_row = False
            if full_row:
                reg_completed += 1

            # Check the move completes a rectangle
            full_rectangle = True
            # Get coordinates for the left upper corner in the n by m rectangle where (i,j) is located
            i_left_top = i - (i % n)
            j_left_top = j - (j % m)
            for k in range(n):
                for l in range(m):
                    if game_state.board.get((i_left_top + k, j_left_top + l)) == SudokuBoard.empty:
                        full_rectangle = False
            if full_rectangle:
                reg_completed += 1
            
            scores = [0, 1, 3, 7]

            return scores[reg_completed]

        all_moves = [Move((i, j), value) for i in range(N) for j in range(N)
                     for value in range(1, N+1) if possible(i, j, value)]
        move = random.choice(all_moves)
        self.propose_move(move)
        while True:
            time.sleep(0.2)
            self.propose_move(random.choice(all_moves))

