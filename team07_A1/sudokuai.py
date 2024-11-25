import copy
import random
import time

import competitive_sudoku.sudokuai
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def possible(game_state: GameState, i: int, j: int, value: int) -> bool:
        """Check if it is possible to place `value` on position (`i`, `j`) in the current game state"""

        # check if taboo move
        if TabooMove((i, j), value) in game_state.taboo_moves:
            return False

        # check if cell is empty
        if game_state.board.get((i, j)) != SudokuBoard.empty:
            return False

        # check horizontal line
        for idx in range(game_state.board.N):
            if game_state.board.get(square=(i, idx)) == value:
                return False

        # check vertical line
        for idx in range(game_state.board.N):
            if game_state.board.get(square=(idx, j)) == value:
                return False

        # check region
        region_i = i // game_state.board.m
        region_j = j // game_state.board.n

        for i_in in range(game_state.board.m):
            for j_in in range(game_state.board.n):
                ind_i = (region_i * game_state.board.m) + i_in
                ind_j = (region_j * game_state.board.n) + j_in

                if game_state.board.get(square=(ind_i, ind_j)) == value:
                    return False

        return True
    
    @staticmethod
    def evaluate_game(game_state: GameState) -> int:
        """
        Compute the score of the game (basically `player_1_score` - `player_2_score`)
        """
        return game_state.scores[0] - game_state.scores[1]
    

    @staticmethod
    def calculate_score(i: int, j: int, value: int, game_state: GameState):
        """Calculate the score of a move"""
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

    @staticmethod
    def minimax(game_state: GameState, alpha: int = -100000, beta: float = 100000, depth: int = 3):
        """
        Minimax tree search for the current game state up to a given `depth`.
        This method also implements alpha/beta pruning for the search optimization.
        """
        # terminate recursion if reached max depth
        if depth == 0:
            return SudokuAI.evaluate_game(game_state), None

        # extract possible moves
        player_squares = game_state.player_squares()

        # If all moves are available - enumerate them
        if player_squares is None:
            player_squares = [(i, j) for i in range(game_state.board.N) for j in range(game_state.board.N)]

        # Apply additional filtering of possible moves (e.g. check possible values, taboo, cell emptiness)
        possible_moves = [
            Move((i, j), value)
            for i, j in player_squares
            for value in range(1, game_state.board.N + 1)
            if SudokuAI.possible(game_state, i, j, value)
        ]

        # terminate recursion if there are no possible moves
        if len(possible_moves) == 0:
            return SudokuAI.evaluate_game(game_state), None

        # recursive step
        if game_state.current_player == 1:
            # maximizing case
            value_minimax = float("-inf")
            best_move = None

            for move in possible_moves:
                # put new value, change player, add square to occupied squares
                game_state.board.put(move.square, move.value)
                game_state.occupied_squares1.append(move.square)
                move_score = SudokuAI.calculate_score(move.square[0], move.square[1], move.value, game_state)
                game_state.scores[game_state.current_player - 1] += move_score
                game_state.current_player = 2

                # recurse with a new game state
                next_value_minimax, _ = SudokuAI.minimax(game_state=game_state, alpha=alpha, beta=beta, depth=depth - 1)

                # undo the move in the game (avoiding copying game state)
                game_state.board.put(move.square, game_state.board.empty)
                game_state.occupied_squares1.pop()
                game_state.current_player = 1
                game_state.scores[game_state.current_player - 1] -= move_score

                # update best value and best move if needed
                if next_value_minimax > value_minimax:
                    value_minimax = next_value_minimax
                    best_move = move

                # alpha/beta pruning logic
                if value_minimax >= beta:
                    break
                alpha = max(alpha, value_minimax)

        elif game_state.current_player == 2:
            # minimizing case
            value_minimax = float("inf")
            best_move = None

            for move in possible_moves:
                # put new value, change player, add square to occupied squares
                game_state.board.put(move.square, move.value)
                game_state.occupied_squares2.append(move.square)
                move_score = SudokuAI.calculate_score(move.square[0], move.square[1], move.value, game_state)
                game_state.scores[game_state.current_player - 1] += move_score
                game_state.current_player = 1

                # recurse with a new game state
                next_value_minimax, _ = SudokuAI.minimax(game_state=game_state, alpha=alpha, beta=beta, depth=depth - 1)

                # undo the move in the game (avoiding copying game state)
                game_state.board.put(move.square, game_state.board.empty)
                game_state.occupied_squares2.pop()
                game_state.current_player = 2
                game_state.scores[game_state.current_player - 1] -= move_score

                # update best value and best move if needed
                if next_value_minimax < value_minimax:
                    value_minimax = next_value_minimax
                    best_move = move

                # alpha/beta pruning logic
                if value_minimax <= alpha:
                    break
                beta = min(beta, value_minimax)

        return value_minimax, best_move


    def compute_best_move(self, game_state: GameState) -> None:
        # initialize first move with a random move
        player_squares = game_state.player_squares()
        if player_squares is None:
            player_squares = [(i, j) for i in range(game_state.board.N) for j in range(game_state.board.N)]
        possible_moves = [
            Move((i, j), value)
            for i, j in player_squares
            for value in range(1, game_state.board.N + 1)
            if SudokuAI.possible(game_state, i, j, value)
        ]
        move = random.choice(possible_moves)
        self.propose_move(move)

        # iterative deepening: choose N**2+1 to be sure to loop over every depth
        N = game_state.board.N
        for i in range(N**2 + 1):
            _, move = SudokuAI.minimax(game_state, depth=i)
            if move is not None:
                self.propose_move(move)

