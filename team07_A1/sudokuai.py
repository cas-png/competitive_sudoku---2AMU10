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
    def possible(game_state, i, j, value):
        """check if move is possible in the current state of the game"""

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
    def evaluate_game(game_state: GameState):
        """
        calculate score for the game (player_1_score - player_2_score)
        player score is the number of row/columns/blocks closed
        """

        # check rows
        closed_rows_1 = 0
        closed_rows_2 = 0
        for i in range(game_state.board.N):
            cnt_1 = 0
            cnt_2 = 0

            for j in range(game_state.board.N):
                if (i, j) in game_state.occupied_squares1:
                    cnt_1 += 1
                elif (i, j) in game_state.occupied_squares2:
                    cnt_2 += 1

            if cnt_1 == game_state.board.N:
                closed_rows_1 += 1
            elif cnt_2 == game_state.board.N:
                closed_rows_2 += 1

        # check cols
        closed_cols_1 = 0
        closed_cols_2 = 0
        for i in range(game_state.board.N):
            cnt_1 = 0
            cnt_2 = 0
            for j in range(game_state.board.N):
                if (j, i) in game_state.occupied_squares1:
                    cnt_1 += 1
                elif (j, i) in game_state.occupied_squares2:
                    cnt_2 += 1

            if cnt_1 == game_state.board.N:
                closed_cols_1 += 1
            elif cnt_2 == game_state.board.N:
                closed_cols_2 += 1

        # check blocks
        closed_blocks_1 = 0
        closed_blocks_2 = 0
        for b_i in range(game_state.board.N // game_state.board.m):
            for b_j in range(game_state.board.N // game_state.board.n):

                cnt_1 = 0
                cnt_2 = 0

                for i in range(game_state.board.m):
                    for j in range(game_state.board.n):

                        if (
                            b_i * game_state.board.m + i,
                            b_j * game_state.board.n + j,
                        ) in game_state.occupied_squares1:
                            cnt_1 += 1
                        elif (
                            b_i * game_state.board.m + i,
                            b_j * game_state.board.n + j,
                        ) in game_state.occupied_squares2:
                            cnt_2 += 1

                if cnt_1 == game_state.board.N:
                    closed_blocks_1 += 1
                if cnt_2 == game_state.board.N:
                    closed_blocks_2 += 1

        player_1_score = closed_rows_1 + closed_cols_1 + closed_blocks_1
        player_2_score = closed_rows_2 + closed_cols_2 + closed_blocks_2

        return player_1_score - player_2_score

    @staticmethod
    def minimax(game_state, depth):
        """
        minimax tree search algorithm up to a given depth
        """
        if depth == 0:
            return SudokuAI.evaluate_game(game_state), None

        # extract possible moves
        player_squares = game_state.player_squares()

        if player_squares is None:
            player_squares = [(i, j) for i in range(game_state.board.N) for j in range(game_state.board.N)]

        possible_moves = [
            Move((i, j), value)
            for i, j in player_squares
            for value in range(1, game_state.board.N + 1)
            if SudokuAI.possible(game_state, i, j, value)
        ]


        if len(possible_moves) == 0:
            # terminate recursion if there are no possible moves
            return SudokuAI.evaluate_game(game_state), None


        if game_state.current_player == 1:
            # maximizing case
            value = float("-inf")
            best_move = None

            for move in possible_moves:
                new_state = copy.deepcopy(game_state)

                # change board, current player, occupied squares
                new_state.board.put(move.square, move.value)
                new_state.occupied_squares1.append(move.square)
                new_state.current_player = 2

                next_value, _ = SudokuAI.minimax(new_state, depth - 1)
                if next_value > value:
                    value = next_value
                    best_move = move

            return value, best_move

        elif game_state.current_player == 2:
            # minimizing case
            value = float("inf")
            best_move = None

            for move in possible_moves:
                new_state = copy.deepcopy(game_state)

                # change board, current player, occupied squares
                new_state.board.put(move.square, move.value)
                new_state.occupied_squares2.append(move.square)
                new_state.current_player = 1

                next_value, _ = SudokuAI.minimax(new_state, depth - 1)
                if next_value < value:
                    value = next_value
                    best_move = move

            return value, best_move

    @staticmethod
    def select_move(game_state: GameState) -> Move:
        """Select MINIMAX move if found and RANDOM otherwise"""
        val, move = SudokuAI.minimax(game_state, depth=3)

        if move is not None:
            # minimax found move
            print("*" * 100)
            print(f"Selected MINIMAX move with score {val} | square = {move.square}; value = {move.value};")
            print("*" * 100)
            return move

        # if minimax couldn't find a move, just take random
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
        print("*" * 100)
        print(f"Selected RANDOM move | square = {move.square}; value = {move.value};")
        print("*" * 100)
        return move

    def compute_best_move(self, game_state: GameState) -> None:

        self.propose_move(self.select_move(game_state))

        while True:
            time.sleep(0.2)
            self.propose_move(self.select_move(game_state))
