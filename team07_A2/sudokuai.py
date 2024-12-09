from typing import Optional, Set, Tuple

import competitive_sudoku.sudokuai
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_possible_value(game_state: GameState, i: int, j: int) -> Optional[int]:
        """Returns possible value for the position (i, j) of the game and None if it does not exist."""
        taboo_moves = {v.value for v in game_state.taboo_moves if v.square == (i, j)}

        vertical = set()
        for idx in range(game_state.board.N):
            val = game_state.board.get(square=(i, idx))
            if val != game_state.board.empty:
                vertical.add(val)

        horizontal = set()
        for idx in range(game_state.board.N):
            val = game_state.board.get(square=(idx, j))
            if val != game_state.board.empty:
                horizontal.add(val)

        region = set()
        region_i = i // game_state.board.m
        region_j = j // game_state.board.n
        for i_in in range(game_state.board.m):
            for j_in in range(game_state.board.n):
                ind_i = (region_i * game_state.board.m) + i_in
                ind_j = (region_j * game_state.board.n) + j_in
                val = game_state.board.get(square=(ind_i, ind_j))
                if val != game_state.board.empty:
                    region.add(val)

        options = set(range(1, game_state.board.N + 1)).difference(vertical, horizontal, region, taboo_moves)

        return next(iter(options)) if len(options) > 0 else None

    @staticmethod
    def compute_possible_regions(game_state: GameState) -> Tuple[int, int]:
        init_player = game_state.current_player

        game_state.current_player = 1
        p1_possible = game_state.player_squares()

        game_state.current_player = 2
        p2_possible = game_state.player_squares()

        game_state.current_player = init_player

        p1_total = 0
        p2_total = 0

        for i in range(game_state.board.N):
            p1_count = 0
            p2_count = 0
            missing = 0
            for j in range(game_state.board.N):
                if game_state.board.get(square=(i, j)) == game_state.board.empty:
                    missing += 1
                    if (i, j) in p1_possible:
                        p1_count += 1
                    if (i, j) in p2_possible:
                        p2_count += 1

            # if p1_count == missing:
            #     p1_total += 1
            # if p2_count == missing:
            #     p2_total += 1

            if missing == 0:
                continue

            p1_total += (p1_count / missing) ** 2
            p2_total += (p2_count / missing) ** 2

        for i in range(game_state.board.N):
            p1_count = 0
            p2_count = 0
            missing = 0
            for j in range(game_state.board.N):
                if game_state.board.get(square=(j, i)) == game_state.board.empty:
                    missing += 1
                    if (j, i) in p1_possible:
                        p1_count += 1
                    if (j, i) in p2_possible:
                        p2_count += 1

            # if p1_count == missing:
            #     p1_total += 1
            # if p2_count == missing:
            #     p2_total += 1

            if missing == 0:
                continue

            p1_total += (p1_count / missing) ** 2
            p2_total += (p2_count / missing) ** 2

        for reg_i in range(game_state.board.N // game_state.board.m):
            for reg_j in range(game_state.board.N // game_state.board.n):
                p1_count = 0
                p2_count = 0
                missing = 0
                for i in range(game_state.board.m):
                    for j in range(game_state.board.n):
                        cell = (game_state.board.m * reg_i + i, game_state.board.n * reg_j + j)
                        if game_state.board.get(square=cell) == game_state.board.empty:
                            missing += 1
                            if cell in p1_possible:
                                p1_count += 1
                            if cell in p2_possible:
                                p2_count += 1

                # if p1_count == missing:
                #     p1_total += 1
                # if p2_count == missing:
                #     p2_total += 1

                if missing == 0:
                    continue

                p1_total += (p1_count / missing) ** 2
                p2_total += (p2_count / missing) ** 2

        return p1_total, p2_total

    @staticmethod
    def compute_avg_depths(game_state: GameState) -> Tuple[float, float]:
        """Compute the average depth of moves for each player"""
        if game_state.occupied_squares1 and game_state.occupied_squares2:
            p1_avg = sum(t[0] for t in game_state.occupied_squares1) / len(game_state.occupied_squares1)
            p2_avg = sum(t[0] for t in game_state.occupied_squares2) / len(game_state.occupied_squares2)
            return p1_avg + 1, game_state.board.N - p2_avg
        return 0, 0

    @staticmethod
    def evaluate_game(game_state: GameState) -> float:
        """Compute the heuristic value for the score of the game"""

        # Score difference
        weight = 1.0
        score_difference = weight * (game_state.scores[0] - game_state.scores[1])

        # Compute difference in number of regions each player can possibly complete
        weight = 3.0
        player1_possible, player2_possible = SudokuAI.compute_possible_regions(game_state)
        possible_regions_difference = weight * (player1_possible - player2_possible)

        # Compute difference in average depth for each player
        weight = 1.0
        player1_max_depth, player2_max_depth = SudokuAI.compute_avg_depths(game_state)
        max_depth_difference = weight * (player1_max_depth - player2_max_depth)

        return score_difference + possible_regions_difference + max_depth_difference

    @staticmethod
    def calculate_score(game_state: GameState, move: Move) -> int:
        """Calculate the score of a single move in the current state of the game"""
        reg_completed = 0

        # check column
        full_col = True
        for k in range(game_state.board.N):
            if game_state.board.get((k, move.square[1])) == SudokuBoard.empty:
                full_col = False
        if full_col:
            reg_completed += 1

        # check row
        full_row = True
        for l in range(game_state.board.N):
            if game_state.board.get((move.square[0], l)) == SudokuBoard.empty:
                full_row = False
        if full_row:
            reg_completed += 1

        # check block
        full_rectangle = True
        i_left_top = move.square[0] - (move.square[0] % game_state.board.n)
        j_left_top = move.square[1] - (move.square[1] % game_state.board.m)

        for k in range(game_state.board.n):
            for l in range(game_state.board.m):
                if game_state.board.get((i_left_top + k, j_left_top + l)) == SudokuBoard.empty:
                    full_rectangle = False
        if full_rectangle:
            reg_completed += 1

        scores = [0, 1, 3, 7]
        return scores[reg_completed]

    @staticmethod
    def neighbors(square: Tuple[int, int], N: int) -> Set[Tuple[int, int]]:
        """Get neighbors of a square (including the square itself)"""
        nbh = set()
        row, col = square
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if 0 <= r < N and 0 <= c < N:
                    nbh.add((r, c))

        return nbh

    @staticmethod
    def get_playing_region(game_state: GameState) -> Set[Tuple[int, int]]:
        """Get neighbors of all occupied squares by both players"""
        full_nbh = set()
        for move in game_state.occupied_squares1 + game_state.occupied_squares2:
            full_nbh.update(SudokuAI.neighbors(move, game_state.board.N))
        return full_nbh

    @staticmethod
    def minimax(game_state: GameState, alpha: int = -100, beta: int = 100, depth: int = 5):
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

        # Apply additional filtering of possible moves (e.g. check possible values, taboo, cell emptiness, ...)
        possible_moves = list()

        intersecting = SudokuAI.get_playing_region(game_state).intersection(set(player_squares))
        if len(intersecting) == 0:
            intersecting = player_squares

        for (i, j) in intersecting:
            val = SudokuAI.get_possible_value(game_state, i, j)
            if val is not None:
                possible_moves.append(Move(square=(i, j), value=val))

        # terminate recursion if there are no possible moves
        if len(possible_moves) == 0:
            return SudokuAI.evaluate_game(game_state), None

        # recursive step
        if game_state.current_player == 1:
            # maximizing case
            value = float("-inf")
            best_move = None

            for move in possible_moves:
                # put new value, add square to occupied squares, change scores, change player
                game_state.board.put(move.square, move.value)
                game_state.occupied_squares1.append(move.square)
                move_score = SudokuAI.calculate_score(game_state, move)
                game_state.scores[game_state.current_player - 1] += move_score
                game_state.current_player = 2

                # recurse with a new game state
                next_value, _ = SudokuAI.minimax(game_state=game_state, alpha=alpha, beta=beta, depth=depth - 1)

                # DEBUG
                # print(f"move = {move}, depth = {depth}, value = {next_value}, init scores = {game_state.scores}")

                # undo the move in the game (avoiding copying game state)
                game_state.board.put(move.square, game_state.board.empty)
                game_state.occupied_squares1.pop()
                game_state.current_player = 1
                game_state.scores[game_state.current_player - 1] -= move_score

                # update best value and best move if needed
                if next_value > value:
                    value = next_value
                    best_move = move

                # alpha/beta pruning logic
                if value >= beta:
                    break
                alpha = max(alpha, value)

        else:
            # minimizing case
            value = float("inf")
            best_move = None

            for move in possible_moves:
                # put new value, add square to occupied squares, change scores, change player
                game_state.board.put(move.square, move.value)
                game_state.occupied_squares2.append(move.square)
                move_score = SudokuAI.calculate_score(game_state, move)
                game_state.scores[game_state.current_player - 1] += move_score
                game_state.current_player = 1

                # recurse with a new game state
                next_value, _ = SudokuAI.minimax(game_state=game_state, alpha=alpha, beta=beta, depth=depth - 1)

                # DEBUG
                # print(f"move = {move}, depth = {depth}, value = {next_value}, init scores = {game_state.scores}")

                # undo the move in the game (avoiding copying game state)
                game_state.board.put(move.square, game_state.board.empty)
                game_state.occupied_squares2.pop()
                game_state.current_player = 2
                game_state.scores[game_state.current_player - 1] -= move_score

                # update best value and best move if needed
                if next_value < value:
                    value = next_value
                    best_move = move

                # alpha/beta pruning logic
                if value <= alpha:
                    break
                beta = min(beta, value)

        return value, best_move

    def compute_best_move(self, game_state: GameState) -> None:
        for depth in range(2, game_state.board.N**2):
            val, move = self.minimax(game_state, depth=depth)
            # print("*" * 100)
            # print(f"Selected MINIMAX move - {move} with score - {val} | depth - {depth}")
            # print("*" * 100)
            self.propose_move(move)
