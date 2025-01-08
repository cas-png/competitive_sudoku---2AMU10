import re
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
from gym import spaces

from execute import solve_sudoku
from random_player import SudokuAI
from sudoku import (Move, SudokuBoard, TabooMove, parse_game_state,
                    pretty_print_game_state, GameState)

SUDOKU_SOLVER = "../bin/solve_sudoku"


def get_possible_value(game_state: GameState, i: int, j: int) -> Optional[int]:
    """Return a possible valid value for position (i, j) on the board given the current game state.

    This function checks the vertical, horizontal, and regional constraints, as well as taboo moves,
    to determine a valid value that can be placed at the specified position. If no valid moves
    exist, it returns None.
    """
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


class CompetitiveSudokuEnv(gym.Env):
    """Custom Gym environment for competitive Sudoku gameplay."""
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
            self: "CompetitiveSudokuEnv",
            board_file: str = "../boards/empty-3x3.txt",
            playmode: str = "rows",
            agent_player: int = 1,
            render_mode: str = "ansi",
    ) -> None:
        """Initialize the CompetitiveSudokuEnv.

        Args:
            board_file: Path to the text file containing the board configuration.
            playmode: Mode of play, e.g., 'rows'.
            agent_player: Indicates whether the agent is player 1 or 2.
            render_mode: Rendering mode; defaults to 'ansi'.
        """
        self.board_file = board_file
        self.playmode = playmode

        text = Path(self.board_file).read_text()
        self.game_state = parse_game_state(text, self.playmode)

        self.agent_player = agent_player
        self.opponent_agent = SudokuAI()

        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.game_state.board.N ** 2,), dtype=int)
        self.action_space = gym.spaces.Discrete(self.game_state.board.N ** 2)

        self.move_number = 0
        self.number_of_moves = self.game_state.board.squares.count(SudokuBoard.empty)
        self.finished_players = set()

        # If the agent is player 2, let the opponent make the first move.
        if self.agent_player == 2:
            _, move = self.opponent_agent.compute_best_move(self.game_state)
            self.game_state.board.put(move.square, move.value)
            self.game_state.moves.append(move)
            self.game_state.occupied_squares().append(move.square)
            self.move_number += 1
            self.game_state.current_player = 3 - self.game_state.current_player

    def _get_obs(self) -> List[int]:
        """Return the current observation of the board state as a list of integers."""
        observation = self.game_state.board.squares.copy()

        value = 1 if self.agent_player == 1 else -1
        for square in self.game_state.occupied_squares1:
            idx = self.game_state.board.square2index(square)
            observation[idx] = value

        value = -1 if self.agent_player == 1 else 1
        for square in self.game_state.occupied_squares2:
            idx = self.game_state.board.square2index(square)
            observation[idx] = value

        return observation

    def get_allowed_inds(self) -> List[int]:
        """Return a list of allowed action indices based on the current player's valid moves."""
        player_squares = self.game_state.player_squares()

        if player_squares is None:
            player_squares = [(i, j) for i in range(self.game_state.board.N) for j in range(self.game_state.board.N)]

        possible_moves = list()
        for i, j in player_squares:
            val = get_possible_value(self.game_state, i, j)
            if val is not None:
                possible_moves.append((i, j))

        return [self.game_state.board.square2index(square) for square in possible_moves]

    def reset(self):
        """Reset the environment to its initial state and return the initial observation."""
        self.__init__(self.board_file, self.playmode, self.agent_player, self.render_mode)
        return self._get_obs(), dict()

    def render(self):
        """Render the current game state in ANSI format."""
        if self.render_mode == "ansi":
            print(pretty_print_game_state(self.game_state))
        else:
            raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported")

    def show_info(self, info):
        """Print additional information provided in the `info` dictionary."""
        for key, value in info.items():
            print(key, "\t", value)

    def step(self, best_move: Move):
        """Execute a step in the environment using the provided best move and return the result.

        Args:
            best_move: The move chosen by the agent for this step.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        info = dict()
        player_squares = self.game_state.player_squares()
        player_score = 0
        opponent_score = 0

        if player_squares:
            info["player"] = best_move
            i, j = best_move.square
            value = best_move.value

            board_text = str(self.game_state.board)
            options = f'--move "{self.game_state.board.square2index(best_move.square)} {value}"'

            if player_squares is not None:
                allowed = " ".join(str(self.game_state.board.square2index(square)) for square in player_squares)
                options += f' --allowed="{allowed}"'

            output = solve_sudoku(SUDOKU_SOLVER, board_text, options)

            if "has no solution" in output:
                self.game_state.moves.append(TabooMove(best_move.square, value))
                self.game_state.taboo_moves.append(TabooMove(best_move.square, value))

            if "The score is" in output:
                match = re.search(r"The score is ([-\d]+)", output)
                player_score += int(match.group(1))
                self.game_state.board.put(best_move.square, value)
                self.game_state.moves.append(best_move)
                self.game_state.occupied_squares().append(best_move.square)
                self.move_number += 1

            self.game_state.scores[self.game_state.current_player - 1] += player_score
            self.game_state.current_player = 3 - self.game_state.current_player

        # Opponent's turn.
        player_squares = self.game_state.player_squares()
        if player_squares:
            _, move = self.opponent_agent.compute_best_move(self.game_state)
            info["opponent"] = move
            i, j = move.square
            value = move.value

            board_text = str(self.game_state.board)
            options = f'--move "{self.game_state.board.square2index(move.square)} {value}"'

            if player_squares is not None:
                allowed = " ".join(str(self.game_state.board.square2index(square)) for square in player_squares)
                options += f' --allowed="{allowed}"'

            output = solve_sudoku(SUDOKU_SOLVER, board_text, options)

            if "has no solution" in output:
                self.game_state.moves.append(TabooMove(move.square, value))
                self.game_state.taboo_moves.append(TabooMove(move.square, value))

            if "The score is" in output:
                match = re.search(r"The score is ([-\d]+)", output)
                opponent_score = int(match.group(1))
                self.game_state.board.put(move.square, value)
                self.game_state.moves.append(move)
                self.game_state.occupied_squares().append(move.square)
                self.move_number += 1

            self.game_state.scores[self.game_state.current_player - 1] += opponent_score
            self.game_state.current_player = 3 - self.game_state.current_player
        else:
            self.finished_players.add(self.game_state.current_player)
            self.game_state.current_player = 3 - self.game_state.current_player

        player_squares = self.game_state.player_squares()
        if not player_squares:
            self.finished_players.add(self.game_state.current_player)
            self.game_state.current_player = 3 - self.game_state.current_player
            # Let opponent finish remaining moves.

            while self.move_number < self.number_of_moves and len(self.finished_players) < 2:
                player_squares = self.game_state.player_squares()

                _, move = self.opponent_agent.compute_best_move(self.game_state)
                i, j = move.square
                value = move.value
                board_text = str(self.game_state.board)
                options = f'--move "{self.game_state.board.square2index(move.square)} {value}"'

                if player_squares is not None:
                    allowed = " ".join(str(self.game_state.board.square2index(square)) for square in player_squares)
                    options += f' --allowed="{allowed}"'

                output = solve_sudoku(SUDOKU_SOLVER, board_text, options)

                if "has no solution" in output:
                    self.game_state.moves.append(TabooMove(move.square, value))
                    self.game_state.taboo_moves.append(TabooMove(move.square, value))

                if "The score is" in output:
                    match = re.search(r"The score is ([-\d]+)", output)
                    opponent_score += int(match.group(1))
                    self.game_state.board.put(move.square, value)
                    self.game_state.moves.append(move)
                    self.game_state.occupied_squares().append(move.square)
                    self.move_number += 1

                self.game_state.scores[self.game_state.current_player - 1] += player_score

        reward = player_score / 7  # Normalize the score to [0, 1]
        info["reward"] = reward
        terminated = False
        if self.move_number >= self.number_of_moves or len(self.finished_players) >= 1:
            terminated = True
        info["terminated"] = terminated

        return self._get_obs(), reward, terminated, False, info
