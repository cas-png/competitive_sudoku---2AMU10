import torch
import torch.nn as nn
from typing import List, Optional
from competitive_sudoku.sudoku import GameState, Move
import competitive_sudoku.sudokuai


class DQN(nn.Module):
    """
    A simple Deep Q-Network model for approximating Q-values in the competitive Sudoku game.

    Attributes:
        fc1: First fully connected layer.
        fc3: Output fully connected layer.
    """

    def __init__(self, input_dim=81, hidden_dim=512, output_dim=81):
        """
        Initialize the DQN model with a single hidden layer.

        Args:
            input_dim: Dimension of the input layer.
            hidden_dim: Dimension of the hidden layer.
            output_dim: Dimension of the output layer.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor representing the board state.

        Returns:
            Output tensor of Q-values for each possible action.
        """
        x = torch.relu(self.fc1(x))
        return self.fc3(x)


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given Sudoku board configuration using Deep Q-Learning.

    The AI maintains separate models for different board sizes (2x2, 2x3, 3x3) and for each player.
    """
    _models: dict = {
        "2x2": {
            1: DQN(input_dim=16, hidden_dim=256, output_dim=16),
            2: DQN(input_dim=16, hidden_dim=256, output_dim=16),
        },
        "2x3": {
            1: DQN(input_dim=36, hidden_dim=256, output_dim=36),
            2: DQN(input_dim=36, hidden_dim=256, output_dim=36),
        },
        "3x3": {
            1: DQN(input_dim=81, hidden_dim=512, output_dim=81),
            2: DQN(input_dim=81, hidden_dim=512, output_dim=81),
        },
    }

    def __init__(self):
        """
        Initialize the Sudoku AI by loading pre-trained models for various board configurations.
        """
        super().__init__()
        for board in ["2x2", "2x3", "3x3"]:
            for player in [1, 2]:
                self._models[board][player].load_state_dict(
                    torch.load(f"team07_A3_RL/models/model_{board}_{player}.pt", weights_only=True)
                )

    def get_possible_value(self, game_state: GameState, i: int, j: int) -> Optional[int]:
        """
        Return a possible value for position (i, j) on the board given the current game state.

        This function checks the vertical, horizontal, and regional constraints, as well as taboo moves,
        to determine a valid value that can be placed at the specified position. If no valid moves
        exist, it returns None.

        Args:
            game_state: The current state of the game.
            i: Row index.
            j: Column index.

        Returns:
            A valid value for the square (i, j) or None if not possible.
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

    def get_allowed_inds(self, game_state: GameState) -> List[int]:
        """
        Get the list of indices of squares where the current player can legally place a value.

        This method evaluates each possible square based on the game state to determine valid moves.

        Args:
            game_state: The current state of the game.

        Returns:
            A list of 1D indices corresponding to valid squares for the next move.
        """
        player_squares = game_state.player_squares()

        if player_squares is None:
            player_squares = [(i, j) for i in range(game_state.board.N) for j in range(game_state.board.N)]

        possible_moves = list()
        for i, j in player_squares:
            val = self.get_possible_value(game_state, i, j)
            if val is not None:
                possible_moves.append((i, j))

        return [game_state.board.square2index(square) for square in possible_moves]

    def _get_obs(self, game_state: GameState) -> List[int]:
        """
        Construct and return an observation of the current game state suitable for input into the DQN.

        Args:
            game_state: The current state of the game.

        Returns:
            A list of integers representing the board where values indicate occupancy
            by either player 1, player 2, or empty squares.
        """
        observation = game_state.board.squares.copy()

        value = 1 if game_state.current_player == 1 else -1
        for square in game_state.occupied_squares1:
            idx = game_state.board.square2index(square)
            observation[idx] = value

        value = -1 if game_state.current_player == 1 else 1
        for square in game_state.occupied_squares2:
            idx = game_state.board.square2index(square)
            observation[idx] = value

        return observation

    def select_action(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best action for the current game state using the Deep Q-Network.

        It retrieves the allowed moves, computes Q-values for those moves,
        and selects the move with the highest Q-value.

        Args:
            game_state: The current state of the game.

        Returns:
            A Move object representing the chosen move, or None if no moves are allowed.
        """
        allowed = self.get_allowed_inds(game_state)
        state = self._get_obs(game_state)

        if len(allowed) == 0:
            return None

        with torch.no_grad():
            state_tensor = torch.Tensor(state)
            board_key = f"{game_state.board.m}x{game_state.board.n}"
            current_player = game_state.current_player
            model = self._models[board_key][current_player]
            q_values = model(state_tensor)
            max_idx = torch.argmax(q_values[allowed])
            # Convert 1D index back to 2D coordinates
            i, j = allowed[max_idx] // game_state.board.N, allowed[max_idx] % game_state.board.N
            return Move(square=(i, j), value=self.get_possible_value(game_state, i, j))

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Compute and propose the best move for the given game state.

        This function serves as the main interface for the AI to suggest a move
        based on the current board configuration.

        Args:
            game_state: The current state of the game.
        """
        self.propose_move(self.select_action(game_state))
