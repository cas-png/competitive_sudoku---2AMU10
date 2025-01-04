from typing import Optional, Set, Tuple
import copy
import math
import random

import competitive_sudoku.sudokuai
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard


class MCTSNode:
    """Represents each node separately in the Monte Carlo tree.
    """
    def __init__(self, game_state: GameState, move: Move = None, parent = None):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.visits = 0
        self.s_A = 0
        self.s_B = 0
        self.s_draw = 0
        self.signed_score = 0
        self.children = list()


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
    def get_possible_moves(game_state: GameState) -> Optional[int]:
        """Returns possible value for the position (i, j) of the game and None if it does not exist."""
        # extract possible moves
        player_squares = game_state.player_squares()

        # If all moves are available - enumerate them
        if player_squares is None:
            player_squares = [(i, j) for i in range(game_state.board.N) for j in range(game_state.board.N)]

        # Apply additional filtering of possible moves (e.g. check possible values, taboo, cell emptiness)
        possible_moves = list()
        for i, j in player_squares:
            val = MCTSNode.get_possible_value(game_state, i, j)
            if val is not None:
                possible_moves.append(Move(square=(i, j), value=val))
        
        return possible_moves


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
    def put_move(game_state: GameState, move: Move):
        """Puts a move on the board and change all the things that follow from this move.
        """
        new_game_state = copy.deepcopy(game_state)
        new_game_state.board.put(move.square, move.value)
        new_game_state.occupied_squares1.append(move.square)
        move_score = MCTSNode.calculate_score(new_game_state, move)
        new_game_state.scores[new_game_state.current_player - 1] += move_score

        if new_game_state.current_player == 1:
            new_game_state.current_player = 2
        else:
            new_game_state.current_player = 1
        
        return new_game_state

 
    def expand_tree(self):
        """Expand the tree by adding the children as leafs.
        """
        possible_moves = MCTSNode.get_possible_moves(self.game_state)
        for move in possible_moves:
            new_game_state = MCTSNode.put_move(self.game_state, move)
            self.children.append(MCTSNode(new_game_state, move, self))


    def select_child(self, C: float = 2.0):
        """Select the child with the highest UCT score.
        """ 
        best_UCT_score = float("-inf")
        best_child = None
        for child in self.children:
            if child.visits == 0:
                best_UCT_score = float("inf")
                best_child = child
                return best_child
            else:
                UCT_score = (child.signed_score / child.visits) + C * math.sqrt(math.log(self.visits) / child.visits)
                if UCT_score > best_UCT_score:
                    best_UCT_score = UCT_score
                    best_child = child
        
        return best_child
    
    @staticmethod
    def check_winner(game_state: GameState):
        """Check which player wins. Returns 1 if player 1 wins, 2 if player 2 wins, and 0 if there is a draw.
        """
        if game_state.scores[0] > game_state.scores[1]:
            winner = 1
        elif game_state.scores[1] > game_state.scores[0]:
            winner = 2
        else:
            winner = 0
        return winner
    
    @staticmethod
    def check_full_grid(game_state: GameState):
        """Checks if the grid is full.
        """
        for i in range(game_state.board.N):
            for j in range(game_state.board.N):
                val = game_state.board.get(square=(i, j))
                if val == game_state.board.empty:
                    return False
        
        return True
    
    def random_playout(self):
        """Simulation between random agents.
        """
        state = copy.deepcopy(self.game_state)

        while not MCTSNode.check_full_grid(state):
            possible_moves = MCTSNode.get_possible_moves(state)
            if len(possible_moves) != 0:
                move = random.choice(possible_moves)
                state =  MCTSNode.put_move(state, move)
            else:
                break
        
        return MCTSNode.check_winner(state)
    
    def backpropagate(self, outcome, player_winner):
        """Backpropagate trough the tree. 
        Outcome is a string, which determines if 'A' wins, 'B' wins or a draw.
        Player_winner is 1 if player 1 wins, 2 if player 2 wins, and 0 if there is a draw
        """
        self.visits += 1

        if outcome == 'Draw':
            self.s_draw += 1
        elif outcome == 'A':
            self.s_A += 1
        else:
            self.s_B += 1
        
        if self.game_state.current_player == player_winner: # Player A is allowed to move in this node
            self.signed_score = -self.s_A
        else: # Player A is allowed to move in this node
            self.signed_score = self.s_A
        
        if self.parent != None:
            self.parent.backpropagate(outcome, player_winner)




class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()

    def compute_best_move(self, game_state: GameState) -> None:
        root = MCTSNode(game_state)
        player_A = game_state.current_player # Assume that player A is allowed to put a move in the root

        root.expand_tree()

        for _ in range(1000000):
            leaf = self.select_leaf(root)
            if leaf.visits > 0:
                leaf.expand_tree()
                leaf = self.select_leaf(leaf)
            winner_random_playout = leaf.random_playout()  # Either 0, 1 or 2
            if winner_random_playout == 0: # Draw
                outcome_random_playout = 'Draw'
            elif winner_random_playout == player_A: # Player A wins
                outcome_random_playout = 'A'
            else: # Player B wins
                outcome_random_playout = 'B'

            leaf.backpropagate(outcome_random_playout, winner_random_playout)

            # Robust child
            robust_node = root.children[0]
            largest_visits = 0
            for child in root.children:
                if child.visits > largest_visits:
                    robust_node = child
                    largest_visits = child.visits

            robust_move = robust_node.move
            if robust_move:
                self.propose_move(robust_move)

            # Max child
            # max_node = root.children[0]
            # largest_avg_score = float("-inf")
            # for child in root.children:
            #     if child.visits == 0:
            #         max_node = child
            #         break
            #     else:
            #         child_avg_score = (child.signed_score / child.visits)
            #         if child_avg_score > largest_avg_score:
            #             max_node = child
            #             largest_avg_score = child_avg_score

            # max_move = max_node.move
            # if max_move:
            #     self.propose_move(max_move)
    

    def select_leaf(self, node):
        """Recursively select the child with the highest UCT score.
        """
        for _ in range(node.game_state.board.N**2+1): # Max depth of the tree cannot be larger than N**2 (+1)
            if node.children == list():
                break
            node = node.select_child()
        return node
            
            
