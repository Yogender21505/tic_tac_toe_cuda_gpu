import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Board representation
EMPTY = 0
X = 1
O = 2

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.int32)

    def print_board(self):
        for row in self.board:
            print(' '.join(['_' if cell == EMPTY else 'X' if cell == X else 'O' for cell in row]))
        print()

    def is_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == player:
            return True
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == player:
            return True
        return False

    def is_draw(self):
        return not np.any(self.board == EMPTY)

    def make_move(self, player, row, col):
        if self.board[row, col] == EMPTY:
            self.board[row, col] = player
            return True
        return False

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int32)

# GPU code for Minimax
minimax_code = """
__device__ int evaluate(int *board) {
    for (int row = 0; row < 3; ++row) {
        if (board[row * 3] == board[row * 3 + 1] && board[row * 3 + 1] == board[row * 3 + 2]) {
            if (board[row * 3] != 0)
                return (board[row * 3] == 1) ? 10 : -10;
        }
    }
    for (int col = 0; col < 3; ++col) {
        if (board[col] == board[3 + col] && board[3 + col] == board[6 + col]) {
            if (board[col] != 0)
                return (board[col] == 1) ? 10 : -10;
        }
    }
    if (board[0] == board[4] && board[4] == board[8]) {
        if (board[0] != 0)
            return (board[0] == 1) ? 10 : -10;
    }
    if (board[2] == board[4] && board[4] == board[6]) {
        if (board[2] != 0)
            return (board[2] == 1) ? 10 : -10;
    }
    return 0;
}

__global__ void minimax(int *board, int depth, int isMax, int *result) {
    int score = evaluate(board);
    if (score == 10) {
        *result = score - depth;
        return;
    }
    if (score == -10) {
        *result = score + depth;
        return;
    }
    bool movesLeft = false;
    for (int i = 0; i < 9; ++i) {
        if (board[i] == 0) {
            movesLeft = true;
            break;
        }
    }
    if (!movesLeft) {
        *result = 0;
        return;
    }

    int best;
    if (isMax) {
        best = -1000;
        for (int i = 0; i < 9; ++i) {
            if (board[i] == 0) {
                board[i] = 1;
                int temp_result;
                minimax<<<1, 1>>>(board, depth + 1, false, &temp_result);
                cudaDeviceSynchronize();
                best = max(best, temp_result);
                board[i] = 0;
            }
        }
    } else {
        best = 1000;
        for (int i = 0; i < 9; ++i) {
            if (board[i] == 0) {
                board[i] = 2;
                int temp_result;
                minimax<<<1, 1>>>(board, depth + 1, true, &temp_result);
                cudaDeviceSynchronize();
                best = min(best, temp_result);
                board[i] = 0;
            }
        }
    }
    *result = best;
}
"""

mod = SourceModule(minimax_code)
minimax = mod.get_function("minimax")

def find_best_move(board, isMax):
    board_flat = board.flatten().astype(np.int32)
    result = np.zeros(1, dtype=np.int32)
    minimax(
        drv.InOut(board_flat), np.int32(0), np.int32(isMax), drv.Out(result),
        block=(1, 1, 1), grid=(1, 1)
    )
    return result[0]

# Game play function
def play_game():
    game = TicTacToe()
    players = [X, O]
    current_player = 0

    while True:
        # Replace f-string usage
        print("Player {} turn".format(players[current_player]))

        game.print_board()
        
        if players[current_player] == X:
            # GPU 1: Minimax algorithm
            move_value = find_best_move(game.board, True)
            print("Minimax value: {}".format(move_value))
            # Execute the best move found by minimax (simplified for demonstration)
            for i in range(3):
                for j in range(3):
                    if game.make_move(X, i, j):
                        break
                else:
                    continue
                break
        else:
            # GPU 2: Heuristic-based move (random move for demonstration)
            move_made = False
            for i in range(3):
                for j in range(3):
                    if game.make_move(O, i, j):
                        move_made = True
                        break
                if move_made:
                    break
        
        if game.is_winner(players[current_player]):
            print("Player {} wins!".format(players[current_player]))
            game.print_board()
            break
        
        if game.is_draw():
            print("The game is a draw!")
            game.print_board()
            break
        
        current_player = 1 - current_player

if __name__ == "__main__":
    play_game()
