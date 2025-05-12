import numpy as np
from functools import reduce

class SudokuSolver():
    def __init__(self, board):
        self.board = board
        self.is_valid = self._valid_begin()

    def solve(self):
        if(self.is_valid):
            is_solved = self._solve_sudoku()
            return is_solved
        else:
            return False

    def _valid_begin(self):
        for row in self.board:
            non_zero = row[row != 0]
            if len(non_zero) != len(np.unique(non_zero)):
                return False
        
        for col in self.board.T:
            non_zero = col[col != 0]
            if len(non_zero) != len(np.unique(non_zero)):
                return False
        
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                subgrid = self.board[i:i+3, j:j+3].flatten()
                non_zero = subgrid[subgrid != 0]
                if len(non_zero) != len(np.unique(non_zero)):
                    return False
        
        return True

    def _get_candidates(self, search_space):
        return np.setdiff1d(np.arange(1, 10), reduce(np.union1d, search_space))

    def _solve_sudoku(self):
        missing = self._get_missing()
        if not self._had_missing(missing):
            return True
        missing_col = self._get_missing_col(missing)
        missing_row = self._get_missing_row(missing)
        search_space = (
                self._get_col(missing_col),
                self._get_row(missing_row),
                self._get_square(missing_col, missing_row)
            )
        for candidate in self._get_candidates(search_space):
            self.board[missing_row, missing_col] = candidate
            if self._solve_sudoku():
                return True
        self.board[missing_row, missing_col] = 0
        return False

    def _get_col(self, idx):
        return self.board[:, idx].reshape(9)

    def _get_row(self, idx):
        return self.board[idx, :].reshape(9)

    def _get_square(self, col, row):
        col = col // 3 * 3
        row = row // 3 * 3
        return self.board[row:row+3, col:col+3].reshape(9)

    def _get_missing(self):
        return np.where(self.board == 0)

    def _had_missing(self, missing):
        return len(missing[0])

    def _get_missing_col(self, missing):
        return missing[1][0]

    def _get_missing_row(self, missing):
        return missing[0][0]