import numpy as np
from ai_sudoku_solver import __version__, SudokuSolver


def test_version():
    assert __version__ == '1.0.3'


def test_run():
    model = SudokuSolver("sudoku-net-v1")
    grid = np.array([[
        [0, 0, 4, 3, 0, 0, 2, 0, 9],
        [0, 0, 5, 0, 0, 9, 0, 0, 1],
        [0, 7, 0, 0, 6, 0, 0, 4, 3],
        [0, 0, 6, 0, 0, 2, 0, 8, 7],
        [1, 9, 0, 0, 0, 7, 4, 0, 0],
        [0, 5, 0, 0, 8, 3, 0, 0, 0],
        [6, 0, 0, 0, 0, 0, 1, 0, 5],
        [0, 0, 3, 5, 0, 8, 6, 9, 0],
        [0, 4, 2, 9, 1, 0, 3, 0, 0]
    ]])
    assert model(grid).shape == grid.shape