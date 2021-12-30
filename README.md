# [ai-Sudoku-Solver](https://github.com/Ritvik19/ai-Sudoku-Solver)

Solving Sudoku Puzzles using Artificial Neural Networks

[![Downloads](https://pepy.tech/badge/ai-sudoku-solver)](https://pepy.tech/project/ai-sudoku-solver)
[![Downloads](https://pepy.tech/badge/ai-sudoku-solver/month)](https://pepy.tech/project/ai-sudoku-solver)
[![Downloads](https://pepy.tech/badge/ai-sudoku-solver/week)](https://pepy.tech/project/ai-sudoku-solver)

---

## Table of Contents

- [ai-Sudoku-Solver](#ai-sudoku-solver)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Gallery](#model-gallery)

---

## Installation

```bash
pip install ai-sudoku-solver
```

---

## Usage

Instantiate a SudokuSolver object

```python
from ai_sudoku_solver import SudokuSolver

solver = SudokuSolver("sudoku-net-v1")
```

Call the model on your puzzles

```python
puzzle = np.array([[
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
solution = solver(puzzle)
# array([[
#     [8, 6, 4, 3, 7, 1, 2, 5, 9],
#     [3, 2, 5, 8, 4, 9, 7, 6, 1],
#     [9, 7, 1, 2, 6, 5, 8, 4, 3],
#     [4, 3, 6, 1, 9, 2, 5, 8, 7],
#     [1, 9, 8, 6, 5, 7, 4, 3, 2],
#     [2, 5, 7, 4, 8, 3, 9, 1, 6],
#     [6, 8, 9, 7, 3, 4, 1, 2, 5],
#     [7, 1, 3, 5, 2, 8, 6, 9, 4],
#     [5, 4, 2, 9, 1, 6, 3, 7, 8]
# ]])
```

---

## Model Gallery

| model         | # parameters | trained on |
| ------------- | -----------: | ---------: |
| sudoku-net-v1 |    3,784,729 | 1M puzzles |
