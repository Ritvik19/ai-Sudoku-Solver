__version__ = "1.0.7"

import os

import numpy as np
from huggingface_hub import from_pretrained_keras
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


class SudokuSolver:
    """Utility class for the pipeline which solves the Sudoku Puzzles
    To solve Sudoku Puzzles 
    
    initialize a Sudoku Solver object
    
    create an array of dimension (n, 9, 9)
    where n is the number of puzzles you want to solve
    also, replace the blank items with a zero.
    
    then, just call the sudoku solver object on the array
    
    Args:
        model_name (str): THe name of the model to be used
        
    """

    def __init__(self, model_name):
        self.model = from_pretrained_keras(model_name)

    def __call__(self, grids):
        """This function solves quizzes. 
        It will fill blanks one after the other. Each time a digit is filled, 
        the new grid will be fed again to the solver to predict the next digit. 
        again and again, until there is no more blank
        
        Args:
            grids (np.array), shape (?, 9, 9): Batch of quizzes to solve
            
        Returns:
            grids (np.array), shape (?, 9, 9): Solved quizzes.
        """
        grids = grids.copy()
        for _ in range((grids == 0).sum((1, 2)).max()):
            preds = np.array(self.model.predict(to_categorical(grids)))
            probs = preds.max(2).T
            values = preds.argmax(2).T + 1
            zeros = (grids == 0).reshape((grids.shape[0], 81))

            for grid, prob, value, zero in zip(grids, probs, values, zeros):
                if any(zero):
                    where = np.where(zero)[0]
                    confidence_position = where[prob[zero].argmax()]
                    confidence_value = value[confidence_position]
                    grid.flat[confidence_position] = confidence_value
        return grids
