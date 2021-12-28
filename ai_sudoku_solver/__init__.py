__version__ = '1.0.0'

import os

import numpy as np
import requests
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


class SudokuSolver():
    def __init__(self, model_name_or_path):
        self.model = self.load_model(model_name_or_path)
        
    def load_model(self, model_name_or_path):
        model_root_path = os.path.join(os.path.expanduser("~"), 'ai-models-cache')
        model_file_path = os.path.join(model_root_path, f'{model_name_or_path}.h5')
        if not os.path.exists(model_file_path):
            if not os.path.exists(model_root_path):
                os.mkdir(model_root_path)
        
            response = requests.get(f"https://ritvik19.github.io/ai-models/{model_name_or_path}.h5", stream=True)
            with open(model_file_path, "wb") as handle:
                for data in tqdm(response.iter_content(1_048_576)):
                    handle.write(data)
        
        return tf_load_model(model_file_path)
        
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