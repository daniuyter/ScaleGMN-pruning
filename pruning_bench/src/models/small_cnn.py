import numpy as np
import os
import sys
# Add the directory containing CNN_evaluator to the system path
cnn_evaluator_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'CNN helper functions'))
if cnn_evaluator_dir not in sys.path:
    sys.path.append(cnn_evaluator_dir)

from CNN_evaluator import convert_weight_vector_to_model

def load_small_cnn():
    relative_path = os.path.join('..', '..', 'configs', 'CNN helper functions', 'optimal_weights.npy')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_path = os.path.abspath(os.path.join(script_dir, relative_path))
    # Load the weights
    optimal_weights = np.load(absolute_path)
    
    # Convert the weights to a model
    model = convert_weight_vector_to_model(optimal_weights)
    print("Model loaded successfully.")
    # Print the model 
    print(model)
    return model