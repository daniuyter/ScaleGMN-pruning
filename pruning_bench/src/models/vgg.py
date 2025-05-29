import numpy as np
import os
import sys
from collections import OrderedDict
import torch

vgg_helper_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'VGG helper functions'))
if vgg_helper_dir not in sys.path:
    sys.path.append(vgg_helper_dir)

from vgg import vgg19

def load_vgg19():
    model = vgg19()
    # ToDo change to GPU if available
    relative_path = os.path.join('..', '..', 'configs', 'VGG helper functions', 'model_best_cpu.pth.tar')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.abspath(os.path.join(script_dir, relative_path))
    # Load the model
    checkpoint = torch.load(checkpoint_path)
    original_state_dict = checkpoint['state_dict']

    # Create a new state_dict without the 'module.' in it (e.g. features.module.0.weight -> features.0.weight)
    new_state_dict = OrderedDict()
    for k, v in original_state_dict.items():
        name = k.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("Model loaded successfully.")
    # Print the model
    print(model)
    return model
