import torch
import torch_geometric
from torch_geometric.data import Data
import yaml
import torch.nn.functional as F
import torch.nn as nn
import os
from src.data import dataset
from tqdm import tqdm
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN
from src.utils.loss import select_criterion
from src.utils.optim import setup_optimization
from src.data.data_utils import get_node_types, get_edge_types
from src.utils.helpers import overwrite_conf, count_parameters, assert_symms, set_seed, mask_input, mask_hidden, count_named_parameters
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import wandb
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.data.cifar10_dataset import cnn_to_tg_data
import pandas as pd
import collections
import time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

from prune_utils import (
    load_model_layout, unflatten_weights, calculate_parameter_sparsity,
    generate_static_graph_attributes, calculate_pruning_loss, evaluate_pruned_model,
    PrunedCNN, get_scalegmn_prediction, fine_tune_pruned_model, magnitude_prune_weights,
    get_cifar10_loaders, transform_weights_biases, run_gradual_pruning_evaluation
)

def initialize_config(args):
    """Initialize configuration from YAML file and overwrite with CLI arguments."""
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))
    assert_symms(conf)
    return conf


def load_best_cnn_weights(conf, model_layout, device):
    """Load the best CNN weights and biases."""
    weights_path = conf['data']['data_path']
    all_weights = np.load(weights_path, mmap_mode='r')
    best_cnn_weights = all_weights[172538].astype(np.float32)
    print(f'Using weights index: 172538')
    raw_weights, raw_biases = unflatten_weights(best_cnn_weights, model_layout, device)
    return raw_weights, raw_biases


def setup_optimizer_and_scheduler(params_to_optimize, conf):
    """Set up optimizer and scheduler."""
    conf_opt = conf['optimization']
    optimizer, scheduler = setup_optimization(
        params_to_optimize,
        optimizer_name=conf_opt['optimizer_name'],
        optimizer_args=conf_opt['optimizer_args'],
        scheduler_args=conf_opt['scheduler_args']
    )
    return optimizer, scheduler


def evaluate_models(models, device):
    """Evaluate pruned models."""
    for model_path in models:
        cnn_model = PrunedCNN().to(device)
        cnn_model.load_state_dict(torch.load(model_path, map_location=device))
        _, _, test_loader = get_cifar10_loaders(batch_size=256)
        test_acc = evaluate_pruned_model(cnn_model, test_loader, device)
        print(f"Test accuracy for {model_path}: {test_acc:.2f}")


def main(args):
    conf = initialize_config(args)
    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")
    set_seed(conf['train_args']['seed'])

    model_layout = load_model_layout(conf['data']['layout_path'])
    raw_weights, raw_biases = load_best_cnn_weights(conf, model_layout, device)
    static_graph_attrs = generate_static_graph_attributes(
        raw_weights, [1, 16, 16, 16, 10], device
    )

    net = ScaleGMN(conf['scalegmn_args']).to(device)
    net.load_state_dict(torch.load(conf['scalegmn_args']['pretrained_path'], map_location=device))
    net.eval()

    run_gradual_pruning_evaluation(
        raw_weights, raw_biases, 
        net, model_layout, 
        {'sequential/conv2d/kernel:0':   'features.0.weight', # First Conv2d
        'sequential/conv2d/bias:0':     'features.0.bias',
        'sequential/conv2d_1/kernel:0': 'features.3.weight', # Second Conv2d (index 3: Conv,ReLU,Drop)
        'sequential/conv2d_1/bias:0':   'features.3.bias',
        'sequential/conv2d_2/kernel:0': 'features.6.weight', # Third Conv2d (index 6: Conv,ReLU,Drop)
        'sequential/conv2d_2/bias:0':   'features.6.bias',
        'sequential/dense/kernel:0':    'fc.weight',         # Final Linear layer (index 11 if features/fc separate)
        'sequential/dense/bias:0':      'fc.bias'},
        {"activation": "ReLU", # Or load from conf
         "dropout_rate": 0.053518, # Or load from conf
         "num_classes": 10},
        static_graph_attrs, device, conf
    )

if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    main(args)

