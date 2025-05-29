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
import wandb
import copy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.data.cifar10_dataset import cnn_to_tg_data
import pandas as pd
import collections
import time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from prune_utils import load_model_layout, unflatten_weights, calculate_parameter_sparsity, generate_static_graph_attributes, calculate_pruning_loss, evaluate_pruned_model, PrunedCNN, get_scalegmn_prediction, fine_tune_pruned_model, magnitude_prune_weights, get_cifar10_loaders, transform_weights_biases, run_gradual_pruning_evaluation 


def setup_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a pruned CNN model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file to evaluate.")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[-1], help="GPU IDs to use (-1 for CPU).")
    parser.add_argument("--conf", type=str, help="Path to the configuration file.")
    return parser

def main(model):
    cnn_model = PrunedCNN().to(device)
    print("Instantiated CNN model with defined architecture:")
    cnn_model.load_state_dict(torch.load(model, map_location=device))

    train_loader, val_loader, test_loader = get_cifar10_loaders(256)
    test_acc = evaluate_pruned_model(cnn_model, test_loader, device)
    print(test_acc, 'test_accuracy model', model)

if __name__ == "__main__":
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")
    model = args.model_path
    main(model)