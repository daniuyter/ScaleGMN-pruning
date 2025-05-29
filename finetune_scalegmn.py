import torch
import torch_geometric
from torch_geometric.loader import DataLoader as PyGDataLoader 
from torch.utils.data import random_split, Dataset as TorchDataset # Renamed PyG Dataset

import yaml
import torch.nn.functional as F
import os
# from src.data import dataset # Removed old dataset import
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import wandb
import traceback
import pandas as pd
import collections # For PrunedCNN state_dict loading if needed
import json # For loading JSON layout if used


# Assuming these utility modules are in your project structure (src.utils.*)
# You need to ensure these paths are correct and the modules exist.
from src.utils.setup_arg_parser import setup_arg_parser 
from src.scalegmn.models import ScaleGMN 
from src.utils.loss import select_criterion 
from src.utils.optim import setup_optimization 
from src.utils.helpers import overwrite_conf, count_parameters, assert_symms, set_seed 
# mask_input, mask_hidden, count_named_parameters are not used in this version

# Imports from your prune_utils and cifar10_dataset - ensure these are correct
# and the files are in your Python path or src structure.
from prune_utils import (
    load_model_layout, unflatten_weights, generate_static_graph_attributes, 
    PrunedCNN, transform_weights_biases
)
from src.data.cifar10_dataset import cnn_to_tg_data


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# --- PrunedCNNGraphDataset Class (as provided by user, with modifications) ---
class PrunedCNNGraphDataset(TorchDataset): # Inherit from torch.utils.data.Dataset
    def __init__(self, root_dir, log_file_name="_dataset_generation_log.csv", 
                 model_layout_path='/home/scur2670/scalegmn/data/cifar10/layout.csv',
                 weights_path='/home/scur2670/scalegmn/data/cifar10/weights.npy',
                 tf_to_pytorch_mapping_dict=None,
                 static_layer_layout_list=None, # e.g. [1,16,16,16,10]
                 device_for_processing='cpu', # Device for pre-calculation of static_attrs
                 transform=None): # Removed pre_transform as it's not used by PyG Dataset directly this way
        
        self.root_dir = root_dir
        self.log_file_path = os.path.join(root_dir, log_file_name)
        self.transform = transform
        self.device_for_processing = torch.device(device_for_processing)

        if not os.path.exists(self.log_file_path):
            raise FileNotFoundError(f"Log file not found: {self.log_file_path}")
            
        self.log_df = pd.read_csv(self.log_file_path)
        self.log_df.dropna(subset=['saved_file'], inplace=True)
        self.log_df = self.log_df[self.log_df['saved_file'].apply(lambda x: isinstance(x, str) and x.endswith('.pt'))]
        
        print(f"Dataset initialized. Found {len(self.log_df)} processable entries in the log.")

        # Load static components once
        print("Loading static components for PrunedCNNGraphDataset...")
        self.model_layout = load_model_layout(model_layout_path)
        
        if tf_to_pytorch_mapping_dict is None:
             self.tf_to_pytorch_mapping = {
                'sequential/conv2d/kernel:0':   'features.0.weight',
                'sequential/conv2d/bias:0':     'features.0.bias',
                'sequential/conv2d_1/kernel:0': 'features.3.weight',
                'sequential/conv2d_1/bias:0':   'features.3.bias',
                'sequential/conv2d_2/kernel:0': 'features.6.weight',
                'sequential/conv2d_2/bias:0':   'features.6.bias',
                'sequential/dense/kernel:0':    'fc.weight',
                'sequential/dense/bias:0':      'fc.bias',
            }
        else:
            self.tf_to_pytorch_mapping = tf_to_pytorch_mapping_dict

        # Load a template weight for static_graph_attributes (e.g., the 7th network as in user's code)
        # This assumes all_weights.npy is accessible and contains enough networks.
        try:
            all_weights_template = np.load(weights_path, mmap_mode='r')
            if len(all_weights_template) <= 7:
                print(f"Warning: Not enough networks in {weights_path} to use index 7 for template. Using index 0.")
                template_idx = 0
            else:
                template_idx = 7
            best_cnn_weights_template = all_weights_template[template_idx].astype(np.float32)
            raw_weights_template, raw_biases_template = unflatten_weights(best_cnn_weights_template, self.model_layout, device='cpu')
            
            self.static_graph_attrs = generate_static_graph_attributes(
                weights_template=raw_weights_template,
                layer_layout_list=[1,16,16,16,10], # Use provided or default
                device=self.device_for_processing 
            )
            print("Static graph attributes generated successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Template weights file not found: {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Error generating static graph attributes: {e}")


    def __len__(self): # PyTorch Dataset uses __len__
        return len(self.log_df)

    def __getitem__(self, idx): # PyTorch Dataset uses __getitem__
        if idx >= len(self.log_df):
            raise IndexError("Index out of bounds")
            
        log_entry = self.log_df.iloc[idx]
        file_path = log_entry['saved_file']

        if not os.path.exists(file_path):
            print(f"ERROR: Data file not found at {file_path} for index {idx}.")
            # Return a dummy/placeholder or handle error appropriately
            return None # Dataloader will skip None items if collate_fn handles it

        try:
            loaded_pt_data = torch.load(file_path, map_location='cpu') 
        except Exception as e:
            print(f"ERROR: Could not load .pt file {file_path}. Error: {e}")
            return None

        final_ft_raw_weights = []
        final_ft_raw_biases = []
        finetuned_state_dict = loaded_pt_data['model_state_dict']

        current_w_idx_orig = 0
        current_b_idx_orig = 0

        # Iterate based on self.model_layout to ensure correct order
        for layer_info in self.model_layout:
            tf_name = layer_info['varname']
            pytorch_name = self.tf_to_pytorch_mapping.get(tf_name)

            is_weight_layer = 'kernel' in tf_name.lower() or 'weight' in tf_name.lower()
            is_bias_layer = 'bias' in tf_name.lower() or 'bias' in tf_name.lower()

            if pytorch_name and pytorch_name in finetuned_state_dict:
                param_tensor = finetuned_state_dict[pytorch_name].clone().detach()
                if is_weight_layer:
                    final_ft_raw_weights.append(param_tensor)
                    current_w_idx_orig +=1
                elif is_bias_layer:
                    final_ft_raw_biases.append(param_tensor)
                    current_b_idx_orig +=1
            elif is_bias_layer: # Handle case where bias might be False in model but present in layout as None
                # This assumes unflatten_weights (for template) produces a list for biases
                # that matches the full layout, including None for biases=False layers.
                # This part needs careful alignment with how your template biases are structured.
                # For simplicity, if not in state_dict, assume it's meant to be None if it's a bias layer.
                final_ft_raw_biases.append(None) 
                current_b_idx_orig +=1


        transformed_ft_weights_for_scalegmn = []
        for w in final_ft_raw_weights:
            if w is not None:
                 transformed_ft_weights_for_scalegmn.append(
                     transform_weights_biases(w.to(self.device_for_processing), self.static_graph_attrs["max_kernel_size"])
                 )
            else: transformed_ft_weights_for_scalegmn.append(None)

        transformed_ft_biases_for_scalegmn = []
        for b in final_ft_raw_biases:
            if b is not None:
                transformed_ft_biases_for_scalegmn.append(
                    transform_weights_biases(b.to(self.device_for_processing), self.static_graph_attrs["max_kernel_size"])
                )
            else: transformed_ft_biases_for_scalegmn.append(None)
        
        # Create PyG Data object using your cnn_to_tg_data
        # This function is external and needs to be correctly defined.
        # It should return a torch_geometric.data.Data object.
        graph_data = cnn_to_tg_data(
            weights=transformed_ft_weights_for_scalegmn,
            biases=transformed_ft_biases_for_scalegmn,
            conv_mask=self.static_graph_attrs["conv_mask"],
            direction=self.static_graph_attrs["direction"], # Assuming 'scalegmn_args.direction'
            layer_layout_list=[1,16,16,16,10], # from generate_static_graph_attributes
            mask_hidden=self.static_graph_attrs["mask_hidden"], # from generate_static_graph_attributes
            node2type=self.static_graph_attrs["node2type"],
            edge2type=self.static_graph_attrs["edge2type"],
            fmap_size=self.static_graph_attrs["fmap_size"],
            sign_mask=self.static_graph_attrs["sign_mask"],
            device=self.device_for_processing # Device for graph construction
        )
        
        # Add target and other metadata to the graph_data object
        graph_data.y = torch.tensor([loaded_pt_data['accuracy_at_epoch_end']], dtype=torch.float)
        graph_data.pretrained_accuracy = torch.tensor([loaded_pt_data.get('pretrained_accuracy', -1.0)], dtype=torch.float)
        graph_data.initial_sparsity = torch.tensor([loaded_pt_data.get('actual_initial_sparsity_before_ft', 0.0)], dtype=torch.float)
        graph_data.epoch_sparsity = torch.tensor([loaded_pt_data.get('sparsity_at_epoch_end', 0.0)], dtype=torch.float)
        graph_data.epoch_num = torch.tensor([loaded_pt_data.get('epoch', 0)], dtype=torch.long)
        graph_data.target_sparsity_applied = torch.tensor([loaded_pt_data.get('target_sparsity',0.0)], dtype=torch.float)
        graph_data.original_network_id = torch.tensor([loaded_pt_data.get('original_network_id', -1)], dtype=torch.long)

        if self.transform:
            graph_data = self.transform(graph_data)
        

        return graph_data


def main(args_cli): 

    with open(args_cli.conf, 'r') as f:
        conf = yaml.safe_load(f)
    conf = overwrite_conf(conf, vars(args_cli))

    assert_symms(conf) 

    print("--- Configuration ---")
    print(yaml.dump(conf, default_flow_style=False))
    print("---------------------")

    device = torch.device("cuda", args_cli.gpu_ids[0]) if args_cli.gpu_ids[0] >= 0 and torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if conf.get("wandb", False): 
        wandb.init(config=conf, **conf.get("wandb_args", {}))

    set_seed(conf['train_args']['seed'])
    # =============================================================================================
    #  SETUP NEW DATASET (PrunedCNNGraphDataset) AND DATALOADER
    # =============================================================================================
    print("Setting up PrunedCNNGraphDataset...")
    pruned_dataset_conf = conf['data'].get('pruned_cnn_dataset', {})
    pruned_dataset_root = pruned_dataset_conf.get('root_dir', './dataset_pruned') # Default from your generation
    pruned_dataset_log_filename = pruned_dataset_conf.get('log_filename', '_dataset_generation_log.csv')
    
    # Static components for the dataset class, fetched from config or hardcoded defaults
    # These paths are for the *template* model used by generate_static_graph_attributes
    model_layout_path = pruned_dataset_conf.get('model_layout_path', 'data/cifar10/layout.csv')
    template_weights_path = pruned_dataset_conf.get('template_weights_path', 'data/cifar10/weights.npy')
    static_layer_layout_list = [1,16,16,16,10] 

    try:
        full_dataset = PrunedCNNGraphDataset(
            root_dir=pruned_dataset_root, 
            log_file_name=pruned_dataset_log_filename,
            model_layout_path=model_layout_path,
            weights_path=template_weights_path, # Path to the .npy file for template
            static_layer_layout_list=static_layer_layout_list,
            device_for_processing='cpu' # Static attrs and graph construction on CPU
        )
    except FileNotFoundError as e:
        print(f"ERROR: Could not initialize dataset. File not found: {e}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while initializing the dataset: {e}")
        traceback.print_exc()
        return

    if len(full_dataset) == 0:
        print("ERROR: The loaded PrunedCNNGraphDataset is empty."); return

    split_ratios = conf['data'].get('train_val_test_split_ratios', [0.8, 0.15, 0.05])
    # ... (rest of your splitting logic - ensure it handles small datasets gracefully) ...
    train_size = int(split_ratios[0] * len(full_dataset))
    val_size = int(split_ratios[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    if test_size < 0: val_size += test_size; test_size = 0
    if val_size < 0: train_size += val_size; val_size = 0
    if train_size <=0 and len(full_dataset) > 0 :
        if len(full_dataset) == 1: train_size = 1; val_size=0; test_size=0
        elif len(full_dataset) > 1 : train_size = 1; val_size = max(0, len(full_dataset)-1); test_size = 0
    
    print(f"Splitting dataset: Total={len(full_dataset)}, Train={train_size}, Val={val_size}, Test={test_size}")
    if train_size == 0 and len(full_dataset) > 0: print("ERROR: Training split size is 0."); return
    
    if len(full_dataset) < 3 and train_size > 0 : # Simplified small dataset handling
        train_set = torch.utils.data.Subset(full_dataset, range(train_size))
        val_indices = list(range(train_size, train_size + val_size)) if val_size > 0 else []
        val_set = torch.utils.data.Subset(full_dataset, val_indices)
        test_indices = list(range(train_size+val_size, len(full_dataset))) if test_size > 0 else []
        test_set = torch.utils.data.Subset(full_dataset, test_indices)
    elif train_size > 0 :
         train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size],
                                                    generator=torch.Generator().manual_seed(conf['train_args']['seed']))
    else: print("ERROR: Could not create a valid training set split."); return


    print(f'Len train set: {len(train_set)}')
    print(f'Len val set: {len(val_set)}')
    print(f'Len test set: {len(test_set)}')

    # Custom collate_fn to filter out None items if get() returns None on error
    def collate_fn_filter_none(batch):
        batch = [data for data in batch if data is not None]
        if not batch: return None # Return None if batch becomes empty
        return PyGDataLoader.collate(batch) # Use default PyG collate

    train_loader = PyGDataLoader(train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=conf.get("num_workers", 0), pin_memory=True, collate_fn=collate_fn_filter_none) # num_workers=0 for easier debugging
    val_loader = PyGDataLoader(val_set, batch_size=conf["batch_size"], shuffle=False, num_workers=conf.get("num_workers", 0), collate_fn=collate_fn_filter_none) if len(val_set) > 0 else None
    test_loader = PyGDataLoader(test_set, batch_size=conf["batch_size"], shuffle=False, num_workers=conf.get("num_workers", 0), collate_fn=collate_fn_filter_none) if len(test_set) > 0 else None

    # =============================================================================================
    #  DEFINE MODEL
    # =============================================================================================
    
    conf['scalegmn_args']["layer_layout"] = static_layer_layout_list
    # conf['scalegmn_args']['input_nn'] = 'conv'
    net = ScaleGMN(conf['scalegmn_args'])
    print(net)
    cnt_p = count_parameters(net=net)

    for p in net.parameters():
        p.requires_grad = True

    net = net.to(device)

    net = ScaleGMN(conf['scalegmn_args'])
    print("\n--- ScaleGMN Model ---"); print(net); print("----------------------")

    fine_tune_conf = conf.get('fine_tune_args', {})
    pretrained_scalegmn_path = 'results/experiment/model_seed_0.pt'
    if pretrained_scalegmn_path and os.path.exists(pretrained_scalegmn_path):
        print(f"Loading pre-trained ScaleGMN model from: {pretrained_scalegmn_path}")
        try:
            net.load_state_dict(torch.load(pretrained_scalegmn_path, map_location=device))
            print("Pre-trained ScaleGMN model loaded successfully.")
        except Exception as e: print(f"ERROR: Could not load pre-trained ScaleGMN model: {e}. Training from scratch.")
    elif pretrained_scalegmn_path: print(f"WARNING: Pre-trained ScaleGMN model path not found: {pretrained_scalegmn_path}. Training from scratch.")
    else: print("INFO: No pre-trained ScaleGMN model path specified.")

    cnt_p = count_parameters(net=net)
    print(f"Number of parameters in ScaleGMN: {cnt_p}")
    if conf.get("wandb", False): wandb.log({'number of parameters': cnt_p}, step=0)

    for p in net.parameters(): p.requires_grad = True
    net = net.to(device)

    # =============================================================================================
    #  DEFINE LOSS & OPTIMIZATION
    # =============================================================================================
    criterion = select_criterion(conf['train_args']['loss'], {})
    optimizer_lr = fine_tune_conf.get('fine_tune_lr', conf['optimization']['optimizer_args'].get('lr', 0.0001))
    current_optimizer_args = conf['optimization']['optimizer_args'].copy()
    current_optimizer_args['lr'] = 1e-4
    print(f"Setting up optimizer for fine-tuning with LR: 1e-4")
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler = setup_optimization(model_params, optimizer_name=conf['optimization']['optimizer_name'], 
                                            optimizer_args=current_optimizer_args, scheduler_args=conf['optimization']['scheduler_args'])
    # =============================================================================================
    # FINE-TUNING LOOP
    # =============================================================================================
    step = 0
    best_val_tau = -float("inf")
    best_test_results, best_val_results, best_train_results_eval = None, None, None
    
    num_fine_tune_epochs = fine_tune_conf.get('fine_tune_epochs', conf['train_args']['num_epochs'])
    print(f"Starting fine-tuning for 200 epochs.")
	
    for epoch in range(200):
        net.train()
        if not train_loader: print(f"Skipping FT epoch {epoch+1} due to empty training loader."); continue
        
        epoch_train_losses = [] # To store losses for this epoch
        len_dataloader = len(train_loader)
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_fine_tune_epochs} FT")
        for i, batch in enumerate(epoch_pbar):
            if batch is None: print(f"Warning: Skipping empty batch at step {i} in epoch {epoch+1}."); continue
            step = epoch * len_dataloader + i 
            batch = batch.to(device); gt_target = batch.y.to(device) / 100
            optimizer.zero_grad()
            pred_acc = F.sigmoid(net(batch)).squeeze(-1)
            loss = criterion(pred_acc, gt_target)
            loss.backward()
            epoch_train_losses.append(loss.item()) # Store batch loss
            if conf['optimization'].get('clip_grad', False): torch.nn.utils.clip_grad_norm_(net.parameters(), conf['optimization']['clip_grad_max_norm'])
            optimizer.step()
            if conf.get("wandb", False) and step % conf.get("wandb_log_freq", 10) == 0:
                log_dict_wandb_train = {f"fine_tune/batch_loss": loss.item()}
                try: log_dict_wandb_train["fine_tune/batch_rsq"] = r2_score(gt_target.cpu().numpy(), pred_acc.detach().cpu().numpy())
                except ValueError: log_dict_wandb_train["fine_tune/batch_rsq"] = 0.0 
                wandb.log(log_dict_wandb_train, step=step)
        avg_epoch_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else float('nan')
        print(f"Epoch {epoch+1} FT Average Training Loss: {avg_epoch_train_loss:.4f}")
        if conf.get("wandb", False): wandb.log({f"fine_tune/avg_epoch_train_loss": avg_epoch_train_loss, "fine_tune_epoch_num": epoch + 1}, step=step)


        if scheduler and scheduler[0]: 
            if scheduler[1] == 'ReduceLROnPlateau' and val_loader and best_val_results: scheduler[0].step(best_val_results['avg_loss']) 
            elif scheduler[1] != 'ReduceLROnPlateau': scheduler[0].step()

        if conf.get("validate", True) and val_loader:
            print(f"Validation after fine-tuning epoch {epoch+1}:")
            val_loss_dict = evaluate(net, val_loader, criterion, device, "val_ft")
            print(f"  Val Results: L1 Err: {val_loss_dict['avg_err']:.4f}, Loss: {val_loss_dict['avg_loss']:.4f}, Rsq: {val_loss_dict['rsq']:.4f}, Tau: {val_loss_dict['tau']:.4f}")
            
            test_loss_dict_at_val, train_eval_loss_dict = None, None
            if test_loader: test_loss_dict_at_val = evaluate(net, test_loader, criterion, device, "test_ft_at_val")
            if conf.get("eval_on_train_set", False) and train_loader: train_eval_loss_dict = evaluate(net, train_loader, criterion, device, "train_ft_eval_end_epoch")

            if val_loss_dict['tau'] >= best_val_tau:
                best_val_tau = val_loss_dict['tau']
                best_test_results = test_loss_dict_at_val
                best_val_results = val_loss_dict
                best_train_results_eval = train_eval_loss_dict 
                if fine_tune_conf.get("save_best_ft_model", True):
                    best_model_save_dir = os.path.join("results", conf.get("experiment_name", "experiment") + "_ft")
                    os.makedirs(best_model_save_dir, exist_ok=True)
                    best_model_path = os.path.join(best_model_save_dir, f"best_ft_model_seed_{conf['train_args']['seed']}.pt")
                    try: torch.save(net.state_dict(), best_model_path); print(f"  Saved best fine-tuned model to {best_model_path} (Val Tau: {best_val_tau:.4f})")
                    except Exception as e: print(f"  Error saving best fine-tuned model: {e}")

            if conf.get("wandb", False):
                wandb_log_val = {}
                # Log all validation metrics every epoch
                wandb_log_val.update({"val_ft/l1_err": val_loss_dict['avg_err'], "val_ft/loss": val_loss_dict['avg_loss'], "val_ft/rsq": val_loss_dict['rsq'], "val_ft/kendall_tau": val_loss_dict['tau']})
                if conf.get("wandb_log_scatter_plot", True) and val_loss_dict and len(val_loss_dict['actual']) > 0 :
                    plt.figure(); plt.scatter(val_loss_dict['actual'], val_loss_dict['pred']); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"Val Scatter FT Ep {epoch+1}"); wandb_log_val["val_ft/scatter"] = wandb.Image(plt); plt.close()
                if best_val_results: wandb_log_val.update({"val_ft/best_rsq_so_far": best_val_results['rsq'], "val_ft/best_tau_so_far": best_val_results['tau']}) # Log current best
                if test_loss_dict_at_val: wandb_log_val.update({"test_ft_at_val/l1_err": test_loss_dict_at_val['avg_err'], "test_ft_at_val/loss": test_loss_dict_at_val['avg_loss'], "test_ft_at_val/rsq": test_loss_dict_at_val['rsq'], "test_ft_at_val/kendall_tau": test_loss_dict_at_val['tau']})
                if best_test_results: wandb_log_val.update({"test_ft_at_val/best_rsq_so_far": best_test_results['rsq'], "test_ft_at_val/best_tau_so_far": best_test_results['tau']})
                if train_eval_loss_dict: wandb_log_val.update({"train_ft_eval_end_epoch/l1_err": train_eval_loss_dict['avg_err'], "train_ft_eval_end_epoch/loss": train_eval_loss_dict['avg_loss'], "train_ft_eval_end_epoch/rsq": train_eval_loss_dict['rsq'], "train_ft_eval_end_epoch/kendall_tau": train_eval_loss_dict['tau']})
                wandb_log_val["fine_tune_epoch_completed"] = epoch + 1 # Use a different key from batch step
                wandb.log(wandb_log_val) # Log per epoch
    print("\n" + "="*80 + "\nFINE-TUNING COMPLETED\n" + "="*80)
    
    final_test_results_after_ft = None
    if test_loader: 
        print("\nEvaluating on Test Set after final fine-tuning epoch...")
        final_test_results_after_ft = evaluate(net, test_loader, criterion, device, "final_test_ft")
    else: print("INFO: Test set is empty. Skipping final test evaluation.")

    results_dir = os.path.join("results", conf.get("experiment_name", "experiment") + "_ft")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_ft_seed_{conf['train_args']['seed']}.yaml")
    final_model_save_path = os.path.join(results_dir, f"final_ft_model_seed_{conf['train_args']['seed']}.pt")
    try: torch.save(net.state_dict(), final_model_save_path); print(f"\nFinal fine-tuned model saved to {final_model_save_path}")
    except Exception as e: print(f"\nError saving final fine-tuned model: {e}")

    results_summary = {
        "final_test_ft": {k: float(v) if isinstance(v, (np.float32, np.float64, float, np.generic)) else v for k, v in final_test_results_after_ft.items()} if final_test_results_after_ft else None,
        "best_validation_ft": {k: float(v) if isinstance(v, (np.float32, np.float64, float, np.generic)) else v for k, v in best_val_results.items()} if best_val_results else None,
        "test_at_best_validation_ft": {k: float(v) if isinstance(v, (np.float32, np.float64, float, np.generic)) else v for k, v in best_test_results.items()} if best_test_results else None,
        "hyperparameters": conf
    }
    with open(results_file, 'w') as f: yaml.dump(results_summary, f, default_flow_style=False)
    print(f"\nFine-tuning results saved to {results_file}")
    
    if conf.get("wandb", False) and wandb.run: 
        wandb_summary = {}
        if final_test_results_after_ft: wandb_summary.update({f"final_test_ft_{k}": v for k,v in final_test_results_after_ft.items() if isinstance(v, (int, float, np.generic))})
        if best_val_results: wandb_summary.update({f"best_val_ft_{k}": v for k,v in best_val_results.items() if isinstance(v, (int, float, np.generic))})
        if best_test_results: wandb_summary.update({f"test_at_best_val_ft_{k}": v for k,v in best_test_results.items() if isinstance(v, (int, float, np.generic))})
        if wandb_summary: wandb.run.summary.update(wandb_summary)
        wandb.finish()


@torch.no_grad()
def evaluate(net, loader, loss_fn, device, scope="eval"): 
    net.eval()
    pred_all, actual_all, err_list, losses_list = [], [], [], []
    if not loader or (hasattr(loader, 'dataset') and len(loader.dataset) == 0):
        # print(f"INFO ({scope}): Loader is empty or dataset is empty. Skipping evaluation.") # Less verbose
        return {"avg_err": 0.0, "avg_loss": 0.0, "rsq": 0.0, "tau": 0.0, "actual": np.array([]), "pred": np.array([])}

    # eval_pbar = tqdm(loader, desc=f"Evaluating ({scope})", leave=False) # Can be too verbose
    for batch in loader: # Changed from eval_pbar
        if batch is None: continue # Skip if collate_fn_filter_none returned None
        batch = batch.to(device)
        gt_target = batch.y.to(device) / 100
        pred_target = net(batch)
        if pred_target.ndim > 1 and pred_target.shape[-1] == 1: pred_target = pred_target.squeeze(-1)
        if not (pred_target.min() >= 0 and pred_target.max() <= 1): pred_target = torch.sigmoid(pred_target)
        err_list.append(torch.abs(pred_target - gt_target).mean().item())
        losses_list.append(loss_fn(pred_target, gt_target).item())
        pred_all.append(pred_target.detach().cpu().numpy())
        actual_all.append(gt_target.cpu().numpy())
        # eval_pbar.set_postfix({"loss": np.mean(losses_list) if losses_list else 0.0})


    avg_err = np.mean(err_list) if err_list else 0.0
    avg_loss = np.mean(losses_list) if losses_list else 0.0
    actual_all = np.concatenate(actual_all) if actual_all else np.array([])
    pred_all = np.concatenate(pred_all) if pred_all else np.array([])
    
    rsq, tau = 0.0, 0.0
    if len(actual_all) > 1 and len(pred_all) > 1 and len(actual_all) == len(pred_all):
        try:
            rsq = r2_score(actual_all, pred_all)
            tau = kendalltau(actual_all, pred_all).correlation
            if np.isnan(tau): tau = 0.0 # Handle NaN case from kendalltau
        except Exception as e:
            print(f"Warning ({scope}): Could not calculate R2 or Kendall Tau: {e}")
    
    return {"avg_err": avg_err, "avg_loss": avg_loss, "rsq": rsq, "tau": tau, "actual": actual_all, "pred": pred_all}


if __name__ == '__main__':
    arg_parser_main = setup_arg_parser() 
    args_main = arg_parser_main.parse_args() 

    if isinstance(args_main.gpu_ids, int):
        args_main.gpu_ids = [args_main.gpu_ids]
    main(args_main)
