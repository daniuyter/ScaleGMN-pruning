import argparse
import os
import sys
import torch
import copy
import math
import itertools
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from models.small_cnn import load_small_cnn
    from models.vgg import load_vgg19
    from prune.magnitude import magnitude_prune
    from prune.snip import snip_prune
    from utils.data import get_cifar10_data, _seed_everything
    from fine_tune import fine_tune_pruned_model
    from utils.evaluate import evaluate_model
    # Optional: if you have a sparsity calculation utility
    # from utils.evaluate import calculate_parameter_sparsity
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)


def prune_ft_eval(model, fraction, prune_method, train_loader, val_loader, test_loader, device,
                  max_epochs_safeguard=1000, finetune_lr=1e-4, prune_bias=False, snip_minibatches=1,
                  patience=10, min_delta=1e-5):
    """
    Prune (one-shot), fine-tune, and evaluate the model.
    Used for 'magnitude' and 'snip' methods.
    """
    pruned_model_instance = None
    if prune_method == 'magnitude':
        pruned_model_instance = magnitude_prune(model, prune_fraction=fraction, prune_biases=prune_bias)
    elif prune_method == 'snip':
        pruned_model_instance = snip_prune(model, prune_fraction=fraction, loader=train_loader, device=device, prune_biases=prune_bias, mini_batches=snip_minibatches)
    else:
        raise ValueError(f"Unknown one-shot pruning method: {prune_method}")
    print(f"Model pruned with one-shot method '{prune_method}' to fraction: {fraction}")

    fine_tuned_model, best_val_loss_ft, val_acc_at_best_loss_ft, epochs_run_ft = fine_tune_pruned_model(
        model_to_fine_tune=pruned_model_instance,
        ft_train_loader=train_loader,
        ft_val_loader=val_loader,
        max_epochs_safeguard=max_epochs_safeguard,
        finetune_lr=finetune_lr,
        current_device=device,
        patience=patience,
        min_delta=min_delta
    )
    print("Fine-tuning completed.")

    print("Evaluating the fine-tuned model on the test set...")
    test_accuracy, _ = evaluate_model(
        model=fine_tuned_model,
        data_loader=test_loader,
        device=device
    )
    print(f"Test Accuracy for fraction {fraction}: {test_accuracy * 100:.2f}%")

    return test_accuracy, best_val_loss_ft, val_acc_at_best_loss_ft, epochs_run_ft


def iterative_magnitude_prune_ft_eval(
    initial_model_to_prune,
    target_overall_pruning_fraction,
    num_imp_steps,
    train_loader, val_loader, test_loader, device,
    # Fine-tuning parameters for each IMP step (from main grid search)
    ft_max_epochs_per_step, ft_lr_per_step,
    ft_patience_per_step, ft_min_delta_per_step,
    prune_bias=False
):
    """
    Performs Iterative Magnitude Pruning (IMP), fine-tuning at each step,
    and evaluates the final model.
    """
    current_model = copy.deepcopy(initial_model_to_prune)
    current_model.to(device)

    print(f"\n--- Starting Iterative Magnitude Pruning (IMP) ---")
    print(f"Target final overall fraction: {target_overall_pruning_fraction}, Steps: {num_imp_steps}")

    target_overall_sparsities_at_each_step = []
    if num_imp_steps <= 0:
        raise ValueError("num_imp_steps must be positive.")
    if num_imp_steps == 1: # Equivalent to one-shot magnitude pruning
        target_overall_sparsities_at_each_step = [target_overall_pruning_fraction]
    else:
        # s_k = S_final * ( (t/T_total_imp_steps) ** 3 ) # Cubic schedule from "To prune, or not to prune"
        # Or a simpler linear interpolation of sparsity, or the exponential one:
        # s_k = 1 - ( (1-S_final) ** (k/N) )
        fraction_to_keep_overall_final = 1.0 - target_overall_pruning_fraction
        for i_step in range(1, num_imp_steps + 1):
            target_kept_overall_this_step = fraction_to_keep_overall_final ** (i_step / num_imp_steps)
            target_overall_sparsities_at_each_step.append(1.0 - target_kept_overall_this_step)
        target_overall_sparsities_at_each_step[-1] = target_overall_pruning_fraction # Ensure exact final target

    print(f"Target overall sparsities for magnitude_prune at each IMP step: {[f'{s:.4f}' for s in target_overall_sparsities_at_each_step]}")

    best_val_loss_from_final_ft_step = float('inf')
    val_acc_at_best_loss_from_final_ft_step = 0.0
    total_epochs_run_all_ft_steps = 0

    for step_idx in range(num_imp_steps):
        current_step_target_overall_sparsity = target_overall_sparsities_at_each_step[step_idx]
        print(f"\nIMP Step {step_idx + 1}/{num_imp_steps}: Pruning to overall target sparsity {current_step_target_overall_sparsity:.4f}")

        current_model = magnitude_prune(
            current_model,
            prune_fraction=current_step_target_overall_sparsity, # This is the *overall* target
            prune_biases=prune_bias
        )
        # try: # Optional: print current sparsity
        #     current_sparsity_val = calculate_parameter_sparsity(current_model.parameters())
        #     print(f"  Model sparsity after pruning: {current_sparsity_val:.2f}%")
        # except NameError: pass


        print(f"  Fine-tuning model after IMP step {step_idx + 1}...")
        current_model, best_val_loss_this_ft, val_acc_this_ft, epochs_this_ft = fine_tune_pruned_model(
            model_to_fine_tune=current_model,
            ft_train_loader=train_loader,
            ft_val_loader=val_loader,
            max_epochs_safeguard=ft_max_epochs_per_step,
            finetune_lr=ft_lr_per_step,
            current_device=device,
            patience=ft_patience_per_step,
            min_delta=ft_min_delta_per_step
        )
        total_epochs_run_all_ft_steps += epochs_this_ft
        print(f"  IMP Step {step_idx + 1} Fine-tuning: Best Val Loss: {best_val_loss_this_ft:.4f}, Val Acc: {val_acc_this_ft*100:.2f}%, Epochs: {epochs_this_ft}")

        if step_idx == num_imp_steps - 1: # Metrics from the final fine-tuning step
            best_val_loss_from_final_ft_step = best_val_loss_this_ft
            val_acc_at_best_loss_from_final_ft_step = val_acc_this_ft

    print("\n--- IMP complete ---")
    print("Evaluating the final IMP model on the test set...")
    final_imp_test_accuracy, _ = evaluate_model(
        model=current_model,
        data_loader=test_loader,
        device=device
    )
    print(f"Final IMP Test Accuracy for target fraction {target_overall_pruning_fraction}: {final_imp_test_accuracy * 100:.2f}%")

    return final_imp_test_accuracy, best_val_loss_from_final_ft_step, val_acc_at_best_loss_from_final_ft_step, total_epochs_run_all_ft_steps


def main():
    parser = argparse.ArgumentParser(description="Load, prune, fine-tune, and evaluate a model.")
    parser.add_argument('--model', type=str, help="Name of the model to load (e.g., 'small_cnn')", default='small_cnn')

    # data args
    parser.add_argument('--data_root', type=str, help="Root directory for the dataset", default=None)
    parser.add_argument('--val_split', type=float, help="Fraction of training data to use for validation", default=0.1)
    parser.add_argument('--num_workers', type=int, help="Number of subprocesses to use for data loading", default=2)
    parser.add_argument('--grayscale', action='store_true', help="Use grayscale images")
    parser.add_argument('--batch_size', type=int, help="Batch size for the dataloaders", default=128)
    parser.add_argument('--batch_size_values', type=int, nargs='+', help="List of batch sizes for grid search", default=None)

    # pruning args
    parser.add_argument('--pruning_method', type=str, help="Pruning method to use (magnitude, snip, imp)", default='magnitude')
    parser.add_argument('--pruning_fractions', type=float, nargs='+', help="List of pruning fractions to apply", default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument('--prune_biases_search_values', type=int, nargs='+', help="Grid search values for prune_biases (0 for False, 1 for True). Used for magnitude/imp.", default=None)
    parser.add_argument('--snip_minibatches_search_values', type=int, nargs='+', help="Grid search values for SNIP mini_batches.", default=None)
    # IMP specific arguments
    parser.add_argument('--num_imp_steps_values', type=int, nargs='+', help="Grid search values for number of iterations for Iterative Magnitude Pruning", default=None)


    # experiment args
    parser.add_argument('--seeds', type=int, nargs='+', help="List of seeds for experiments", default=[42, 123, 789])
    parser.add_argument('--output_dir', type=str, help="Directory to save results", default="../outputs") # Relative to src
    parser.add_argument('--lr', type=float, help="Learning rate for fine-tuning", default=1e-4)
    parser.add_argument('--patience', type=int, help="Patience for early stopping", default=10)
    parser.add_argument('--min_delta', type=float, help="Minimum delta for improvement in early stopping", default=1e-5)
    parser.add_argument('--max_epochs_safeguard', type=int, help="Max epochs safeguard for fine-tuning with early stopping", default=1000)

    # Grid search specific for experiment params
    parser.add_argument('--lr_search_values', type=float, nargs='+', help="Grid search values for fine-tuning learning rate.", default=None)
    parser.add_argument('--patience_search_values', type=int, nargs='+', help="Grid search values for early stopping patience.", default=None)
    parser.add_argument('--min_delta_search_values', type=float, nargs='+', help="Grid search values for early stopping min_delta.", default=None)

    parser.add_argument('--grid_search_csv_name', type=str, help="Name for the CSV file to store all grid search results.", default="grid_search_results.csv")
    args = parser.parse_args()

    is_grid_search_run = any([
        args.batch_size_values is not None,
        args.lr_search_values is not None,
        args.prune_biases_search_values is not None and args.pruning_method in ['magnitude', 'imp'],
        args.snip_minibatches_search_values is not None and args.pruning_method == 'snip',
        args.num_imp_steps_values is not None and args.pruning_method == 'imp',
        args.patience_search_values is not None,
        args.min_delta_search_values is not None
    ])

    BASE_DIR = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(BASE_DIR, args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.model == 'small_cnn' and not args.grayscale:
        print(f"INFO: Model '{args.model}' expects grayscale input. Forcing grayscale data loading.")
        args.grayscale = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == 'small_cnn': initial_model = load_small_cnn()
    elif args.model == 'vgg19': initial_model = load_vgg19()
    else:
        print(f"Model '{args.model}' is not recognized."); sys.exit(1)

    lr_values_to_search = args.lr_search_values if args.lr_search_values is not None else [args.lr]
    batch_size_values_to_search = args.batch_size_values if args.batch_size_values is not None else [args.batch_size]
    patience_values_to_search = args.patience_search_values if args.patience_search_values is not None else [args.patience]
    min_delta_values_to_search = args.min_delta_search_values if args.min_delta_search_values is not None else [args.min_delta]

    prune_biases_values_to_search = [False] # Default if not magnitude/imp or not specified
    if args.pruning_method in ['magnitude', 'imp']:
        if args.prune_biases_search_values is not None:
            prune_biases_values_to_search = [bool(val) for val in args.prune_biases_search_values]
        # else: it will use the default [False] or a single value if --prune_biases was a single arg (not implemented here)

    snip_minibatches_values_to_search = [1] # Default if not snip or not specified
    if args.pruning_method == 'snip':
        if args.snip_minibatches_search_values is not None:
            snip_minibatches_values_to_search = args.snip_minibatches_search_values

    num_imp_steps_values_to_search = [1] # Default if not imp or not specified (1 step = one-shot)
    if args.pruning_method == 'imp':
        if args.num_imp_steps_values is not None:
            num_imp_steps_values_to_search = args.num_imp_steps_values
        else: # If --pruning_method imp but no --num_imp_steps_values, use a sensible default like 5
            num_imp_steps_values_to_search = [5]


    hyperparam_combinations_tuples = list(itertools.product(
        lr_values_to_search,
        patience_values_to_search,
        min_delta_values_to_search,
        prune_biases_values_to_search,    # Relevant for magnitude, imp
        snip_minibatches_values_to_search,# Relevant for snip
        num_imp_steps_values_to_search,   # Relevant for imp
        batch_size_values_to_search
    ))

    all_run_details = []
    global_best_val_performance_per_fraction = {
        f: {'best_val_loss': float('inf'), 'val_acc_at_best_loss': 0.0,
            'hyperparams_tuple': None, 'hyperparams_str': "", 'seed': None,
            'epochs_finetuned': 0, 'pruning_method_for_best': None} # Added method
        for f in args.pruning_fractions
    }

    for combo_idx, current_hyperparams_tuple in enumerate(hyperparam_combinations_tuples):
        current_lr, current_patience, current_min_delta, \
        current_pb_setting, current_smb_setting, current_imp_steps, current_bs_setting = current_hyperparams_tuple

        # Skip irrelevant combinations
        if args.pruning_method not in ['magnitude', 'imp'] and current_pb_setting != prune_biases_values_to_search[0]:
            if len(prune_biases_values_to_search) > 1: continue
        if args.pruning_method != 'snip' and current_smb_setting != snip_minibatches_values_to_search[0]:
            if len(snip_minibatches_values_to_search) > 1: continue
        if args.pruning_method != 'imp' and current_imp_steps != num_imp_steps_values_to_search[0]:
            if len(num_imp_steps_values_to_search) > 1: continue


        current_hyperparams_str = f"BS:{current_bs_setting}, LR:{current_lr}, Pat:{current_patience}, MinDel:{current_min_delta}"
        if args.pruning_method in ['magnitude', 'imp']:
            current_hyperparams_str += f", PruneBiases:{current_pb_setting}"
        if args.pruning_method == 'snip':
            current_hyperparams_str += f", SNIP_MB:{current_smb_setting}"
        if args.pruning_method == 'imp':
            current_hyperparams_str += f", IMP_Steps:{current_imp_steps}"

        print(f"\n--- Grid Search Combo {combo_idx+1}/{len(hyperparam_combinations_tuples)} ({args.pruning_method}) ---")
        print(f"--- Running with Hyperparameters: {current_hyperparams_str} ---")

        results_this_hyperparam_set_for_summary_txt: dict[float, list[float]] = {f: [] for f in args.pruning_fractions}

        for seed in args.seeds:
            print(f"\n-- Seed: {seed} | {current_hyperparams_str} --")
            _seed_everything(seed)
            base_model_for_this_seed = copy.deepcopy(initial_model)

            dataloaders_info = get_cifar10_data(
                grayscale=args.grayscale, batch_size=current_bs_setting,
                val_split=args.val_split, num_workers=args.num_workers,
                data_root=args.data_root, device=device
            )
            train_loader, val_loader, test_loader = dataloaders_info['train'], dataloaders_info.get('val'), dataloaders_info['test']
            if not val_loader and args.val_split > 0: print("ERROR: val_split > 0 but no val_loader."); sys.exit(1)
            if not val_loader: print("INFO: No validation loader, early stopping in FT will not be active.")


            for fraction in args.pruning_fractions:
                model_to_process = copy.deepcopy(base_model_for_this_seed)
                print(f"\n-- Seed: {seed}, Method: {args.pruning_method}, Target Fraction: {fraction:.2f} | {current_hyperparams_str} --")

                test_acc, val_loss, val_acc, ft_epochs = -1.0, float('inf'), 0.0, 0

                if args.pruning_method == 'imp':
                    test_acc, val_loss, val_acc, ft_epochs = iterative_magnitude_prune_ft_eval(
                        initial_model_to_prune=model_to_process,
                        target_overall_pruning_fraction=fraction,
                        num_imp_steps=current_imp_steps,
                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device,
                        ft_max_epochs_per_step=args.max_epochs_safeguard, # Using main FT args
                        ft_lr_per_step=current_lr,
                        ft_patience_per_step=current_patience,
                        ft_min_delta_per_step=current_min_delta,
                        prune_bias=current_pb_setting
                    )
                elif args.pruning_method in ['magnitude', 'snip']:
                    test_acc, val_loss, val_acc, ft_epochs = prune_ft_eval(
                        model=model_to_process, fraction=fraction, prune_method=args.pruning_method,
                        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device,
                        max_epochs_safeguard=args.max_epochs_safeguard, finetune_lr=current_lr,
                        prune_bias=current_pb_setting, snip_minibatches=current_smb_setting,
                        patience=current_patience, min_delta=current_min_delta
                    )
                else:
                    raise ValueError(f"Unknown pruning method in main loop: {args.pruning_method}")

                results_this_hyperparam_set_for_summary_txt[fraction].append(test_acc * 100)
                print(f"  For Target Fraction {fraction:.2f} (Seed {seed}, {current_hyperparams_str}):")
                print(f"    Method {args.pruning_method}: Val Loss (final step for IMP): {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%, Total FT Epochs: {ft_epochs}")
                print(f"    Resulting Test Accuracy: {test_acc * 100:.2f}%")

                if val_loader is not None and val_loss < global_best_val_performance_per_fraction[fraction]['best_val_loss']:
                    global_best_val_performance_per_fraction[fraction].update({
                        'best_val_loss': val_loss, 'val_acc_at_best_loss': val_acc,
                        'hyperparams_tuple': current_hyperparams_tuple, 'hyperparams_str': current_hyperparams_str,
                        'seed': seed, 'epochs_finetuned': ft_epochs,
                        'pruning_method_for_best': args.pruning_method # Store the method
                    })
                    print(f"    NEW GLOBAL BEST validation performance for fraction {fraction:.2f} found!")

                run_detail = {
                    'model': args.model, 'pruning_method': args.pruning_method,
                    'num_imp_steps': current_imp_steps if args.pruning_method == 'imp' else None,
                    'batch_size': current_bs_setting, 'lr': current_lr, 'patience': current_patience, 'min_delta': current_min_delta,
                    'prune_biases': current_pb_setting if args.pruning_method in ['magnitude', 'imp'] else None,
                    'snip_mini_batches': current_smb_setting if args.pruning_method == 'snip' else None,
                    'seed': seed, 'fraction': fraction,
                    'test_accuracy_immediate': test_acc * 100,
                    'best_val_loss_ft': val_loss if val_loader else None,
                    'val_acc_at_best_loss_ft': val_acc * 100 if val_loader else None,
                    'epochs_finetuned_ft': ft_epochs
                }
                all_run_details.append(run_detail)

        summary_filename = f"summary_{args.model}_{args.pruning_method}_LR{current_lr}_BS{current_bs_setting}_Pat{current_patience}"
        if args.pruning_method == 'imp': summary_filename += f"_IMPSteps{current_imp_steps}"
        summary_filename += ".txt"
        summary_path = os.path.join(output_dir, summary_filename)
        with open(summary_path, 'w') as f:
            f.write(f"Hyperparameters: {current_hyperparams_str}\n")
            seed_headers = "\t".join(str(s) for s in args.seeds)
            f.write(f"Fraction\t{seed_headers}\tMean Test Acc\tStd Test Acc\n")
            for fraction_val, accs in results_this_hyperparam_set_for_summary_txt.items():
                if not accs: continue
                mean = sum(accs) / len(accs) if accs else 0
                std = math.sqrt(sum((a - mean) ** 2 for a in accs) / len(accs)) if len(accs) > 0 else 0
                # Pre-format the list of accuracies
                accs_formatted_list = [f"{a:.2f}" for a in accs]
                accs_joined_str = "\t".join(accs_formatted_list)
                # Use the pre-formatted string in the final f-string
                f.write(f"{fraction_val:.2f}\t{accs_joined_str}\t{mean:.2f}\t{std:.2f}\n")
        print(f"Intermediate summary for {current_hyperparams_str} saved to {summary_path}")

    if all_run_details:
        pd.DataFrame(all_run_details).to_csv(os.path.join(output_dir, args.grid_search_csv_name), index=False)
        print(f"\nAll grid search run details saved to {os.path.join(output_dir, args.grid_search_csv_name)}")
    else: print("\nNo results generated to save to CSV.")

    print("\n\n--- Grid Search Complete ---")
    print("--- Summary of Best Hyperparameters (Selected by Validation Loss) per Pruning Fraction ---")
    final_results_summary = []

    for fraction, best_info in global_best_val_performance_per_fraction.items():
        if best_info['hyperparams_tuple'] is None:
            print(f"\nFraction {fraction:.2f}: No valid configuration found."); continue

        print(f"\nFraction {fraction:.2f}: Method: {best_info['pruning_method_for_best']}")
        print(f"  Best Val Loss: {best_info['best_val_loss']:.4f}, Val Acc: {best_info['val_acc_at_best_loss'] * 100:.2f}%")
        print(f"  Achieved with Seed: {best_info['seed']}, FT Epochs: {best_info['epochs_finetuned']}")
        print(f"  Hyperparameters: {best_info['hyperparams_str']}")
        print(f"  Re-evaluating on Test Set for Fraction {fraction:.2f} with best validation-selected config...")

        eval_lr, eval_patience, eval_min_delta, eval_pb, eval_smb, eval_imp_steps, eval_bs = best_info['hyperparams_tuple']
        eval_seed = best_info['seed']
        eval_method = best_info['pruning_method_for_best']

        _seed_everything(eval_seed)
        eval_model_base = copy.deepcopy(initial_model)
        eval_dataloaders_info = get_cifar10_data(grayscale=args.grayscale, batch_size=eval_bs, val_split=args.val_split, num_workers=args.num_workers, data_root=args.data_root, device=device)
        eval_train_loader, eval_val_loader, eval_test_loader = eval_dataloaders_info['train'], eval_dataloaders_info.get('val'), eval_dataloaders_info['test']

        final_test_acc, final_val_loss, final_val_acc, final_ft_epochs = -1.0, float('inf'), 0.0, 0

        if eval_method == 'imp':
            final_test_acc, final_val_loss, final_val_acc, final_ft_epochs = iterative_magnitude_prune_ft_eval(
                initial_model_to_prune=eval_model_base, target_overall_pruning_fraction=fraction, num_imp_steps=eval_imp_steps,
                train_loader=eval_train_loader, val_loader=eval_val_loader, test_loader=eval_test_loader, device=device,
                ft_max_epochs_per_step=args.max_epochs_safeguard, ft_lr_per_step=eval_lr,
                ft_patience_per_step=eval_patience, ft_min_delta_per_step=eval_min_delta, prune_bias=eval_pb
            )
        elif eval_method in ['magnitude', 'snip']:
            final_test_acc, final_val_loss, final_val_acc, final_ft_epochs = prune_ft_eval(
                model=eval_model_base, fraction=fraction, prune_method=eval_method,
                train_loader=eval_train_loader, val_loader=eval_val_loader, test_loader=eval_test_loader, device=device,
                max_epochs_safeguard=args.max_epochs_safeguard, finetune_lr=eval_lr,
                prune_bias=eval_pb, snip_minibatches=eval_smb, patience=eval_patience, min_delta=eval_min_delta
            )
        print(f"  FINAL Test Accuracy for Fraction {fraction:.2f} (Val-Selected Config): {final_test_acc * 100:.2f}%")
        final_results_summary.append({
            'fraction': fraction, 'pruning_method': eval_method,
            'best_val_loss': best_info['best_val_loss'], 'val_acc_at_best_loss': best_info['val_acc_at_best_loss'] * 100,
            'selected_hyperparams_str': best_info['hyperparams_str'], 'selected_seed': best_info['seed'],
            'selected_ft_epochs': best_info['epochs_finetuned'], 'final_reported_test_accuracy': final_test_acc * 100
        })

    if final_results_summary:
        final_summary_df = pd.DataFrame(final_results_summary)
        final_summary_csv_path = os.path.join(output_dir, f"final_best_configs_summary_{args.model}.csv") # General name
        final_summary_df.to_csv(final_summary_csv_path, index=False)
        print(f"\nSummary of final best configurations and their test accuracies saved to: {final_summary_csv_path}")

if __name__ == "__main__":
    main()
