Note: This repository is partially adapted from ORIGINAL PAPER.

# Neural Network Pruning with Scale Equivarient Graph Metanetworks (ScaleGMN)
Repository for 'Pruning with Scale Equivariant Graph Metanetworks' (June 2025)

by Freek Byrman, Tobias Groot, Bart Kuipers and Daniel Uyterlinde

link to paper

## Abstract
_insert abstract_

# **Section 1: Reproduction original experiments**

### Setup

To create a clean virtual environment and install the necessary dependencies execute:
```bash
git clone git@github.com:daniuyter/scalegmn-pruning.git
cd ScaleGMN-final/
conda env create -n scalegmn --file environment.yml
conda activate scalegmn
```

### Data
First, create the `data/` directory in the root of the repository:
```bash
mkdir data
````
Alternatively, you can specify a different directory for the data by changing
the corresponding fields in the config file.

### INR Classification and Editing
For the INR dataset, we use the data provided by [DWS](https://github.com/AvivNavon/DWSNets) and [NFN](https://github.com/AllanYangZhou/nfn/).
The datasets can be downloaded from the following links: 

```bash
DATA_DIR=./data
wget "https://www.dropbox.com/sh/56pakaxe58z29mq/AABrctdu2U65jGYr2WQRzmMna/mnist-inrs.zip?dl=0" -O "$DATA_DIR/mnist-inrs.zip"
unzip -q "$DATA_DIR/mnist-inrs.zip" -d "$DATA_DIR"
rm "$DATA_DIR/mnist-inrs.zip" # remove the zip file
# generate the splits
python src/utils/generate_data_splits.py --data_path $DATA_DIR/mnist-inrs --save_path $DATA_DIR/mnist-inrs
```

#### Phase canonicalization
For the INR dataset, we preprocess each datapoint to canonicalize the phase symmetry (see [Algorithm 1](https://arxiv.org/pdf/2406.10685v1#algocf.1) in the appendix).
To run the phase canonicalization script, run the following command:

```bash
python src/phase_canonicalization/canonicalization.py --conf src/phase_canonicalization/mnist.yml
```

The above script will store the canonicalized dataset in a new directory `data/<dataset>_canon/`. The training scripts will automatically use the canonicalized dataset, if it exists.
To use the dataset specified in the config file (and not search for `data/<dataset>_canon/`), set the `data.switch_to_canon` field of the config to `False` or simply use the CLI argument `--data.switch_to_canon False`. 

### Generalization prediction
We follow the experiments from [NFN](https://github.com/AllanYangZhou/nfn/) and use the datasets provided by [Unterthiner et al,
2020](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy). The dataset can be downloaded from the following links:
- [CIFAR10](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz)

Similarly, extract the dataset in the directory `data/` and execute:

For the CIFAR10 dataset:
```bash
tar -xvf cifar10.tar.xz
# download cifar10 splits
wget https://github.com/AllanYangZhou/nfn/raw/refs/heads/main/experiments/predict_gen_data_splits/cifar10_split.csv -O data/cifar10/cifar10_split.csv
```

### Experiments
For every experiment, we provide the corresponding configuration file in the `config/` directory.
Each config contains the selected hyperparameters for the experiment, as well as the paths to the dataset.
To enable wandb logging, use the CLI argument `--wandb True`. For more useful CLI arguments, check the [src/utils/setup_arg_parser.py](src/utils/setup_arg_parser.py) file.

**Note:** To employ a GMN accounting only for the permutation symmetries, simply set 
`--scalegmn_args.symmetry=permutation`.

### INR Editing
To train and evaluate ScaleGMN on the INR editing task, use the configs under
[configs/mnist_editing](configs/mnist_editing) directory and execute:

```bash
python inr_editing.py --conf configs/mnist_editing/scalegmn_bidir.yml
```

### Generalization prediction
To train and evaluate ScaleGMN on the INR classification task, 
select any config file under [configs/cifar10](configs/cifar10)
or [configs/svhn](configs/svhn). For example, to 
train ScaleGMN on the CIFAR10 dataset on heterogeneous activation functions,
execute the following:

```bash
python predicting_generalization.py --conf configs/cifar10/scalegmn_hetero.yml
```


# **Section 2: Pruning with ScaleGMN**

### Invariant Pruning
First, make sure to set your preferred pruning settings in the config file:

```bash
configs/cifar10/scalegmn_relu.yml
```
The values in this file are the original values we used for our experiments.

To run our Invariant Pruning method, the best way is to run the following file:
 
```bash
gradual_pruning.py --conf configs/cifar10/scalegmn_relu.yml
```

This will gradually prune (at sparsities 60-70-80-90%) and finetune the best CNN from SmallCNNZoo on the CIFAR10-GS split, which we discussed in the paper. The pruned models will be selected based on best validation accuracy, and will be saved to the directory from where you can load and evaluate them on CIFAR10-GS.

To evaluate the pruned models, run the following command:

```bash
python evaluate_model.py --conf configs/cifar10/scalegmn_relu.yml --model_path <model path>
```

### Equivariant Pruning


# Citation

```bib
@article{kalogeropoulos2024scale,
    title={Scale Equivariant Graph Metanetworks},
    author={Kalogeropoulos, Ioannis and Bouritsas, Giorgos and Panagakis, Yannis},
    journal={Advances in Neural Information Processing Systems},
    year={2024}
}
```
