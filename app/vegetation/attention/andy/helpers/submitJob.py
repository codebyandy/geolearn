from hydroDL import kPath
from clean_runs_dir import delete_crashed_subdirs

from itertools import product
import argparse
import os

DEFAULT_METHODS = ["cherry"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_DROPOUTS = [0.6]
DEFAULT_EMBEDDING_SIZES = [64]
DEFAULT_BATCH_SIZES = [500]
DEFAULT_OPTIMIZERS = ["adam"]
DEFAULT_LEARNING_RATES = [0.01]
DEFAULT_ITERS_PER_EPOCH = [20]
DEFAULT_SCHED_START_EPOCHS = [200]

def submitJob(jobName, cmdLine, nH=24, nM=16):
    jobFile = os.path.join(kPath.dirJob, jobName)
    with open(jobFile, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines('#SBATCH --job-name={}\n'.format(jobName))
        fh.writelines('#SBATCH --output={}.out\n'.format(jobFile))
        fh.writelines('#SBATCH --error={}.err\n'.format(jobFile))
        fh.writelines('#SBATCH --time={}:0:0\n'.format(nH))
        fh.writelines('#SBATCH --mem={}000\n'.format(nM))
        fh.writelines('#SBATCH --mail-type=ALL\n')
        fh.writelines('#SBATCH --mail-user=avhuynh@stanford.edu\n')
        if kPath.host == 'icme':
            fh.writelines('source activate pytorch\n')
        elif kPath.host == 'sherlock':
            fh.writelines('source /home/users/avhuynh/envs/pytorch/bin/activate\n')
        fh.writelines(cmdLine)
    os.system('sbatch {}'.format(jobFile))

def submitJobGPU(jobName, cmdLine, nH=24, nM=16):
    jobFile = os.path.join(kPath.dirJob, jobName)
    with open(jobFile, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines('#SBATCH -p gpu\n')
        fh.writelines('#SBATCH -G 1\n')
        fh.writelines('#SBATCH --job-name={}\n'.format(jobName))
        fh.writelines('#SBATCH --output={}.out\n'.format(jobFile))
        fh.writelines('#SBATCH --error={}.err\n'.format(jobFile))
        fh.writelines('#SBATCH --time={}:0:0\n'.format(nH))
        fh.writelines('#SBATCH --mem={}000\n'.format(nM))
        fh.writelines('#SBATCH --qos=normal\n')
        fh.writelines('#SBATCH -C "GPU_SKU:P100_PCIE|GPU_SKU:RTX_2080Ti|GPU_SKU:V100_PCIE|GPU_SKU:V100S_PCIE|GPU_SKU:V100_SXM2"')
        fh.writelines('#SBATCH --mail-type=ALL\n')
        fh.writelines('#SBATCH --mail-user=kuaifang@stanford.edu\n')
        if kPath.host == 'icme':
            fh.writelines('source activate pytorch\n')
        elif kPath.host == 'sherlock':
            fh.writelines('source /home/users/kuaifang/envs/pytorch/bin/activate\n')
        fh.writelines('hostname\n')
        fh.writelines('nvidia-smi -L\n')
        fh.writelines(cmdLine)
    os.system('sbatch {}'.format(jobFile))

def main(args):
    parent_directory = os.path.join(kPath.dirVeg, "runs")
    delete_crashed_subdirs(parent_directory)

    methods_lst = args.methods
    seeds_lst = args.seeds
    dropouts_lst = args.dropouts
    embedding_sizes_lst = args.embedding_sizes
    batch_sizes_lst = args.batch_sizes
    optimizers_lst = args.optimizers
    learning_rates_lst = args.learning_rates
    iters_per_epoch_lst = args.iters_per_epoch
    sched_start_epochs_lst = args.sched_start_epochs
    wandb_name = args.wandb_name
    note = args.note
    split_version = args.split_version
    dataset = args.dataset
    cross_val = args.cross_val

    hyperparam_combos = list(product(methods_lst, seeds_lst, dropouts_lst, embedding_sizes_lst, batch_sizes_lst, \
                                     optimizers_lst, learning_rates_lst, iters_per_epoch_lst, sched_start_epochs_lst))

    for i, (method, seed, dropout, embedding_size, batch_size, optimizer, learning_rate, iters_per_epoch, sched_start_epoch) in enumerate(hyperparam_combos):
        if note:
            run_name = f'{note}_{embedding_size}_{dropout}_{seed}'
        else:
            run_name = f'{embedding_size}_{dropout}_{seed}'
        train_path = f'/home/users/avhuynh/lfmc/geolearn/app/vegetation/attention/andy/src/models/{method}_pick/train.py'
        cmd_line = f'python {train_path} --run_name {run_name} --dropout {dropout} --nh {embedding_size} --batch_size {batch_size} --seed {seed}' 
        cmd_line += f' --optimizer {optimizer} --learning_rate {learning_rate} --iters_per_epoch {iters_per_epoch} --sched_start_epoch {sched_start_epoch}'
        cmd_line += f' --epochs 500 --satellites no_landsat --wandb_name {wandb_name}'
        cmd_line += f' --split_version {split_version} --dataset {dataset} --cross_val {cross_val}'
        
        print(i, cmd_line)
        submitJob(run_name, cmd_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--methods", type=list, default=DEFAULT_METHODS)
    parser.add_argument("--seeds", type=list, default=DEFAULT_SEEDS)
    parser.add_argument("--dropouts", type=list, default=DEFAULT_DROPOUTS)
    parser.add_argument("--embedding_sizes", type=list, default=DEFAULT_EMBEDDING_SIZES)
    parser.add_argument("--batch_sizes", type=list, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--optimizers", type=list, default=DEFAULT_OPTIMIZERS)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--iters_per_epoch", type=list, default=DEFAULT_ITERS_PER_EPOCH)
    parser.add_argument("--sched_start_epochs", type=list, default=DEFAULT_SCHED_START_EPOCHS)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--split_version", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--cross_val", type=bool, default=True)
    args = parser.parse_args()

    main(args)
