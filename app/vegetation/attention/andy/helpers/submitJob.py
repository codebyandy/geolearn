# Cancel all
# squeue -u $USER | awk '{print $1}' | tail -n+2 | xargs scancel

from hydroDL import kPath

from itertools import product
import argparse
import os


DEFAULT_METHODS = 'cherry'
DEFAULT_SEEDS = '0,1,2'
DEFAULT_DROPOUTS = '0.6'
DEFAULT_EMBEDDING_SIZES = '64'
DEFAULT_LEARNING_RATES = '0.01'


def str_to_lst(str, type):
    if type == 'int':
        return [int(s) for s in str.strip().split(',')]
    elif type ==  'float':
        return [float(s) for s in str.strip().split(',')]
    elif type == 'str':
        return [s for s in str.strip().split(',')]
    else:
        raise Exception(f"str_to_lst does not suppor type {type}.")


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
    methods_lst = str_to_lst(args.methods, 'str')
    seeds_lst = str_to_lst(args.seeds, 'int')
    dropouts_lst = str_to_lst(args.dropouts, 'float')
    embedding_sizes_lst = str_to_lst(args.embedding_sizes, 'int')
    learning_rates_lst = str_to_lst(args.learning_rates, 'float')

    wandb_name = args.wandb_name
    run_name = args.run_name
    split_version = args.split_version
    dataset = args.dataset
    cross_val = args.cross_val
    test_epoch = args.test_epoch

    batch_size = args.batch_size
    optimizer = args.optimizer
    iters_per_epoch = args.iters_per_epoch
    sched_start_epoch = args.sched_start_epoch
    epochs = args.epochs

    hyperparam_combos = list(product(methods_lst, seeds_lst, dropouts_lst, embedding_sizes_lst, learning_rates_lst))
    
    for i, (method, seed, dropout, embedding_size, learning_rate) in enumerate(hyperparam_combos):
        run_name_details = f'{run_name}_{method}_{embedding_size}_{dropout}_{learning_rate}_{seed}'
        save_path = os.path.join(kPath.dirVeg, 'runs', run_name_details)
        print(save_path)
        if os.path.exists(save_path):
            raise Exception(f"run_name {run_name_details} already exists!")
    
    for i, (method, seed, dropout, embedding_size, learning_rate) in enumerate(hyperparam_combos):
        run_name_details = f'{run_name}_{method}_{embedding_size}_{dropout}_{learning_rate}_{seed}'
        print('Combo', i, 'run_name', run_name_details)
        for fold in range(5):
            train_path = f'/home/users/avhuynh/lfmc/geolearn/app/vegetation/attention/andy/src/models/{method}_pick/train.py'
            cmd_line = f'python {train_path} --run_name {run_name} --dropout {dropout} --nh {embedding_size} --batch_size {batch_size} --seed {seed}' 
            cmd_line += f' --optimizer {optimizer} --learning_rate {learning_rate} --iters_per_epoch {iters_per_epoch} --sched_start_epoch {sched_start_epoch}'
            cmd_line += f' --epochs {epochs} --satellites no_landsat --wandb_name {wandb_name}'
            cmd_line += f' --split_version {split_version} --dataset {dataset} --cross_val {cross_val} --test_epoch {test_epoch} --fold {fold}'
            print(' Submitted fold', fold)
            submitJob(run_name, cmd_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--dropouts", default=DEFAULT_DROPOUTS)
    parser.add_argument("--embedding_sizes", default=DEFAULT_EMBEDDING_SIZES)
    parser.add_argument("--batch_size", default=500)
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--learning_rates", default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--iters_per_epoch", default=20)
    parser.add_argument("--sched_start_epoch", default=200)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--split_version", type=str, default='stratified', choices=["dataset", "stratified"])
    parser.add_argument("--dataset", type=str, default="singleDaily-modisgrid-new-const")
    parser.add_argument("--cross_val", type=bool, default=True)
    parser.add_argument("--test_epoch", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    main(args)
