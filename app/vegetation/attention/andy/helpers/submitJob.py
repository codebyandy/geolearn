from hydroDL import kPath

from itertools import product
import argparse
import os


DEFAULT_METHODS = 'cherry'
DEFAULT_SEEDS = '0,1,2'
DEFAULT_DROPOUTS = '0.5'
DEFAULT_EMBEDDING_SIZES = '32'
DEFAULT_LEARNING_RATES = '0.01'
DEFAULT_FOLDS = '0,1,2,3,4'


def str_to_lst(str, type):
    if type == 'int':
        return [int(s) for s in str.strip().split(',')]
    elif type ==  'float':
        return [float(s) for s in str.strip().split(',')]
    elif type == 'str':
        return [s for s in str.strip().split(',')]
    else:
        raise Exception(f'str_to_lst does not suppor type {type}.')


def submitJob(jobName, cmdLine, nH=24, nM=16):
    jobFile = os.path.join(kPath.dirJob, jobName)
    with open(jobFile, 'w') as fh:
        fh.writelines('#!/bin/bash\n')
        fh.writelines('#SBATCH --job-name={}\n'.format(jobName))
        fh.writelines('#SBATCH --output={}.out\n'.format(jobFile))
        fh.writelines('#SBATCH --error={}.err\n'.format(jobFile))
        fh.writelines('#SBATCH --time={}:0:0\n'.format(nH))
        fh.writelines('#SBATCH --mem={}000\n'.format(nM))
        fh.writelines('#SBATCH --mail-type=ALL\n')
        fh.writelines('#SBATCH --mail-user=avhuynh@stanford.edu\n')
        if kPath.host == 'sherlock':
            fh.writelines('source /home/users/avhuynh/envs/pytorch/bin/activate\n')
        fh.writelines(cmdLine)
    os.system('sbatch {}'.format(jobFile))


def main(args):
    methods_lst = str_to_lst(args.methods, 'str')
    seeds_lst = str_to_lst(args.seeds, 'int')
    dropouts_lst = str_to_lst(args.dropouts, 'float')
    embedding_sizes_lst = str_to_lst(args.embedding_sizes, 'int')
    learning_rates_lst = str_to_lst(args.learning_rates, 'float')
    folds_lst = str_to_lst(args.folds, 'int')

    wandb_name = args.wandb_name
    run_name = args.run_name
    exp_name = args.exp_name
    split_version = args.split_version
    dataset = args.dataset
    test_epoch = args.test_epoch

    batch_size = args.batch_size
    loss_fn = args.loss_fn
    optimizer = args.optimizer
    iters_per_epoch = args.iters_per_epoch
    sched_start_epoch = args.sched_start_epoch
    epochs = args.epochs
    weight_decay = args.weight_decay

    hyperparam_combos = list(product(methods_lst, seeds_lst, dropouts_lst, embedding_sizes_lst, learning_rates_lst))
    
    if args.protection:
        for i, (method, seed, dropout, embedding_size, learning_rate) in enumerate(hyperparam_combos):
            run_name_details = f'{run_name}_{method}_{embedding_size}_{dropout}_{learning_rate}_{seed}'
            save_path = os.path.join(kPath.dirVeg, 'runs', run_name_details)
            print("Saving to path ", save_path)
            if os.path.exists(save_path):
                raise Exception(f'run_name {run_name_details} already exists!')

    for i, (method, seed, dropout, embedding_size, learning_rate) in enumerate(hyperparam_combos):
        run_name_details = f'{run_name}_{method}_{embedding_size}_{dropout}_{learning_rate}_{seed}'
        print('=>', run_name, run_name_details)
        for fold in folds_lst:
            train_path = f'/home/users/avhuynh/lfmc/geolearn/app/vegetation/attention/andy/src/models/{method}_pick/train.py'
            cmd_line = f'python {train_path} --run_name {run_name_details} --dropout {dropout} --nh {embedding_size} --batch_size {batch_size} --seed {seed}' 
            cmd_line += f' --optimizer {optimizer} --loss_fn {loss_fn} --learning_rate {learning_rate} --iters_per_epoch {iters_per_epoch} --sched_start_epoch {sched_start_epoch}'
            cmd_line += f' --epochs {epochs}  --wandb_name {wandb_name} --exp_name {exp_name} --weight_decay {weight_decay}'
            cmd_line += f' --split_version {split_version} --dataset {dataset} --test_epoch {test_epoch} --fold {fold} --epochs {epochs}'
            submitJob(run_name, cmd_line)
            # print(run_name, cmd_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--git_branch', type=str, default='master')
    parser.add_argument('--methods', default=DEFAULT_METHODS)
    parser.add_argument('--seeds', default=DEFAULT_SEEDS)
    parser.add_argument('--dropouts', default=DEFAULT_DROPOUTS)
    parser.add_argument('--embedding_sizes', default=DEFAULT_EMBEDDING_SIZES)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--loss_fn', default='l1', choices=['l1', 'mse'])
    parser.add_argument('--learning_rates', default=DEFAULT_LEARNING_RATES)
    parser.add_argument('--iters_per_epoch', default=25)
    parser.add_argument('--sched_start_epoch', default=200)
    parser.add_argument('--wandb_name', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--split_version', type=str, default='stratified')
    parser.add_argument('--dataset', type=str, default='singleDaily-modisgrid-new-const')
    parser.add_argument('--test_epoch', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--folds', default=DEFAULT_FOLDS)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--protection', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    main(args)
