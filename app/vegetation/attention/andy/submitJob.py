from hydroDL import kPath
from itertools import product

import argparse
import os


DEFAULT_METHODS = ["all", "cherry"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_DROPOUTS = [0.2, 0.4, 0.6]
DEFAULT_EMBEDDING_SIZES = [32, 64, 128]
DEFAULT_BATCH_SIZES = [500, 1000, 2000]


def submitJob(jobName, cmdLine, nH=24, nM=16):
    jobFile = os.path.join(kPath.dirJob, jobName)
    with open(jobFile, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines('#SBATCH --job-name={}\n'.format(jobName))
        fh.writelines('#SBATCH --output={}.out\n'.format(jobFile))
        fh.writelines('#SBATCH --error={}.err\n'.format(jobFile))
        fh.writelines('#SBATCH --time={}:0:0\n'.format(nH))
        fh.writelines('#SBATCH --mem={}000\n'.format(nM))
        # fh.writelines('#SBATCH --qos=normal\n')
        # fh.writelines('#SBATCH --partition=owners\n')
        fh.writelines('#SBATCH --mail-type=ALL\n')        
        fh.writelines('#SBATCH --mail-user=avhuynh@stanford.edu\n')
        if kPath.host == 'icme':
            fh.writelines('source activate pytorch\n')
        elif kPath.host == 'sherlock':
            fh.writelines(
                'source /home/users/avhuynh/envs/pytorch/bin/activate\n')
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
        # fh.writelines('#SBATCH --partition=owners\n')
        fh.writelines('#SBATCH --mail-type=ALL\n')
        fh.writelines('#SBATCH --mail-user=kuaifang@stanford.edu\n')
        if kPath.host == 'icme':
            fh.writelines('source activate pytorch\n')
        elif kPath.host == 'sherlock':
            fh.writelines(
                'source /home/users/kuaifang/envs/pytorch/bin/activate\n')
        fh.writelines('hostname\n')
        fh.writelines('nvidia-smi -L\n')
        fh.writelines(cmdLine)
    os.system('sbatch {}'.format(jobFile))


def main(args):
    methods_lst = args.methods
    seeds_lst = args.seeds
    dropouts_lst = args.dropouts
    embedding_sizes_lst = args.embedding_sizes
    batch_sizes_lst = args.batch_sizes

    hyperparam_combos = list(product(methods_lst, seeds_lst, dropouts_lst, embedding_sizes_lst, batch_sizes_lst))

    for i, (method, seed, dropout, embedding_size, batch_size) in enumerate(hyperparam_combos):
        # print(method, seed, dropout, embedding_size)
        run_name = f'{method}_{embedding_size}_{dropout}_{batch_size}'
        train_path = f'/home/users/avhuynh/lfmc/geolearn/app/vegetation/attention/andy/src/models/{method}_pick/train.py'
        cmd_line = f'python {train_path} --run_name {run_name} --dropout {dropout} --nh {embedding_size} --batch_size {batch_size} --seed {seed} --epochs 1000 --dataset singleDaily-nadgrid --satellites no_landsat'
        if method == 'cherry':
            cmd_line += ' --test_epoch 25'
        print(cmd_line)
        submitJob(run_name, cmd_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--methods", type=list, default=DEFAULT_METHODS)
    parser.add_argument("--seeds", type=list, default=DEFAULT_SEEDS)
    parser.add_argument("--dropouts", type=list, default=DEFAULT_DROPOUTS)
    parser.add_argument("--embedding_sizes", type=list, default=DEFAULT_EMBEDDING_SIZES)
    parser.add_argument("--batch_sizes", type=list, default=DEFAULT_BATCH_SIZES)
    args = parser.parse_args()

    main(args)