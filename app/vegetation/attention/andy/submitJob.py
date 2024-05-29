import os
from hydroDL import kPath


def submitJob(jobName, cmdLine, nH=8, nM=16):
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


def submitJobGPU(jobName, cmdLine, nH=8, nM=16):
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

    
dropout_lst = [0.2, 0.4, 0.6]
nh_lst = [24, 32, 64, 128, 256]

# for dropout in dropout_lst:
#     for nh in nh_lst:
#         run_name = f'cherry_pick_do_{dropout}_nh_{nh}'
#         train_path = '/home/users/avhuynh/lfmc/geolearn/app/vegetation/attention/andy/src/train.py'
#         cmd_line = f'python {train_path} --run_name {run_name} --dropout {dropout} --nh {nh} --epochs 1000 --dataset singleDaily-nadgrid --satellites no_landsat'
#         submitJob(run_name, cmd_line)

# for dropout in dropout_lst:
#     for nh in nh_lst:
#         run_name = f'all_pick_do_{dropout}_nh_{nh}'
#         train_path = '/home/users/avhuynh/lfmc/geolearn/app/vegetation/attention/andy/src/train_no_sampling.py'
#         cmd_line = f'python {train_path} --run_name {run_name} --dropout {dropout} --nh {nh} --epochs 1000 --dataset singleDaily-nadgrid --satellites no_landsat --test_epoch 10'
#         submitJob(run_name, cmd_line, nH=24)

for dropout in dropout_lst:
    for nh in nh_lst:
        run_name = f'500m_no_landsat_do_{dropout}_nh_{nh}'
        train_path = '/home/users/avhuynh/lfmc/geolearn/app/vegetation/attention/andy/src/inference.py'
        cmd_line = f'python {train_path} --model_dir {run_name}'
        submitJob(run_name, cmd_line, nH=1)

for dropout in dropout_lst:
    for nh in nh_lst:
        run_name = f'all_pick_do_{dropout}_nh_{nh}'
        train_path = '/home/users/avhuynh/lfmc/geolearn/app/vegetation/attention/andy/src/inference_all_pick.py'
        cmd_line = f'python {train_path} --model_dir {run_name}'
        submitJob(run_name, cmd_line, nH=1)