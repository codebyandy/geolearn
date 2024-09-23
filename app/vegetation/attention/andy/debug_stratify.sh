python helpers/submitJob.py --wandb_name default --exp_name stratify --run_name lower_dropouts --dropouts 0,0.2,0.4 

python helpers/submitJob.py --wandb_name default --exp_name stratify --run_name lower_batch_size --batch_sizes

python helpers/submitJob.py --wandb_name default --exp_name stratify --run_name lower_lr --learning_rates

python helpers/submitJob.py --wand_name default --exp_name stratify --run_name optimizer --optimizers adam,sgd