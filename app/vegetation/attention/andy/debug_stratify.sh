python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name lower_dropouts2 --dropouts 0,0.2,0.4 
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name lower_batch_size --batch_size
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name lower_lr --learning_rates
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name optimizer_sgd2 --optimizer sgd
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name lower_batch_size_32 --batch_size 32 --iters_per_epoch 200
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name lower_batch_size_64 --batch_size 64 --iters_per_epoch 100
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name lower_batch_size_128 --batch_size 128 --iters_per_epoch 50
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name lower_batch_size_256 --batch_size 256 --iters_per_epoch 25
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name embedding_sizes --embedding_sizes 32,16
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name sched_start_epoch_0 --sched_start_epoch 0
python helpers/submitJob.py --wandb_name experiments --exp_name stratify --run_name sched_start_epoch_100 --sched_start_epoch 100 