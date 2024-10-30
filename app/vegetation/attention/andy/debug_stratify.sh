# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_lower_dropouts --dropouts 0,0.2,0.4 # 0 or 0.2
# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_lower_lr --learning_rates 0.001
# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_optimizer_sgd2 --optimizer sgd # stick wtih adam
# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_lower_batch_size_32 --batch_size 32 --iters_per_epoch 200
# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_lower_batch_size_64 --batch_size 64 --iters_per_epoch 100
# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_lower_batch_size_128 --batch_size 128 --iters_per_epoch 50
# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_lower_batch_size_256 --batch_size 256 --iters_per_epoch 25 # this is good!
# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_embedding_sizes --embedding_sizes 32,16 # 32 is probably good!
# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_sched_start_epoch_0 --sched_start_epoch 0 # this is good!
# python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_sched_start_epoch_100 --sched_start_epoch 100 

# python helpers/submitJob.py --wandb_name debugging --exp_name new_splits --run_name stratified --split_version stratified
# python helpers/submitJob.py --wandb_name debugging --exp_name new_splits --run_name dataset --split_version dataset
# python helpers/submitJob.py --wandb_name debugging --exp_name new_splits --run_name stratified_s0 --split_version stratified_s0
# python helpers/submitJob.py --wandb_name debugging --exp_name new_splits --run_name stratified_s1 --split_version stratified_s1
# python helpers/submitJob.py --wandb_name debugging --exp_name new_splits --run_name random_s0 --split_version random_s0
# python helpers/submitJob.py --wandb_name debugging --exp_name new_splits --run_name random_s1 --split_version random_s1


python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_handpick --sched_start_epoch 0 --embedding_sizes 32,48 --batch_size 256 --iters_per_epoch 25 --dropouts 0,0.2,0.1


python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_handpick2_start0 --sched_start_epoch 0 --embedding_sizes 64 --batch_size 256 --iters_per_epoch 25 --dropouts 0.0,0.01 --learning_rate 0.001
python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_handpick2_start100 --sched_start_epoch 100 --embedding_sizes 64 --batch_size 256 --iters_per_epoch 25 --dropouts 0.0,0.01 --learning_rate 0.001
python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_handpick2_start200 --sched_start_epoch 200 --embedding_sizes 64 --batch_size 256 --iters_per_epoch 25 --dropouts 0.0,0.01 --learning_rate 0.001


python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version stratified_s0 --run_name stratified_s0_handpick3 --dropouts 0.1  --embedding_sizes 48,64 --batch_size 256 --iters_per_epoch 25 --loss_fn mse


python helpers/submitJob.py --wandb_name debugging --exp_name stratify --split_version random_s0 --run_name stratified_s0_handpick3 --dropouts 0.1  --embedding_sizes 64 --batch_size 256 --iters_per_epoch 25 --loss_fn mse 

