# python helpers/submitJob.py --wandb_name debugging --run_name OLD_modisgrid-new-const --split_version dataset --dataset singleDaily-modisgrid-new-const 
# python helpers/submitJob.py --wandb_name debugging --run_name OLD_modisgrid-new-const-strat --split_version stratified --dataset singleDaily-modisgrid-new-const 

# python helpers/submitJob.py --wandb_name debugging --run_name OLD_nadgrid --split_version dataset --dataset singleDaily-nadgrid 
# python helpers/submitJob.py --wandb_name debugging --run_name OLD_nadgrid-strat --split_version stratified --dataset singleDaily-nadgrid

# python helpers/submitJob.py --wandb_name debugging --run_name lower_dropout_1seed --split_version dataset --dataset singleDaily-nadgrid --dropout 0.4 --seed 0
# python helpers/submitJob.py --wandb_name debugging --run_name lower_dropout_1seed --split_version dataset --dataset singleDaily-nadgrid --dropout 0.2 --seed 0
# python helpers/submitJob.py --wandb_name debugging --run_name lower_dropout_1seed --split_version dataset --dataset singleDaily-nadgrid --dropout 0.0 --seed 0

# python helpers/submitJob.py --wandb_name debugging --run_name lower_dropout_1seed_strat --split_version stratified --dataset singleDaily-nadgrid --dropout 0.4 --seed 0
# python helpers/submitJob.py --wandb_name debugging --run_name lower_dropout_1seed_strat --split_version stratified --dataset singleDaily-nadgrid --dropout 0.2 --seed 0
# python helpers/submitJob.py --wandb_name debugging --run_name lower_dropout_1seed_strat --split_version stratified --dataset singleDaily-nadgrid --dropout 0.0 --seed 0

srun --nodes=1 --time=01:00:00 --job-name=job1 python src/models/cherry_pick/train2.py \
    --wandb_name debugging \
    --run_name debugging2 \
    --seed 0 \
    --dataset singleDaily-modisgrid-new-const \
    --split_version dataset \
    --fold 1 \
    --nh 64 \
    --dropout 0.6 \
    --batch_size 500 \
    --epochs 50 
srun --nodes=1--time=01:00:00 --job-name=job2 python src/models/cherry_pick/train.py \
    --wandb_name debugging \
    --run_name debugging3 \
    --seed 0 \
    --dataset singleDaily-modisgrid-new-const \
    --split_version dataset \
    --fold 1 \
    --nh 64 \
    --dropout 0.6 \
    --batch_size 500 \
    --epochs 50 

