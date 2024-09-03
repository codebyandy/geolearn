# python src/models/all_pick/train.py --run_name test_all --epochs 5 --iters_per_epoch 1 --test_epoch 1 --satellites 'no_landsat' --testing True
# python src/models/cherry_pick/train.py --run_name test_cherry --epochs 5 --iters_per_epoch 1 --test_epoch 1

# python helpers/submitJob.py --wandb_name new-dataset --note old-data_no-stratify --split_version dataset --dataset singleDaily-nadgrid --cross_val True
# python helpers/submitJob.py --wandb_name default --note old-data_stratify --split_version stratified --dataset singleDaily-nadgrid --cross_val True
# python helpers/submitJob.py --wandb_name new-dataset --note new-data_no-stratify --split_version dataset --dataset singleDaily-modisgrid-new-const --cross_val True
# python helpers/submitJob.py --wandb_name new-dataset --note new-data_stratify --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True

# python helpers/submitJob.py --wandb_name new-dataset --note nadgrid --split_version dataset --dataset singleDaily-nadgrid --cross_val True
# python helpers/submitJob.py --wandb_name new-dataset --note nadgrid-new-const --split_version dataset --dataset singleDaily-nadgrid-new-const --cross_val True
# python helpers/submitJob.py --wandb_name new-dataset --note modisgrid --split_version dataset --dataset singleDaily-modisgrid --cross_val True
# python helpers/submitJob.py --wandb_name new-dataset --note modisgrid-new-const --split_version dataset --dataset singleDaily-modisgrid-new-const --cross_val True

# python helpers/submitJob.py --wandb_name stratify-check2 --note nadgrid --split_version dataset --dataset singleDaily-nadgrid --cross_val True
# python helpers/submitJob.py --wandb_name stratify-check2 --note nadgrid-STRAT --split_version stratified --dataset singleDaily-nadgrid --cross_val True
# python helpers/submitJob.py --wandb_name stratify-check2 --note modisgrid-new-const --split_version dataset --dataset singleDaily-modisgrid-new-const --cross_val True
# python helpers/submitJob.py --wandb_name stratify-check2 --note modisgrid-new-const-STRAT --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True

# He / larger learning rate
# python helpers/submitJob.py --wandb_name stratify-check2 --note nadgrid-STRAT-lr --split_version stratified --dataset singleDaily-nadgrid --cross_val True --seed 2 --learning_rate 0.05
# python helpers/submitJob.py --wandb_name stratify-check2 --note modisgrid-new-const-STRAT-lr --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True --seed 2 --learning_rate 0.05

# Smaller size
# python helpers/submitJob.py --wandb_name stratify-check2 --note modisgrid-new-const-STRAT --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True --seed 0 --test_epoch 10

# smaller learning rate
# python helpers/submitJob.py --wandb_name stratify-check2 --note modisgrid-new-const-STRAT-lr --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True --seed 2 --learning_rate 0.001

# python helpers/submitJob.py --wandb_name debugging --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True
# python helpers/submitJob.py --wandb_name debugging --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True
# python helpers/submitJob.py --wandb_name debugging --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True --test_epoch 10
# python helpers/sub/mitJob.py --wandb_name debugging --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True --test_epoch 10

# python src/models/cherry_pick/train.py --run_name test1 --epochs 4 --test_epoch 2 --fold 0
# python src/models/cherry_pick/train.py --run_name test1 --epochs 4 --test_epoch 2 --fold 1
# python src/models/cherry_pick/train.py --run_name test1 --epochs 4 --test_epoch 2 --fold 2
# python src/models/cherry_pick/train.py --run_name test1 --epochs 4 --test_epoch 2 --fold 3
# python src/models/cherry_pick/train.py --run_name test1 --epochs 4 --test_epoch 2 --fold 4

# quick test
python src/models/cherry_pick/train.py --wandb_name debugging --run_name test1 --epochs 6 --test_epoch 2