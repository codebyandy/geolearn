# python src/models/all_pick/train.py --run_name test_all --epochs 5 --iters_per_epoch 1 --test_epoch 1 --satellites 'no_landsat' --testing True
# python src/models/cherry_pick/train.py --run_name test_cherry --epochs 5 --iters_per_epoch 1 --test_epoch 1

# python helpers/submitJob.py --wandb_name new-dataset --note old-data_no-stratify --split_version dataset --dataset singleDaily-nadgrid --cross_val True
# python helpers/submitJob.py --wandb_name default --note old-data_stratify --split_version stratified --dataset singleDaily-nadgrid --cross_val True
# python helpers/submitJob.py --wandb_name new-dataset --note new-data_no-stratify --split_version dataset --dataset singleDaily-modisgrid-new-const --cross_val True
# python helpers/submitJob.py --wandb_name new-dataset --note new-data_stratify --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val True

python helpers/submitJob.py --wandb_name new-dataset --note nadgrid --split_version dataset --dataset singleDaily-nadgrid --cross_val True
python helpers/submitJob.py --wandb_name new-dataset --note nadgrid-new-const --split_version dataset --dataset singleDaily-nadgrid-new-const --cross_val True
python helpers/submitJob.py --wandb_name new-dataset --note modisgrid --split_version dataset --dataset singleDaily-modisgrid --cross_val True
python helpers/submitJob.py --wandb_name new-dataset --note modisgrid-new-const --split_version dataset --dataset singleDaily-modisgrid-new-const --cross_val True