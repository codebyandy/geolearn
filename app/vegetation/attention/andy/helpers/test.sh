# python src/models/all_pick/train.py --run_name test_all --epochs 5 --iters_per_epoch 1 --test_epoch 1 --satellites 'no_landsat' --testing True
# python src/models/cherry_pick/train.py --run_name test_cherry --epochs 5 --iters_per_epoch 1 --test_epoch 1

python helpers/submitJob.py --seeds [1] --dropouts [0.6] --embedding_sizes [64] --wandb_name default ---note nadgrid_no_stratify --split_version dataset --dataset singleDaily-nadgrid --cross_val False
python helpers/submitJob.py --seeds [1] --dropouts [0.6] --embedding_sizes [64] --wandb_name default ---note nadgrid_stratify --split_version stratified --dataset singleDaily-nadgrid --cross_val False
python helpers/submitJob.py --seeds [1] --dropouts [0.6] --embedding_sizes [64] --wandb_name default ---note modisgrid_no_stratify --split_version dataset --dataset singleDaily-modisgrid-new-const --cross_val False
python helpers/submitJob.py --seeds [1] --dropouts [0.6] --embedding_sizes [64] --wandb_name default ---note modisgrid_stratify --split_version stratified --dataset singleDaily-modisgrid-new-const --cross_val False