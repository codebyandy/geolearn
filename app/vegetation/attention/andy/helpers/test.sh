python helpers/submitJob.py --wandb_name debugging --run_name modisgrid-new-const --split_version dataset --dataset singleDaily-modisgrid-new-const --seeds 0 --folds 0,4 --protection False
python helpers/submitJob.py --wandb_name debugging --run_name modisgrid-new-const --split_version dataset --dataset singleDaily-modisgrid-new-const --seeds 1 --folds 1 --protection False
python helpers/submitJob.py --wandb_name debugging --run_name modisgrid-new-const --split_version dataset --dataset singleDaily-modisgrid-new-const --seeds 2 --folds 1 --protection False

python helpers/submitJob.py --wandb_name debugging --run_name modisgrid-new-const-strat --split_version stratified --dataset singleDaily-modisgrid-new-const --seeds 2 --folds 3 --protection False