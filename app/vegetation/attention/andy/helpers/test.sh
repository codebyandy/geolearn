python helpers/submitJob.py --wandb_name debugging --run_name OLD_modisgrid-new-const --split_version dataset --dataset singleDaily-modisgrid-new-const 
python helpers/submitJob.py --wandb_name debugging --run_name OLD_modisgrid-new-const-strat --split_version stratified --dataset singleDaily-modisgrid-new-const 

pthon helpers/submitJob.py --wandb_name debugging --run_name OLD_nadgrid --split_version dataset --dataset singleDaily-nadgrid 
python helpers/submitJob.py --wandb_name debugging --run_name OLD_nadgrid-strat --split_version stratified --dataset singleDaily-nadgrid