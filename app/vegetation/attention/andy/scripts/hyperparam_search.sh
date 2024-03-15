# python src/train.py --run hyperparam_search_lr1 --dataset data_folds.pkl --learning_rate 1e-1
# python src/train.py --run hyperparam_search_lr2 --dataset data_folds.pkl --learning_rate 1e-2
python src/train.py --run hyperparam_search_lr3 --dataset data_folds.pkl --learning_rate 1e-3
python src/train.py --run hyperparam_search_lr4 --dataset data_folds.pkl --learning_rate 1e-4