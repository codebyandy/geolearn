
dropout_lst = [0.2, 0.4, 0.6, 0.8]
nh_lst = [16, 32]


for dropout in dropout_lst:
    for nh in nh_lst:
        run_name = f'dropout_{dropuout}_nh_{nh}'
        cmd_line = f'python KUAI_TRAIN.py --run_name {run_name} --dropout {droput} --nh {nh}'