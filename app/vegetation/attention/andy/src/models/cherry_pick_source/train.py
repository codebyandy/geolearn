"""
This file is for training the transformer model for LFMC prediction.
"""

from model import FinalModel
from data import randomSubset, prepare_data
from inference import update_metrics_dict, get_metrics
from utils import set_seed

from hydroDL import kPath # module by Kuai Fang

import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torch import nn
import torch
import numpy as np
import pandas as pd
import wandb

import json
import os
import argparse
import time

import pdb


def train(args, saveFolder, run_details):
    nh = args.nh
    epochs = args.epochs
    learning_rate = args.learning_rate
    nIterEp = args.iters_per_epoch
    sched_start_epoch = args.sched_start_epoch
    loss_fn = args.loss_fn
    optimizer = args.optimizer
    dropout = args.dropout
    batch_size = args.batch_size
    test_epoch = args.test_epoch
    split_version = args.split_version
    fold = args.fold
    weight_decay = args.weight_decay

    # Store all required data in `data` dictionary
    data = prepare_data(args.dataset, args.rho)

    # Load previously generated dataset splits
    splits_path = os.path.join(kPath.dirVeg, 'model', 'attention', split_version, 'subset.json')
    with open(splits_path) as f:
        splits_dict = json.load(f)
    split_indicies = {
        'train' : splits_dict[f'trainInd_k{fold}5'],
        'test_quality_sites' : splits_dict[f'testInd_k{fold}5'],
        'test_poor_sites' : splits_dict['testInd_underThresh']
    }

    # Set up model 
    xS, xM, pS, pM, xcT, yT = randomSubset(data, split_indicies["train"], batch_size) # Get sample to get shapes for model
    nTup = (xS.shape[-1], xM.shape[-1]) # (list[int]) Number of input features for each remote sensing source (i.e. Sentinel, Modis) 
    lTup = (xS.shape[1], xM.shape[1]) # (tuple[int]) N)umber of sampled days for each remote sensing source
    nxc = data['xc'].shape[-1] # (int): number of constant variables
    
    model = FinalModel(nTup, nxc, nh, dropout)
    # loss
    if loss_fn == 'l1':
        loss_fn = nn.L1Loss(reduction='mean')
    elif loss_fn == 'mse':
        loss_fn = nn.MSELoss(reduction='mean')
    # optimizer
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)    
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=800)
    
    # Training loop
    epoch_wall_times = []
    iteration_wall_times = []
    test_wall_times = []
    
    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        
        metrics = {
            "train_epoch_mean_loss" : 0,
            "train_epoch_mean_rmse" : 0,
            "train_epoch_mean_corrcoef" : 0,
            "train_epoch_mean_coefdet" : 0
        }
        
        for _ in range(nIterEp):
            iteration_start = time.time()
            optimizer.zero_grad()
            
            # One training iteration
            xS, xM, pS, pM, xcT, yT = randomSubset(data, split_indicies["train"], batch_size)
            xTup = (xS, xM)
            pTup = (pS, pM)
     
            yP = model(xTup, pTup, xcT, lTup)      
            loss = loss_fn(yP, yT)
            loss.backward() # backpropagation
            optimizer.step() # update weights
    
            # Get iteration training metrics      
            minibatch_metrics = get_metrics(yP.detach().numpy(), yT.detach().numpy())
            metrics['train_epoch_mean_loss'] += loss.item()
            metrics['train_epoch_mean_rmse']+= minibatch_metrics['rmse']
            metrics['train_epoch_mean_corrcoef'] += minibatch_metrics['corrcoef']
            metrics['train_epoch_mean_coefdet'] += minibatch_metrics['coefdet']

            iteration_wall_times.append(time.time() - iteration_start)

        if ep > sched_start_epoch:
            scheduler.step()

        metrics = {metric : sum / nIterEp for metric, sum in metrics.items()}
        print('Epoch: {} | Loss: {:.3f} Coefdet: {:.3f}'.format(ep, \
                                                                metrics['train_epoch_mean_loss'], \
                                                                metrics['train_epoch_mean_coefdet']))
        
        epoch_wall_times.append(time.time() - epoch_start)

        # Get metrics on full test set every `test_epoch`
        if (ep > 0 and ep % test_epoch == 0) or ep == epochs:
            test_start = time.time() 
            print("Testing on full test set") 
            
            model.eval()
            with torch.no_grad():
                full_set_metrics = {}
                update_metrics_dict(full_set_metrics, data, split_indicies['train'], model, 'train')
                update_metrics_dict(full_set_metrics, data, split_indicies['test_quality_sites'], model, 'qual')
                update_metrics_dict(full_set_metrics, data, split_indicies['test_poor_sites'], model, 'poor')
                metrics.update(full_set_metrics)

                # Update sheet with (epoch, metrics) for this run
                full_set_metrics.update({'epoch' : ep})
                new_metrics_df = pd.DataFrame(full_set_metrics, index=[0])

                # move epochs to front
                cols = list(new_metrics_df)
                cols.insert(0, cols.pop(cols.index('epoch')))
                new_metrics_df = new_metrics_df.loc[:, cols]

                metrics_path = os.path.join(saveFolder, 'metrics.csv')
                if os.path.exists(metrics_path):
                    prev_metrics_df = pd.read_csv(metrics_path)
                    new_metrics_df = pd.concat([prev_metrics_df, new_metrics_df])   
                new_metrics_df.to_csv(os.path.join(saveFolder, 'metrics.csv'), index=False)
                torch.save(model.state_dict(), os.path.join(saveFolder, f'model_ep{ep}.pth'))

            test_wall_times.append(time.time() - test_start)

        if not args.testing:
            wandb.log(metrics)

    # Save mean wall times
    time_path = os.path.join(saveFolder, 'time.csv')
    time_data = {
        'mean_epoch_time': [np.mean(epoch_wall_times)],
        'mean_iteration_time': [np.mean(iteration_wall_times)],
        'mean_test_time': [np.mean(test_wall_times)]
    }
    time_df = pd.DataFrame(time_data)
    time_df.to_csv(time_path, index=False)

    # Update sheet containing all runs and best metrics
    metrics_path = os.path.join(saveFolder, 'metrics.csv')
    metrics_df = pd.read_csv(metrics_path)
    reported_metrics = metrics_df.iloc[-1]

    run_details.update(time_data)
    run_details.update(reported_metrics)
    # update_metrics_dict(run_details, data, split_indicies['train'], model, 'train')
    if not args.testing:
        wandb.log(metrics)

    run_details_df = pd.DataFrame(run_details)
    all_run_details_path = os.path.join(kPath.dirVeg, 'runs', 'runs.csv')
    if os.path.exists(all_run_details_path):
        all_run_details_df = pd.read_csv(all_run_details_path)
        all_run_details_df = pd.concat([all_run_details_df, run_details_df])
        all_run_details_df.to_csv(all_run_details_path, index=False)
    else:
        run_details_df.to_csv(all_run_details_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    # admin
    parser.add_argument("--wandb_name", type=str, default="default")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument("--run_name", type=str, default='')   
    parser.add_argument("--testing", type=bool, default=False)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    # dataset 
    parser.add_argument("--dataset", type=str, default="singleDaily-modisgrid-new-const")
    parser.add_argument("--split_version", type=str, default="stratified")
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    # model
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--rho", type=int, default=45)
    parser.add_argument("--nh", type=int, default=32)
    parser.add_argument('--loss_fn', type=str, default='l1', choices=['l1', 'mse'])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    # training
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--iters_per_epoch", type=int, default=20)
    parser.add_argument("--sched_start_epoch", type=int, default=200)
    parser.add_argument("--test_epoch", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Set up save dir to save hyperparameters, metrics, models, time cost
    save_path = os.path.join(kPath.dirVeg, 'runs', args.run_name)

    # Create save dir if it doesn't already exist
    try: # prevent race conditions
        os.mkdir(save_path)
        print('Created new directory')
    except:
        print(f'Adding to existing directory.')

    # Create fold sub save dir  
    fold_save_path = os.path.join(save_path, str(args.fold))
    if os.path.exists(fold_save_path):
        raise Exception('Run already exists!')
    os.mkdir(fold_save_path)
    
    # save hyperparameters
    run_details_path = os.path.join(fold_save_path, 'details.json')
    with open(run_details_path, 'w') as f:
        to_save = vars(args)
        json.dump(to_save, f, indent=4)
    
    with open(run_details_path, 'r') as f:
        run_details = json.load(f)

    if not args.testing:
        # Create wandb log
        wandb.init(
            dir=os.path.join(kPath.dirVeg),
            project=args.wandb_name,
            config=run_details
        )
        wandb.run.name = args.run_name + '_f' + str(args.fold)

    train(args, fold_save_path, run_details)
    