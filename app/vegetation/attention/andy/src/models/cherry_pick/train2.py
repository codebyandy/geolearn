"""
This file is for training the transformer model for LFMC prediction.
"""

from model import FinalModel
from data import randomSubset, prepare_data
from inference import test_metrics, train_metrics
from utils import set_seed

# hydroDL module by Kuai Fang
from hydroDL.data import dbVeg
from hydroDL.data import DataModel
from hydroDL.master import dataTs2Range
from hydroDL import kPath

import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torch import nn
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
import wandb

from datetime import datetime
import json
import os
import argparse
import shutil
import random
import time

import pdb


def train(args, saveFolder, fold):
    nh = args.nh
    epochs = args.epochs
    learning_rate = args.learning_rate
    nIterEp = args.iters_per_epoch
    sched_start_epoch = args.sched_start_epoch
    optimizer = args.optimizer
    dropout = args.dropout
    batch_size = args.batch_size
    test_epoch = args.test_epoch
    satellites = args.satellites
    split_version = args.split_version
    fold = args.fold

    # (1) Store all required data in `data` tuple
    data = prepare_data(args.dataset, args.rho)
    df, dm, iInd, jInd, nMat, pSLst, pMLst, x, rho, xc, yc = data # TODO: Clean up or define in README

    # (2) Load previously generated dataet splits
    data_path = os.path.join(kPath.dirVeg, 'model', 'attention', split_version)
    splits_path = os.path.join(data_path, 'subset.json')

    with open(splits_path) as f:
        splits_dict = json.load(f)
    
    split_indicies = {}
    split_indicies["train"] = splits_dict[f'trainInd_k{fold}5']
    split_indicies["test_quality"] = splits_dict[f'testInd_k{fold}5']
    split_indicies["test_poor"] = splits_dict['testInd_underThresh']

    # (3) Get a random subset to get important shapes
    xS, xM, pS, pM, xcT, yT = randomSubset(data, split_indicies["train"], split_indicies["test_quality"], 'train', batch_size)
    nTup, lTup = (), () # nTup (list[int]): Number of input features for each remote sensing source (i.e. Sentinel, Modis) 
                        # lTup (tuple[int]): number of sampled days for each remote sensing source
    if satellites == "no_landsat":
        print("no landsat model")
        nTup = (xS.shape[-1], xM.shape[-1])
        lTup = (xS.shape[1], xM.shape[1])
    # else:
    #     nTup = (xS.shape[-1], xL.shape[-1], xM.shape[-1])
    #     lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
    nxc = xc.shape[-1] # nxc (int): Number of constant variables
    
    # (4) Set up model
    model = FinalModel(nTup, nxc, nh, dropout)
    loss_fn = nn.L1Loss(reduction='mean')
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)    
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=800)
    
    # (5) Training loop
    epoch_wall_times = []
    iteration_wall_times = []
    test_wall_times = []

    model.train()
    
    for ep in range(epochs):
        epoch_start = time.time()
        
        lossEp = 0
        metrics = {
            "train_minibatch_loss" : 0,
            "train_minibatch_rmse" : 0,
            "train_minibatch_corrcoef" : 0,
            "train_minibatch_coefdet" : 0
        }

        for i in range(nIterEp):
            iteration_start = time.time()
            model.zero_grad()
            
            # One training iteration
            xS, xM, pS, pM, xcT, yT = randomSubset(data, split_indicies["train"] , split_indicies["test_quality"] , 'train', batch_size)
            xTup, pTup = (), ()
            if satellites == "no_landsat":
                xTup = (xS, xM)
                pTup = (pS, pM)
            # else:
            #     xTup = (xS, xL, xM)
            #     pTup = (pS, pL, pM) 
            yP = model(xTup, pTup, xcT, lTup)      
            loss = loss_fn(yP, yT)
            loss.backward()
      
            lossEp += loss.item()
            optimizer.step()
    
            # Get iteration training metrics
            metrics["train_loss"] += loss.item()
            with torch.no_grad():
                obs, pred = yP.detach().numpy(), yT.detach().numpy()
                rmse = np.sqrt(np.mean((obs - pred) ** 2))
                corrcoef = np.corrcoef(obs, pred)[0, 1]
                coef_det = r2_score(obs, pred)

                metrics["train_RMSE"]+= rmse
                metrics["train_rsq"] += corrcoef
                metrics["train_Rsq"] += coef_det

            iteration_end = time.time()
            iteration_wall_times.append(iteration_end - iteration_start)

        metrics = {metric : sum / nIterEp for metric, sum in metrics.items()}
        optimizer.zero_grad()

        # # Get metrics on subset of test set at the end of every epoch
        # xS, xM, pS, pM, xcT, yT = randomSubset(data, split_indicies["train"] , split_indicies["test_quality"], 'test', batch_size)
        # xTup, pTup = (), ()
        # if satellites == "no_landsat":
        #     xTup = (xS, xM)
        #     pTup = (pS, pM)
        # # else:
        # #     xTup = (xS, xL, xM)
        # #     pTup = (pS, pL, pM)  
        # yP = model(xTup, pTup, xcT, lTup)
        # loss = loss_fn(yP, yT)
    
        # metrics["test_loss"] = loss_fn(yP, yT).item()
        # obs, pred = yP.detach().numpy(), yT.detach().numpy()
        # rmse = np.sqrt(np.mean((obs - pred) ** 2))
        # corrcoef = np.corrcoef(obs, pred)[0, 1]
        # coef_det = r2_score(obs, pred)

        # with torch.no_grad():
        #     metrics["test_RMSE"]= rmse
        #     metrics["test_rsq"] = corrcoef
        #     metrics["test_Rsq"] = coef_det
        
        if ep > sched_start_epoch:
            scheduler.step()

        epoch_end = time.time()
        epoch_wall_times.append(epoch_end - epoch_start)

        print(
            '{} {:.3f} {:.3f} {:.3f}'.format(
                ep, lossEp / nIterEp, loss.item(), corrcoef
            )
        )

        # Get metrics on full test set every `test_epoch`
        if ep > 0 and ep % test_epoch == 0:
            print("testing on full test set") 
            test_start = time.time()

            config = {"model" : model, "satellites" : satellites, "epoch" : ep}
            full_test_metrics = test_metrics(data, split_indicies, config)
            full_test_metrics_floats = {k : v[0] for k, v in full_test_metrics.items()}
            metrics.update(full_test_metrics_floats)
            
            if not args.testing:
                # Update sheet with (epoch, metrics) for this run
                new_metrics = pd.DataFrame(full_test_metrics)
                metrics_path = os.path.join(saveFolder, 'metrics.csv')
                if os.path.exists(metrics_path):
                    prev_metrics = pd.read_csv(metrics_path)
                    new_metrics = pd.concat([prev_metrics, new_metrics])   
                new_metrics.to_csv(os.path.join(saveFolder, 'metrics.csv'), index=False)
                torch.save(model.state_dict(), os.path.join(saveFolder, f'model_ep{ep}.pth'))

            test_end = time.time()
            test_wall_times.append(test_end - test_start)

        if not args.testing:
            wandb.log(metrics)

    if not args.testing:
        # Save mean wall times
        time_path = os.path.join(saveFolder, 'time.csv')
        mean_epoch_time = np.mean(epoch_wall_times)
        mean_iteration_time = np.mean(iteration_wall_times)
        mean_test_time = np.mean(test_wall_times)
        time_data = {
            'mean_epoch_time': [mean_epoch_time],
            'mean_iteration_time': [mean_iteration_time],
            'mean_test_time': [mean_test_time]
        }
        time_df = pd.DataFrame(time_data)
        time_df.to_csv(time_path, index=False)

        metrics_path = os.path.join(saveFolder, 'metrics.csv')
        metrics = pd.read_csv(metrics_path)

        reported_metrics = metrics.iloc[-1]
        # old_reported_model_path = os.path.join(saveFolder, f'model_ep{int(reported_metrics.epoch)}.pth')
        # new_reported_model_path = os.path.join(saveFolder, 'best_model.pth')
        # shutil.copyfile(old_reported_model_path, new_reported_model_path)

        # Update sheet containing all runs and best metrics
        run_details.update(time_data)
        run_details.update(reported_metrics)

        config = {"model" : model, "satellites" : satellites, "epoch" : ep}
        full_train_metrics = train_metrics(data, split_indicies, config)
        # full_train_metrics_floats = {k : v[0] for k, v in full_train_metrics.items()}
        run_details.update(full_train_metrics)

        run_details = pd.DataFrame(run_details)

        all_run_details_path = os.path.join(kPath.dirVeg, 'runs', 'best_metrics_all_runs.csv')
        if os.path.exists(all_run_details_path):
            all_run_details = pd.read_csv(all_run_details_path)
            all_run_details = pd.concat([all_run_details, run_details])
            all_run_details.to_csv(all_run_details_path, index=False)
        else:
            run_details.to_csv(all_run_details_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    # admin
    parser.add_argument("--wandb_name", type=str, default="default")
    parser.add_argument("--run_name", type=str, required=True)   
    parser.add_argument("--testing", type=bool, default=False)
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--device", type=int, default=-1)
    # dataset 
    parser.add_argument("--dataset", type=str, default="singleDaily-modisgrid-new-const")
    parser.add_argument("--split_version", type=str, default="dataset", choices=["dataset", "stratified"])
    parser.add_argument("--rho", type=int, default=45)
    parser.add_argument("--satellites", type=str, default="no_landsat")
    # model
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nh", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
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
    
    # create save dir to save hyperparameters, metrics, models, time cost
    save_path = ''
    if not args.testing:
        save_path = os.path.join(kPath.dirVeg, 'runs', f'{args.run_name}')
        run_details_path = os.path.join(save_path, 'details.json')

        # create save dir if it doesn't already exist
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            # save hyperparameters
            with open(run_details_path, 'w') as f:
                to_save = vars(args)
                json.dump(to_save, f, indent=4)
    
        # create fold sub save dir  
        fold_save_path = os.path.join(save_path, str(args.fold))
        if os.path.exists(fold_save_path):
            raise Exception('Run already exists!')
        os.mkdir(fold_save_path)

        with open(run_details_path, 'r') as f:
            run_details = json.load(f)
        wandb.init(
            dir=os.path.join(kPath.dirVeg),
            project=args.wandb_name,
            config=run_details
        )
        wandb.run.name = args.run_name + "_f" + str(args.fold)

    train(args, fold_save_path, run_details)
    
