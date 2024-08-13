"""
This file is for training the transformer model for LFMC prediction.
"""

from model import FinalModel
from data import randomSubset, prepare_data
from inference import test_metrics, train_metrics

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"seed {seed} set")


def train(args, saveFolder, fold):
    run_name = args.run_name
    dataName= args.dataset
    rho = args.rho
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
    wandb_name = args.wandb_name
    split_version = args.split_version

    if not args.testing:
        saveFolder = os.path.join(saveFolder, str(fold))
        os.mkdir(saveFolder)
    
    run_details = {
        "date": datetime.today().strftime('%Y-%m-%d'),
        "run_name": run_name,
        "data_fold": fold,
        "training_method": "cherry_picking",
        "rho": rho,
        "embedding_size": nh,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "iterations_per_epoch": nIterEp,
        "schedule_start_epoch": sched_start_epoch,
        "optimizer": optimizer,
        "dropout": dropout,
        "batch_size": batch_size,
        "test_epoch": test_epoch,
        "satellites": satellites
    }

    # Store all required data in `data` tuple
    data = prepare_data(args.dataset, args.rho)
    df, dm, iInd, jInd, nMat, pSLst, pMLst, x, rho, xc, yc = data # TODO: Clean up or define in README

    # Do not log run if just a test
    if not args.testing:
        wandb.init(
            dir=os.path.join(kPath.dirVeg),
            project=wandb_name,
            config=run_details
        )
        wandb.run.name = run_name

        metrics = [
            "poor_anomaly_coefdet",
            "poor_anomaly_corrcoef",
            "poor_anomaly_rmse",
            "poor_obs_coefdet",
            "poor_obs_corrcoef",
            "poor_obs_rmse",
            "poor_site_coefdet",
            "poor_site_corrcoef",
            "poor_site_rmse",
            "qual_anomaly_coefdet",
            "qual_anomaly_corrcoef",
            "qual_anomaly_rmse",
            "qual_obs_coefdet",
            "qual_obs_corrcoef",
            "qual_obs_rmse",
            "qual_site_coefdet",
            "qual_site_corrcoef",
            "qual_site_rmse",
            "test_RMSE",
            "test_Rsq",
            "test_loss",
            "test_rsq",
            "train_RMSE",
            "train_Rsq",
            "train_loss",
            "train_rsq"
        ]

        # Define each metric to have its summary as "max"
        # for metric in metrics:
        #     wandb.define_metric(metric, summary="max")

    # Load previously generated dataet splits
    dataFolder = os.path.join(kPath.dirVeg, 'model', 'attention', split_version)
    subsetFile = os.path.join(dataFolder, 'subset.json')

    with open(subsetFile) as json_file:
        dictSubset = json.load(json_file)
    print("loaded dictSubset")
    
    split_indicies = {}
    split_indicies["train"] = dictSubset[f'trainInd_k{fold}5']
    split_indicies["test_quality"] = dictSubset[f'testInd_k{fold}5']
    split_indicies["test_poor"] = dictSubset['testInd_underThresh']

    # Get a random subset to get important shapes
    xS, xM, pS, pM, xcT, yT = randomSubset(data, split_indicies["train"], split_indicies["test_quality"], 'train', batch_size)
    nTup, lTup = (), () # nTup (list[int]): Number of input features for each remote sensing source (i.e. Sentinel, Modis) 
                        # lTup (tuple[int]): number of sampled days for each remote sensing source
    if satellites == "no_landsat":
        print("no landsat model")
        nTup = (xS.shape[-1], xM.shape[-1])
        lTup = (xS.shape[1], xM.shape[1])
    else:
        nTup = (xS.shape[-1], xL.shape[-1], xM.shape[-1])
        lTup = (xS.shape[1], xL.shape[1], xM.shape[1])
    nxc = xc.shape[-1] # nxc (int): Number of constant variables
    
    # Set up model
    model = FinalModel(nTup, nxc, nh, dropout)
    loss_fn = nn.L1Loss(reduction='mean')
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)    
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=800
    )
    
    # Training loop
    epoch_wall_times = []
    iteration_wall_times = []
    test_wall_times = []

    model.train()
    
    for ep in range(epochs):
        epoch_start = time.time()
        
        lossEp = 0
        metrics = {"train_loss" : 0, "train_RMSE" : 0, "train_rsq" : 0, "train_Rsq" : 0}
        
        for i in range(nIterEp):
            iteration_start = time.time()
            model.zero_grad()
            
            # One training iteration
            xS, xM, pS, pM, xcT, yT = randomSubset(data, split_indicies["train"] , split_indicies["test_quality"] , 'train', batch_size)
            xTup, pTup = (), ()
            if satellites == "no_landsat":
                xTup = (xS, xM)
                pTup = (pS, pM)
            else:
                xTup = (xS, xL, xM)
                pTup = (pS, pL, pM) 
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

        # Get metrics on subset of test set at the end of every epoch
        xS, xM, pS, pM, xcT, yT = randomSubset(data, split_indicies["train"] , split_indicies["test_quality"], 'test', batch_size)
        xTup, pTup = (), ()
        if satellites == "no_landsat":
            xTup = (xS, xM)
            pTup = (pS, pM)
        else:
            xTup = (xS, xL, xM)
            pTup = (pS, pL, pM)  
        yP = model(xTup, pTup, xcT, lTup)
        loss = loss_fn(yP, yT)
    
        metrics["test_loss"] = loss_fn(yP, yT).item()
        obs, pred = yP.detach().numpy(), yT.detach().numpy()
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        corrcoef = np.corrcoef(obs, pred)[0, 1]
        coef_det = r2_score(obs, pred)

        with torch.no_grad():
            metrics["test_RMSE"]= rmse
            metrics["test_rsq"] = corrcoef
            metrics["test_Rsq"] = coef_det
        
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

    if not args.testing:
        metrics_path = os.path.join(saveFolder, 'metrics.csv')
        metrics = pd.read_csv(metrics_path)

        reported_metrics = metrics.iloc[-1]
        old_reported_model_path = os.path.join(saveFolder, f'model_ep{int(reported_metrics.epoch)}.pth')
        new_reported_model_path = os.path.join(saveFolder, 'best_model.pth')
        shutil.copyfile(old_reported_model_path, new_reported_model_path)

        # Update sheet containing all runs and best metrics
        run_details.update(time_data)
        run_details.update(reported_metrics)

        config = {"model" : model, "satellites" : satellites, "epoch" : ep}
        full_train_metrics = train_metrics(data, split_indicies, config)
        full_train_metrics_floats = {k : v[0] for k, v in full_train_metrics.items()}
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--testing", type=bool, default=False)
    parser.add_argument("--cross_val", type=bool, default=False)
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--wandb_name", type=str, default="default")
    # dataset 
    parser.add_argument("--dataset", type=str, default="singleDaily-modisgrid-new-const")
    parser.add_argument("--split_version", type=str, default="dataset", choices=["dataset", "stratified"])
    parser.add_argument("--rho", type=int, default=45)
    parser.add_argument("--satellites", type=str, default="all")
    # model
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
    
    # create save dir / save hyperparameters
    saveFolder = ""
    if not args.testing:
        date_run_name = datetime.today().strftime('%Y-%m-%d') + '_' + args.run_name
        saveFolder = os.path.join(kPath.dirVeg, 'runs', f"{date_run_name}")
        
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        else:
            raise Exception("Run already exists!")
    
        json_fname = os.path.join(saveFolder, "hyperparameters.json")
        with open(json_fname, 'w') as f:
            tosave = vars(args)
            json.dump(tosave, f, indent=4)

    for fold in range(5):
        if not args.cross_val and fold > 0:
            break 
        train(args, saveFolder, fold)
    


