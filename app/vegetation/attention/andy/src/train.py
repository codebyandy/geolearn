
from model import FinalModel
from inference import inference, calc_metrics
import data

# Kuai Fang package
from hydroDL import kPath

# training
import torch
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import wandb

# misc
import argparse
from tqdm import tqdm
from datetime import datetime
import os
import json
import pickle
import pdb


def get_inputs(inputs, subset):
    """
    Selects the appropriate satellite data input for a given model based on parameters.

    Parameters:
        inputs (str): 
        subset (str): 

    Returns:
        
    """
    xS, xL, xM, pS, pL, pM, xcT, yT = subset
    if inputs == "no_M":
        return ((xS, xL), (pS, pL), xcT, yT)
    elif inputs == "no_S":
        return ((xL, xM), (pL, pM), xcT, yT)
    elif inputs == "no_L":
        return ((xS, xM), (pS, pM), xcT, yT)
    else:
        return ((xS, xL, xM), (pS, pL, pM), xcT, yT)
    

def train(data_tuple, rho, nh, epochs, iters_per_epoch, learning_rate, sched_start_epoch,
          inputs, out_method, testing, weights_path, device, foldFolder):
    # device
    # device = torch.device(f"cuda:{args.device}" if args.device >=0 else "cpu")
    print(f"Using device: {device}")
    
    # prepare model
    nTup, nxc, lTup = data.get_shapes(data_tuple, rho, inputs)
    model = FinalModel(nTup, nxc, nh, out_method)
    loss_fn = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=800)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path}")

    model.train()
    model.to(device)

    pbar = tqdm(range(epochs))
    for ep in pbar:   
        # train on `iters_per_epochs` subsets of trainset 
        metrics = {"train_loss" : 0, "train_RMSE" : 0, "train_rsq" : 0, "train_Rsq" : 0}
        for i in range(iters_per_epoch):
            model.zero_grad()
            subset = data.randomSubset(data_tuple, rho)
            x, pos, xcT, yT = get_inputs(inputs, subset)
            yP = model(x, pos, xcT, lTup)
            loss = loss_fn(yP, yT)
            loss.backward()
            
            metrics["train_loss"] += loss.item()
            with torch.no_grad():
                RMSE, rsq, Rsq = calc_metrics(yT.detach().cpu().numpy(), yP.detach().cpu().numpy())
                metrics["train_RMSE"]+= RMSE
                metrics["train_rsq"] += rsq
                metrics["train_Rsq"] += Rsq
            optimizer.step()

        # get avg of train metrics
        metrics = {metric : sum / iters_per_epoch for metric, sum in metrics.items()}
        
        # get metrics on subset of testset
        optimizer.zero_grad()
        subset = data.randomSubset(data_tuple, rho, 'test')
        x, pos, xcT, yT = get_inputs(inputs, subset)
        yP = model(x, pos, xcT, lTup)        
        metrics["test_loss"] = loss_fn(yP, yT).item()
        with torch.no_grad():
            metrics["test_RMSE"], metrics["test_rsq"], metrics["test_Rsq"] = calc_metrics(yT.detach().cpu().numpy(), yP.detach().cpu().numpy())

        # get metrics on full trainset, testset
        if ep % 50 == 0 and ep != 0:
            _, (full_train_metrics, _, _) = inference(model, data_tuple, rho, inputs, "train")
            _, (full_test_metrics, _, _) = inference(model, data_tuple, rho, inputs)
            metrics["full_train_RMSE"], metrics["full_train_rsq"], metrics["full_train_Rsq"] = full_train_metrics
            metrics["full_test_RMSE"], metrics["full_test_rsq"], metrics["full_test_Rsq"] = full_test_metrics
            if not testing:
                torch.save(model.state_dict(), os.path.join(foldFolder, f'model_{ep}'))

        # scheduler
        if ep > sched_start_epoch:
            scheduler.step()

        # logging
        desc = f"Ep: {ep} | "
        desc += f"RMSE: {round(metrics['test_RMSE'], 2)}, rsq: {round(metrics['test_rsq'], 3)}, Rsq: {round(metrics['test_Rsq'], 3)}"
        pbar.set_description(desc)
        if not testing:
            wandb.log(metrics)

    if not testing:
        torch.save(model.state_dict(), os.path.join(foldFolder, 'final_model'))
    
    return model
    

def main(args, saveFolder):
    # load dataset
    # with open(os.path.join(kPath.dirVeg, args.dataset), 'rb') as f:
    #     data_folds = pickle.load(f)
    with open("/Users/andyhuynh/Documents/lfmc/data/stratified_3_fold.pkl", 'rb') as f:
        data_tuple = pickle.load(f)

    # loop thru folds for cross-validation
    # for i, data_tuple in data_folds.items(): 
    for i in range(3):
        if not args.cross_val and i > 0:
            break # stop after first fold if not doing cross-validation

        # logging setup
        foldFolder = ""
        if not args.testing:
            wandb.init(dir=os.path.join(kPath.dirVeg),
                       config={"learning_rate": args.learning_rate, 
                               "epochs": args.epochs})
            wandb.run.name = args.run + f"_{i}"
            foldFolder = os.path.join(saveFolder, str(i))
            os.mkdir(foldFolder)
    
        # init and train model    
        model = train(data_tuple, args.rho, args.nh,
                        args.epochs, args.iters_per_epoch, args.learning_rate,
                        args.sched_start_epoch, args.inputs, args.out_method, args.testing,
                        args.weights_path, args.device, foldFolder
                       )
        
        if not args.testing:
            wandb.run.finish()
            torch.save(model.state_dict(), os.path.join(foldFolder, 'final_model'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    # admin
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--testing", type=bool, default=False)
    parser.add_argument("--cross_val", type=bool, default=False)
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--device", type=int, default=-1)
    # dataset 
    parser.add_argument("--dataset", type=str, default="data_folds.pkl")
    parser.add_argument("--rho", type=int, default=45)
    parser.add_argument("--inputs", type=str, default="")
    # model
    parser.add_argument("--nh", type=int, default=32)
    parser.add_argument("--out_method", type=str, default="default")
    # training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--iters_per_epoch", type=int, default=20)
    parser.add_argument("--sched_start_epoch", type=int, default=200)
    args = parser.parse_args()

    
    # create save dir / save hyperparameters
    saveFolder = ""
    if not args.testing:
        date = datetime.now().strftime('%y-%m-%d')
        saveFolder = os.path.join(kPath.dirVeg, 'runs', f"{date}_{args.run}")
        
        if not os.path.exists(saveFolder):
            os.mkdir(saveFolder)
        else:
            raise Exception("Run already exists!")
    
        json_fname = os.path.join(saveFolder, "hyperparameters.json")
        with open(json_fname, 'w') as f:
            tosave = vars(args)
            json.dump(tosave, f, indent=4)

    
    main(args, saveFolder)

