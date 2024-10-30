# Leveraging transformers and remote sensing for mapping of live fuel moisture content

This code accompanies the report available [here](https://docs.google.com/document/d/1wP6Jd8b0rMYvZIPV4YHrxTVGuNYRcSZ8DwVss0O-VbI/edit?usp=sharing). Please read the report for a detailed explanation of the methods and results.

## Environment setup
```bash
TODO
```
[`wandb`](https://pypi.org/project/wandb/) can additionally be installed for full functionality of the `train.py` script.

## Repo stucture

There are three main directories in this repository:
- `data`: contains all the code for data preprocessing
- `results`: contains all the code to create tables and figures from the experiment results
- `src`: contains the source code for training and inference of the models

```bash
├── data
├── results
├── src
│   ├── models
│   │   ├── all_pick
│   │   │   ├── data.py
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   ├── utils.py
│   │   │   ├── inference.py
│   │   ├── chery_pick
│   │   │   ├── data.py
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   ├── utils.py
│   │   │   ├── inference.py
```
The `all_pick` directory contains the code for the model that uses all the available data during the temporal window, while the `chery_pick` directory contains the code for the model that uses a subset of the data.

## Training

To train a model, call `train.py`. You can pass an experiment name (`--exp_name`)and a run name (`--run_name`) as arguments. 

The experiment name is used to easily identify and filter for a set of runs for later analysis. The run name is used to distinguish a unique run.

#### Example: 
```bash
python train.py \
    --exp_name exp1 \  
    --run_name run1 \
    --testing True \       # turns off WandB logging
    --fold 0               # specifies the fold to train on
```

**Note** Calling `train.py` will train the model for only a single fold. To train the model for all folds, you can call the script multiple times, changing the `--fold` argument each time.

See the `train.py` script for more information on the available arguments.

## How runs are saved

`train.py` will save the model weights and the training history to a designated local directory. Let's call this directory `runs`. 

```bash
├── runs
│   ├── run1                        # run_name
│   │   ├── 0                           # fold 0
│   │   │   ├── details.json                # arguments to `train.py`
│   │   │   ├── metrics.csv                 # model metrics over time
│   │   │   ├── time.csv                    # run time metrics
│   │   │   ├── model_ep50.pth              # model checkpoint
│   │   ├── 1                           # fold 1
│   │   ├── 2                           # fold 2
│   │   ├── 3                           # fold 3
│   │   ├── 4                           # fold 4
│   ├── runs.csv                    # metrics from last epoch of each run
```




When you train a model with a new `run_name`, within the `runs` directory, a subdirectory will be created with the name of the `run_name`. Within this subdirectory, another subdirectory will be created with the name of the `fold`. Inside each of these subdirectories, the model weights and training history will be saved.

If you are using `WandB`, the model metric logs will also be saved to the cloud. 


## References

This README.md was partially inspired by the one available [here](https://github.com/nasaharvest/presto) by Tseng et al.