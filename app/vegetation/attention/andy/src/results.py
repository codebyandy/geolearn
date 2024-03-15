# from hydroDL import kPath

# from src.model import FinalModel
# from src.inference import inference, a
# import src.data as data

# import torch

# import os
# import pickle

# nh = 32
# rho = 45

# with open(os.path.join(kPath.dirVeg, 'data_tuple.pkl'), 'rb') as f:
#     data_tuple = pickle.load(f)

# nTup, nxc, lTup = data.get_shapes(data_tuple, rho, "")

# model = FinalModel(nTup, nxc, nh, "default")
# save_path = os.path.join(kPath.dirVeg, "runs/24-02-16_default")
# model_weights = os.path.join(save_path, "model")
# model.load_state_dict(torch.load(model_weights))

# inference(model, data_tuple, rho, "")



from inference import analysis

dict = {"TEST": ["24-03-03_TEST/0/model_450", 450],
        "TEST1": ["24-03-03_TEST/0/model_450", 450]}
analysis(dict)


