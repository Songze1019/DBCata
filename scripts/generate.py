import os
import pickle
job_id = os.environ.get('SLURM_JOB_ID', 'default_version') # get slurm_id

import torch

from trainer.pl_cart import pl_module_cart
from model import InferenceModel


# data
stage = 'val'
data_path = f'data/cathub_all/clean/eval/eval_{stage}.pkl'
data_list = pickle.load(open(data_path, 'rb'))

# model
batch_size = 128
model_path = 'tb_logs/bbdm_adspainn/final/checkpoints/scorenet-epoch=3999-avg_val_loss=0.047.ckpt'
model = pl_module_cart.load_from_checkpoint(
    model_path
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inference_model = InferenceModel(model, data_list, batch_size, device)

# generate data and save
steps = 1000
eta = 0.0
# logging
print("=========================Inference Start=============================")
print("Stage: ", stage)
print("Data path: ", data_path)
print("Model path: ", model_path)
print("Steps: ", steps)
print("Eta: ", eta)

generated_data_list = inference_model.inference(steps, eta, device)
inference_model.save_generated_data('evaluation/final', stage, generated_data_list)
