import os
import pickle
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # prevent OOM
job_id = os.environ.get('SLURM_JOB_ID', 'default_version') # get slurm_id

import torch

from trainer.pl_cart import pl_module_cart
from model import InferenceModel


# data
dmae_threshold = 0.05
stage = 'train'
all_data = True
data_path = f'data/cathub/{stage}.pkl'
data_list = pickle.load(open(data_path, 'rb'))
data_output_path = f'data/cathub_label_{int(dmae_threshold * 100)}/{stage}.pkl'
if all_data:
    data_output_path = f'data/cathub_label_{int(dmae_threshold * 100)}/all.pkl'

if stage == 'train' and all_data:
    data_path_val = f'data/cathub/val.pkl'
    data_list_val = pickle.load(open(data_path_val, 'rb'))
    data_list = data_list + data_list_val
    
# model
batch_size = 256
model_path = 'tb_logs/bbdm_adspainn/final/checkpoints/scorenet-epoch=3999-avg_val_loss=0.047.ckpt'
model = pl_module_cart.load_from_checkpoint(
    model_path
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inference_model = InferenceModel(model, data_list, batch_size, device)

# generate data and save
steps = 20
eta = 0.0
# logging
print("=========================Labeling Start=============================")
print("Stage: ", stage)
print("Data output: ", data_output_path)
print("Model path: ", model_path)
print("Steps: ", steps)
print("Eta: ", eta)
print("D-MAE threshold: ", dmae_threshold)

generated_data_list = inference_model.inference(steps, eta, device)
generated_data_list_sde = inference_model.inference(steps, eta + 0.5, device)
generated_data_list_sde2 = inference_model.inference(steps, eta + 1.0, device)

unlabeled_data_list = generated_data_list + generated_data_list_sde + generated_data_list_sde2
data_output_list, num_positive, num_negative = inference_model.label_dmae(
    unlabeled_data_list, dmae_threshold
)
pickle.dump(unlabeled_data_list, open('tmp/data.pkl', 'wb'))
pickle.dump(data_output_list, open(data_output_path, 'wb'))

print("==========================Labeling RESULTS==============================")
print(f"Labeling data to: {data_output_path}")
print(f"Total data length: {len(unlabeled_data_list)}")
print(f"Positive data length: {num_positive}")
print(f"Negative data length: {num_negative}")
print("============================Labeling END================================")
