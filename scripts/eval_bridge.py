from pathlib import Path
import pickle

import torch

from trainer.pl_cart import pl_module_cart
from model import InferenceModel

# data
stage = 'val'
data_path = Path("data/cathub/") / f"{stage}.pkl"
data_list = pickle.load(open(data_path, 'rb'))

# model
batch_size = 256
model_path = 'tb_logs/bbdm_adspainn/final/checkpoints/scorenet-epoch=3999-avg_val_loss=0.047.ckpt'
model = pl_module_cart.load_from_checkpoint(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inference_model = InferenceModel(model, data_list, batch_size, device)

# generate data and save
steps = 20
eta = 0.0

# logging
print("=========================Inference Started=============================")
print("Stage: ", stage)
print("Data path: ", data_path)
print("Model path: ", model_path)
print("Steps: ", steps)
print("Eta: ", eta)

# evaluate
generated_data_list = inference_model.inference(steps, eta, device)
inference_model.process_dataloader(generated_data_list)
inference_model.compute_dmae_mic_results()
inference_model.compute_rmsd_results()

# save structures generated
output_path = Path(f'tmp') / 'diffcata.pkl'
output_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving generated data to {output_path}")
pickle.dump(generated_data_list, open(output_path, 'wb'))
