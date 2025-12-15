import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # prevent OOM
job_id = os.environ.get('SLURM_JOB_ID', 'default_version') # get slurm_id

from dataset.datasets import AdsSlabDataset
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

from trainer.pl_verify import pl_module_v

print("===========================Test Start===============================")
dmae_threshold = 0.05
data_path = f'data/cathub_label_{int(dmae_threshold * 100)}/val.pkl'
dataset = AdsSlabDataset(data_path)
loader = DataLoader(dataset, batch_size=512, shuffle=False)
iterable_loader = iter(loader) 

ckpt_path = 'tb_logs/adspainn-verification/116620/checkpoints/scorenet-epoch=014-avg_val_loss=0.104.ckpt'
bc_model = pl_module_v.load_from_checkpoint(ckpt_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bc_model = bc_model.to(device)
bc_model.eval()

print('dmae_threshold:', dmae_threshold)
print('data_path:', data_path)
print('ckpt_path:', ckpt_path)

pred_list = []
label_list = []
prob_list = []

with torch.no_grad():
    for batch in tqdm(iterable_loader, desc='computing metrics'):
        batch = batch.to(device)
        probs = bc_model.compute_confidence(batch).detach().cpu().numpy().reshape(-1)
        labels = batch.label.detach().cpu().numpy().reshape(-1)
        preds = (probs > 0.5).astype(int)
        pred_list.extend(preds.tolist())
        label_list.extend(labels.tolist())
        prob_list.extend(probs.tolist())

print(classification_report(label_list, pred_list, target_names=['negative', 'positive']))

fpr, tpr, _ = roc_curve(label_list, prob_list)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.legend(loc='lower right', fontsize=16)
plt.tight_layout()
roc_path = f'roc_curve_{job_id}.png'
plt.savefig(roc_path)
plt.close()

print(f'ROC curve saved to {roc_path}')

print("===========================Test End=================================")
