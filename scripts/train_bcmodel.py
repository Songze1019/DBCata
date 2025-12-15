import os
from datetime import datetime
job_id = os.environ.get('SLURM_JOB_ID', 'default_version') # get slurm_id

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import lightning as L
from pprint import pprint
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from dataset.datasets import AdsSlabDataset
from trainer.pl_verify import pl_module_v
from model.net import EGNN, PaiNN, AdsPaiNN


# configurations
train_config = dict(
    debugging = False,
    epoch = 50,
    model_name = 'adspainn',
    batch_size = 512,
    lr = 1e-3,
    schedule_gamma = 0.998,
    num_workers = 0,
    clip_grad = True,
    loss_type = 'bce',
    weight_balance = True, 
)

if train_config['model_name'] == 'painn':
    # painn
    model_config = dict(
        model = PaiNN,
        hidden_channels = 64,
        out_channels = 16,
        num_layers = 3,
        n_frequencies = 10, 
        scalar = True, # True, False, using scalar xh_out or not
        weight = 0.1327, # weight for bce_logits
    )
elif train_config['model_name'] == 'egnn':
    # egnn
    model_config = dict(
        model = EGNN,
        hidden_dim = 256,
        latent_dim = 256,
        num_layers = 4,
        max_atoms = 100,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 40,
        ln = False,
        pred_scalar = False # True(invariant), False(equivariant)
    )
elif train_config['model_name'] == 'adspainn':
    # adspainn
    model_config = dict(
        model = AdsPaiNN,
        cutoff = 4.0,
        hidden_channels = 128,
        out_channels = 16,
        num_rbf = 128,
        rbf = {'name': 'gaussian'},
        envelope = {'name': 'polynomial', 'exponent': 5},
        num_layers = 3,
        n_frequencies = 40, # for fourier features
        scalar = True, # True, False, using scalar xh_out or not
        ftbasis = False, # [True] or False, using fourier basis or not
        weight = 0.1036, # weight for bce_logits
    ) 
else:
    raise ValueError(f"model_name {train_config['model_name']} not supported")

data_config = dict(
    path = 'data/cathub_label_5/', # path to the data
)

utils_config = dict(
    timepoint = datetime.now().strftime("%m-%d-%H:%M:%S"),
)

# create the dataset
# train_dataset = AdsSlabDataset(data_config['path'] + 'train.pkl')
# val_dataset = AdsSlabDataset(data_config['path'] + 'val.pkl')

all_dataset = AdsSlabDataset(data_config['path'] + 'all.pkl')
dataset_size = len(all_dataset)
indices = list(range(dataset_size))
split = int(0.8 * dataset_size)  #  80-20 train-val split

# shuffle the dataset before splitting
import numpy as np
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[:split], indices[split:]
train_dataset = Subset(all_dataset, train_indices)
val_dataset = Subset(all_dataset, val_indices)

# create the dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=train_config['batch_size'],
    num_workers=train_config['num_workers'],
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=train_config['batch_size'],
    num_workers=train_config['num_workers'],
    shuffle=False,
)

# print the config and data_size
config = train_config.copy()
config.update(model_config)
config.update(data_config)
config.update(utils_config)
pprint(config)
print(f"train_size: {len(train_dataset)}")
print(f"val_size: {len(val_dataset)}")

# create the lightning model
model = pl_module_v(train_config, model_config, utils_config, data_config)

# logger and callbacks
if not train_config['debugging']:
    logger = TensorBoardLogger(
        "tb_logs", name=train_config['model_name'] + '-verification', version=job_id
    )
    checkpoint_callback = ModelCheckpoint(
    monitor="avg_val_loss",
    dirpath=f'tb_logs/{train_config["model_name"]}' + '-verification' + f'/{job_id}/checkpoints',
    filename="scorenet-{epoch:03d}-{avg_val_loss:.3f}",
    every_n_epochs=5,
    save_top_k=-1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks = [checkpoint_callback, progress_bar, lr_monitor]
else:
    logger = None
    callbacks = None

# devices and strategy
strategy = 'ddp_find_unused_parameters_true' if torch.cuda.is_available() else 'auto'
num_gpus = torch.cuda.device_count()

# create the trainer
trainer = L.Trainer(
    fast_dev_run=train_config['debugging'], 
    max_epochs=train_config['epoch'],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    deterministic=False,
    logger=logger,
    devices=list(range(num_gpus)) if num_gpus > 1 else 1,
    strategy=strategy,
    log_every_n_steps=1,
    callbacks=callbacks,
    profiler=None,
    accumulate_grad_batches=1,
    limit_train_batches=None,
    limit_val_batches=None,
    # max_time="00:10:00:00",
)

# train the model
trainer.fit(model, train_loader, val_loader)
