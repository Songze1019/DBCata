import os 
job_id = os.environ.get('SLURM_JOB_ID', 'default_version') # get slurm_id

import torch
from torch_geometric.loader import DataLoader
import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from pprint import pprint

from trainer.pl_cart import pl_module_cart
from dataset.datasets import AdsSlabDataset
from configs.config_fine_tune import train_config, data_config, utils_config


# create the dataset
train_dataset = AdsSlabDataset(data_config['path'] + 'train.pkl')
val_dataset = AdsSlabDataset(data_config['path'] + 'val.pkl')

#sub_dataset_train = Subset(train_dataset, range(100))
#sub_dataset_val = Subset(val_dataset, range(100))

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
config.update(data_config)
config.update(utils_config)
pprint(config)
print(f"train_size: {len(train_dataset)}")
print(f"val_size: {len(val_dataset)}")

# load the lightning model
checkpoint_path = train_config['checkpoint_path']
model = pl_module_cart.load_from_checkpoint(
    checkpoint_path,
    train_config=train_config,
    utils_config=utils_config,
    data_config=data_config,
)

# freeze params
if train_config.get('freeze', False):
    pl_module_cart.freeze_params()

# logger and callbacks
mode_name = utils_config['prefix'] + train_config['flow'] + '_' + train_config['model_name'] 
if not train_config['debugging']:
    logger = TensorBoardLogger("tb_logs", name=mode_name, version=job_id)
    checkpoint_callback = ModelCheckpoint(
    monitor="avg_val_loss",
    dirpath=f'tb_logs/{mode_name}/{job_id}/checkpoints',
    filename="scorenet-{epoch:03d}-{avg_val_loss:.3f}",
    every_n_epochs=100,
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
    limit_val_batches=20,
    # max_time="00:10:00:00",
)

# train the model
trainer.fit(model, train_loader, val_loader)
