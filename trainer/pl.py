import os

import lightning as L
import torch
from torch.optim.lr_scheduler import ExponentialLR

from model.utils import ClipQueue, get_grad_norm, batch_matmul
from model.io import write_batch_xyz
from model.en_bbdm import EquivBBDM
from model.en_rf import EquivRectifiedFlow
from model.dmae import calc_dmae_batch_mic
# from model.rmsd_ import rmsd_batch_pbc


class pl_module(L.LightningModule):
    def __init__(self, train_config, model_config, utils_config, data_config):
        super().__init__()
        self.save_hyperparameters()
        self.train_config = train_config
        self.utils_config = utils_config
        self.data_config = data_config
        # equivariant trainsition kernel
        self.score_model = model_config['model'](**model_config)
        # diffusion bridge or rectified flow
        if self.train_config['flow'] == 'bbdm':
            self.flow = EquivBBDM(
                self.score_model,
                self.utils_config,
                loss_type=self.train_config['loss_type'],
                fixed=self.train_config['fixed'],
                frac_noise=self.train_config['frac_noise'],
                train_objective=self.train_config['train_objective'],
                coord=self.train_config['coord'],
            )
        elif self.train_config['flow'] == 'rf':
            self.flow = EquivRectifiedFlow(
                self.score_model,
                self.utils_config,
                loss_type=self.train_config['loss_type'],
                fixed=self.train_config['fixed'],
                frac_noise=self.train_config['frac_noise'],
                coord=self.train_config['coord'],
            )
        else:
            raise ValueError("Invalid flow type")
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        if self.train_config['clip_grad']:
            self.gradnorm_queue = ClipQueue()
            self.gradnorm_queue.add(1000)
        
        if 'ema' in self.train_config and self.train_config['ema']:
            self.ema = EMA(self.flow, decay=self.train_config['ema_decay'])

    def compute_loss(self, batch):
        loss = self.flow(batch)[0]
        return loss
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        if (self.current_epoch + 1) % self.utils_config['sample_per_epoch'] == 0 and batch_idx == 0:
            with torch.no_grad():
                # sample from the model
                mols_pos, node2graph, _ = self.validate_sample(batch)
                # write_outputs if True
                if self.train_config['write_outputs']:
                    time_point = self.utils_config['timepoint']
                    save_dir = f'samples_train/{time_point}/epoch_{self.current_epoch}'
                    os.makedirs(save_dir, exist_ok=True)
                    mask = torch.bincount(node2graph)
                    write_batch_xyz(save_dir, batch['atomic_numbers'], mols_pos, mask)
                    print(f"Epoch {self.current_epoch}: saved samples to {save_dir}")
                # calculate D-MAE
                pos_r_frac = batch_matmul(
                    batch['pos_relaxed'], torch.inverse(batch['cell']), batch.batch
                )
                pos_g_frac = batch_matmul(mols_pos, torch.inverse(batch['cell']), node2graph)
                dmae_train = calc_dmae_batch_mic(pos_r_frac, pos_g_frac, batch['cell'], node2graph)
                self.log('D-MAE_Train', dmae_train, on_step=False, on_epoch=True)
                # calculate RMSD
                # rmsd_train = rmsd_batch_pbc(pos_r_frac, pos_g_frac, batch['cell'], node2graph)
                # rmsd_train_surface = rmsd_batch_pbc(
                #     pos_r_frac, pos_g_frac, batch['cell'], node2graph, batch['fixed']
                # )
                # self.log('RMSD-Train', rmsd_train, on_step=False, on_epoch=True)
                # self.log('RMSD-SURFACE-Train', rmsd_train_surface, on_step=False, on_epoch=True)
        # forward and compute loss
        loss = self.compute_loss(batch)
        self.training_step_outputs.append(loss)
        return loss
        
    def on_validation_start(self) -> None:
        # update EMA
        if 'ema' in self.train_config and self.train_config['ema']:
            self.ema.update(self.flow)
            self.ema.apply(self.flow)
            
    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop. It is independent of forward
        with torch.no_grad():
            if (self.current_epoch + 1) % self.utils_config['sample_per_epoch'] == 0 and batch_idx == 0:
                # sample from the model
                mols_pos, node2graph, _ = self.validate_sample(batch)
                # write_outputs if True
                if self.train_config['write_outputs']:
                    time_point = self.utils_config['timepoint']
                    save_dir = f'samples_val/{time_point}/epoch_{self.current_epoch}'
                    os.makedirs(save_dir, exist_ok=True)
                    mask = torch.bincount(node2graph)
                    write_batch_xyz(save_dir, batch['atomic_numbers'], mols_pos, mask)
                    print(f"Epoch {self.current_epoch}: saved samples to {save_dir}")
                # calculate D-MAE
                pos_r_frac = batch_matmul(
                    batch['pos_relaxed'], torch.inverse(batch['cell']), batch.batch
                )
                pos_g_frac = batch_matmul(mols_pos, torch.inverse(batch['cell']), node2graph)
                dmae_train = calc_dmae_batch_mic(pos_r_frac, pos_g_frac, batch['cell'], node2graph)
                self.log('D-MAE_Validation', dmae_train, on_step=False, on_epoch=True)
                # calculate RMSD
                # rmsd_train = rmsd_batch_pbc(pos_r_frac, pos_g_frac, batch['cell'], node2graph)
                # rmsd_train_surface = rmsd_batch_pbc(
                #     pos_r_frac, pos_g_frac, batch['cell'], node2graph, batch['fixed']
                # )
                # self.log('RMSD-Validation', rmsd_train, on_step=False, on_epoch=True)
                # self.log('RMSD-SURFACE-Validation', rmsd_train_surface, on_step=False, on_epoch=True)
            # forward and compute loss but not backpropagate
            loss = self.compute_loss(batch)
            self.validation_step_outputs.append(loss)
            return loss
    
    def validate_sample(self, batch):
        relaxed_pos_batch, trajs = self.flow.sample(batch)
        return relaxed_pos_batch, batch.batch, trajs
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True)
        print(f"Epoch {self.current_epoch}: avg_val_loss = {avg_loss}")
        torch.cuda.empty_cache() 
        self.validation_step_outputs.clear()
        # restore EMA
        if 'ema' in self.train_config and self.train_config['ema']:
            self.ema.restore(self.flow)
            
    # Executed on the end of each training + validation epoch
    def on_train_epoch_end(self):
        outputs = self.training_step_outputs
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_train_loss', avg_loss, on_step=False, on_epoch=True)
        print(f"Epoch {self.current_epoch}: avg_train_loss = {avg_loss}")
        self.training_step_outputs.clear()
        print('max_memory_allocated:{}GB'.format(torch.cuda.max_memory_allocated() / 1024 ** 3))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.flow.parameters(), lr=self.train_config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=self.train_config['schedule_gamma'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val,
        gradient_clip_algorithm,    
    ):
        if not self.train_config['clip_grad']:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        # max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std()
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std() # modified
        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g["params"]]
        # check grad
        #grad = [p.grad for p in params ]
        #print(grad)
        grad_norm = get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer, 
            gradient_clip_val=max_grad_norm,
            gradient_clip_algorithm="norm",
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}"
            )
 
            
class EMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        # Initialize EMA parameters on the same device as the model's parameters
        self.ema_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup_params = self.ema_params
        for param in self.ema_params.values():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema_param = self.ema_params[name]
                    # Ensure EMA parameter is on the same device as the model's parameter
                    if ema_param.device != param.device:
                        self.ema_params[name] = ema_param.to(param.device)
                        ema_param = self.ema_params[name]
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data * (1 - self.decay))

    def apply(self, model):
        # Backup current model parameters
        self.backup_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        # Apply EMA parameters to the model
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.ema_params[name])

    def restore(self, model):
        # Restore original model parameters
        for name, param in model.named_parameters():
            if name in self.backup_params and param.requires_grad:
                param.data.copy_(self.backup_params[name])
        self.backup_params = None

    def to(self, device):
        # Optional: Move EMA parameters to a specified device
        for name in self.ema_params:
            self.ema_params[name] = self.ema_params[name].to(device)
        if self.backup_params:
            for name in self.backup_params:
                self.backup_params[name] = self.backup_params[name].to(device)
