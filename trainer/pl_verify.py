import lightning as L
import torch
from torch.optim.lr_scheduler import ExponentialLR

from model.utils import ClipQueue, get_grad_norm
from model.bcmodel import BinaryClassificationModel


class pl_module_v(L.LightningModule):
    def __init__(self, train_config, model_config, utils_config, data_config):
        super().__init__()
        self.save_hyperparameters()
        self.train_config = train_config
        self.utils_config = utils_config
        self.data_config = data_config
        
        if train_config['weight_balance']:
            train_config['loss_type'] = 'bce_logits'
        
        network = model_config['model'](**model_config)
        self.bc_model = BinaryClassificationModel(
            network,
            model_config,
            loss_type=train_config['loss_type'],
        )
        
        self.training_step_outputs_loss = []
        self.validation_step_outputs_loss = []
        self.training_step_outputs_acc = []
        self.validation_step_outputs_acc = []
        
        if self.train_config['clip_grad']:
            self.gradnorm_queue = ClipQueue()
            self.gradnorm_queue.add(1000)

    def compute_loss(self, batch):
        loss, pred = self.bc_model(batch)
        return loss, pred
    
    def compute_label(self, batch):
        loss, pred = self.bc_model(batch)
        if self.bc_model.loss_type == 'bce_logits':
            pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        return pred
    
    def compute_confidence(self, batch):
        loss, pred = self.bc_model(batch)
        if self.bc_model.loss_type == 'bce_logits':
            pred = torch.sigmoid(pred)
        return pred
    
    def training_step(self, batch, batch_idx):
        loss, pred = self.compute_loss(batch)
        acc = self.bc_model.calculate_accuracy(pred, batch.label)
        self.training_step_outputs_loss.append(loss)
        self.training_step_outputs_acc.append(acc)
        return loss
    
    def on_train_epoch_end(self):
        outputs = self.training_step_outputs_loss, self.training_step_outputs_acc
        avg_loss = torch.stack(outputs[0]).mean()
        avg_acc = torch.stack(outputs[1]).mean()
        self.log('avg_train_loss', avg_loss, on_step=False, on_epoch=True)
        self.log('avg_train_acc', avg_acc, on_step=False, on_epoch=True)
        print(f"Epoch {self.current_epoch}: avg_train_loss = {avg_loss}")
        print(f"Epoch {self.current_epoch}: avg_train_acc = {avg_acc}")
        self.training_step_outputs_loss.clear()
        self.training_step_outputs_acc.clear()
        print('max_memory_allocated:{}GB'.format(torch.cuda.max_memory_allocated() / 1024 ** 3))
    
    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop. It is independent of forward
        with torch.no_grad():
            loss, pred = self.compute_loss(batch)
            acc = self.bc_model.calculate_accuracy(pred, batch.label)
            self.validation_step_outputs_loss.append(loss)
            self.validation_step_outputs_acc.append(acc)
            return loss
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs_loss, self.validation_step_outputs_acc
        avg_loss = torch.stack(outputs[0]).mean()
        avg_acc = torch.stack(outputs[1]).mean()
        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True)
        self.log('avg_val_acc', avg_acc, on_step=False, on_epoch=True)
        print(f"Epoch {self.current_epoch}: avg_val_loss = {avg_loss}")
        print(f"Epoch {self.current_epoch}: avg_val_acc = {avg_acc}")
        torch.cuda.empty_cache() 
        self.validation_step_outputs_loss.clear()
        self.validation_step_outputs_acc.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.bc_model.parameters(), lr=self.train_config['lr'])
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
