import torch
from torch_scatter import scatter

from model.utils import ScaledSiLU
from model import get_full_edges_index, get_pbc_edges_index, batch_matmul


class BinaryClassificationModel(torch.nn.Module):
    def __init__(self, network: torch.nn.Module, model_config: dict, loss_type: str = 'bce'):
        super(BinaryClassificationModel, self).__init__()
        self.network = network
        self.cutoff = model_config.get('cutoff', 4.0)
        net_channels = model_config['out_channels']
        self.loss_type = loss_type
        if loss_type == 'bce':
            self.node_out = torch.nn.Sequential(
                torch.nn.Linear(net_channels, net_channels // 2),
                ScaledSiLU(),
                torch.nn.Linear(net_channels // 2, 1),
                torch.nn.Sigmoid()
            )
            self.loss_fn = torch.nn.BCELoss()
        elif loss_type == 'bce_logits':
            self.node_out = torch.nn.Sequential(
                torch.nn.Linear(net_channels, net_channels // 2),
                ScaledSiLU(),
                torch.nn.Linear(net_channels // 2, 1),
            )
            pos_weight = torch.tensor([model_config['weight']])
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise ValueError(f"loss_type {loss_type} not supported")

    def forward(self, batch):
        cell = batch['cell']
        fixed_idx = batch['fixed']
        mask_ads = batch['mask_ads']
        pos = batch.pos_generated
        atomic_numbers = batch.atomic_numbers
        node2graph = batch.batch
        # sample same t value for all atoms
        t = torch.zeros(pos.shape[0]).to(pos.device)
        pos_frac = batch_matmul(pos, torch.inverse(cell), node2graph) % 1 # for PPC
        pos = batch_matmul(pos_frac, cell, node2graph)
        edge_index, dist, dist_vec = get_pbc_edges_index(
            pos, node2graph, cell, cutoff=self.cutoff
        )

        node_features = self.network(
            atomic_numbers,
            pos_frac,
            t,
            edge_index,
            dist,
            dist_vec,
            mask_ads = mask_ads,
            fixed_idx = fixed_idx
        )
        
        node_out = self.node_out(node_features)
        prediction = scatter(node_out, node2graph, dim=0, reduce='mean')
        loss = self.loss_fn(prediction, batch.label.float().unsqueeze(1))
        return loss, prediction

    def calculate_accuracy(self, pred, target):
        if self.loss_type == 'bce_logits':
            pred = torch.sigmoid(pred)
        target = target.unsqueeze(1)
        pred = torch.round(pred)
        correct = pred.eq(target).sum()
        return correct / target.shape[0]
