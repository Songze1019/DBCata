from typing import Tuple, List

import torch
from tqdm import tqdm

from model.graph_tools import get_full_edges_index
from model.utils import extract, batch_matmul


class EquivRectifiedFlow(torch.nn.Module):
    def __init__(
        self,
        score_model: torch.nn.Module,
        utils_config: dict,
        loss_type: str='l1',
        fixed: bool=False,
        frac_noise: bool=True,
        coord: str='fractional',
    ):
        super(EquivRectifiedFlow, self).__init__()
        self.model = score_model
        self.loss_type = loss_type
        self.utils_config = utils_config
        self.fixed = fixed
        self.frac_noise = frac_noise
        if coord == 'fractional':
            self.fractional_mode = True
        elif coord == 'cartesian':
            self.fractional_mode = False
        else:
            raise ValueError(f"Unknown coord mode '{coord}'")
        
    def forward(self, batch):
        # initialize
        pos_init = batch['pos']
        pos_relaxed = batch['pos_relaxed']
        atomic_numbers = batch['atomic_numbers']
        cell = batch['cell']
        node2graph = batch['batch']
        fixed_idx = batch['fixed']
        mask_ads = batch['mask_ads']
        
        # sample t
        t = torch.randint(
            0, self.utils_config['num_timesteps'] + 1, (len(cell),), device=cell.device
        ).long()
        t = t[node2graph]
        t_frac = t / self.utils_config['num_timesteps'] # [0, 1]
        
        return self.p_losses(
            atomic_numbers, pos_relaxed, pos_init, cell, node2graph, fixed_idx, t_frac, mask_ads
        )
    
    def p_losses(self, atomic_numbers, pos_relaxed, pos_init, cell, node2graph, fixed_idx, t, mask_ads):
        assert pos_relaxed.shape == pos_init.shape
        
        # fractional coordination and objective vec
        pos_init = batch_matmul(pos_init, torch.inverse(cell), node2graph)
        pos_relaxed = batch_matmul(pos_relaxed, torch.inverse(cell), node2graph)
        
        pos_t, flow_vec = self.q_sample(pos_relaxed, pos_init, t)
        edge_index = get_full_edges_index(node2graph, remove_self_edge=True)
        
        theta_vec = self.model(
            atomic_numbers,
            t,
            pos_t,
            edge_index,
            cell,
            node2graph,
            mask_ads = mask_ads,
            fixed_idx = fixed_idx
        )

        if self.fixed:
            flow_vec_fixed = flow_vec[~ fixed_idx]
            theta_vec_fixed = theta_vec[~ fixed_idx]
            theta_vec[fixed_idx] = 0.
            # loss_fn
            if self.loss_type == 'l1':
                recloss = (flow_vec_fixed - theta_vec_fixed).abs().mean()
            elif self.loss_type == 'l2':
                loss = (flow_vec_fixed - theta_vec_fixed) ** 2
                node2graph = node2graph[~ fixed_idx]
                sum_loss =torch.zeros(max(node2graph) + 1, loss.size(1), device=loss.device)
                graph_idx = node2graph.unsqueeze(1).expand(-1, loss.size(1))
                sum_loss.scatter_add_(0, graph_idx, loss)
                node_counts = torch.bincount(node2graph, minlength=sum_loss.size(0)).unsqueeze(1)
                recloss = (sum_loss / node_counts).mean()
            else:
                raise ValueError(f"Unknown loss_type {self.loss_type}")
        else:            
            # loss_fn
            if self.loss_type == 'l1':
                recloss = (flow_vec - theta_vec).abs().mean()
            elif self.loss_type == 'l2':
                recloss = ((flow_vec - theta_vec) ** 2).mean()
            else:
                raise ValueError(f"Unknown loss_type {self.loss_type}")

        pos_relaxed_recon = self.predict_pos_relaxed_from_flowvec(pos_t, t, theta_vec)
        log_dict = {
            'loss': recloss,
            'pos_relaxed_recon': pos_relaxed_recon,
        }
        
        return recloss, log_dict
        
    def q_sample(self, pos_relaxed, pos_init, t_frac):
        t_frac = t_frac.unsqueeze(-1)
        flow_vec = pos_init - pos_relaxed
        pos_t = (1 - t_frac) * pos_relaxed + t_frac * pos_init
        
        return pos_t, flow_vec
        
    def predict_pos_relaxed_from_flowvec(self, pos_t, t_frac, flow_vec):
        t_frac = t_frac.unsqueeze(-1)
        pos_relaxed_recon = pos_t - t_frac * flow_vec
        
        return pos_relaxed_recon    
    
    @torch.no_grad()
    def sample(self, batch):
        node2graph = batch['batch']
        cell = batch['cell']
        fixed_idx = batch['fixed']
        mask_ads = batch['mask_ads']
        pos_init = batch_matmul(batch['pos'], torch.inverse(cell), node2graph) 
        pos_t = pos_init.clone()
        edge_index = get_full_edges_index(node2graph, remove_self_edge=True)
        trajs = [batch['pos']]
        
        if self.utils_config['skip_sample']:
            sample_steps = self.utils_config['sample_steps']
        else:
            sample_steps = self.utils_config['num_timesteps']
        dt = 1 / sample_steps
        sample_steps_list = range(sample_steps, 0, -1)
        
        for i in tqdm(sample_steps_list, desc='Sampling Rectified Flow_0'):
            t_frac = i / sample_steps
            t = torch.full((pos_t.shape[0], ), t_frac, device=pos_t.device, dtype=torch.long)
            theta_vec = self.model(
                batch['atomic_numbers'],
                t,
                pos_t,
                edge_index,
                cell,
                node2graph,
                mask_ads = mask_ads,
                fixed_idx = fixed_idx
            )
            if self.fixed:
                theta_vec[fixed_idx] = 0.
                pos_t = pos_t - theta_vec * dt
            else:
                pos_t = pos_t - theta_vec * dt
            trajs.append(batch_matmul(pos_t % 1, cell, node2graph))
        pos_relaxed_batch = trajs[-1]
        
        return pos_relaxed_batch, trajs
        
