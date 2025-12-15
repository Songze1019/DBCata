from functools import partial
from typing import Tuple

import torch
import numpy as np
from tqdm import tqdm
from torch_scatter import scatter_add

from model.graph_tools import get_full_edges_index, get_pbc_edges_index
from model.utils import extract, batch_matmul


class EquivBBDM(torch.nn.Module):
    def __init__(
        self,
        score_model: torch.nn.Module,
        utils_config: dict,
        loss_type: str='l1',
        fixed: bool=False,
        frac_noise: bool=True,
        train_objective: str='grad',
        coord: str='fractional',
    ):
        super(EquivBBDM, self).__init__()
        self.model = score_model
        self.loss_type = loss_type
        self.utils_config = utils_config
        self.fixed = fixed
        self.train_objective = train_objective
        self.frac_noise = frac_noise
        # coordination mode
        if coord == 'fractional':
            self.fractional_mode = True
        elif coord == 'cartesian':
            self.fractional_mode = False
            self.cutoff = self.model.cutoff
        else:
            raise ValueError(f"Unknown coord mode '{coord}'")
        # register schedule parameters
        self.register_schedule()
        
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
            0, self.utils_config['num_timesteps'], (len(cell),), device=cell.device
        ).long()
        t = t[node2graph]
        return self.p_losses(
            atomic_numbers, pos_relaxed, pos_init, cell, node2graph, fixed_idx, t, mask_ads
        )
    
    def p_losses(self, atomic_numbers, pos_relaxed, pos_init, cell, node2graph, fixed_idx, t, mask_ads):
        assert pos_relaxed.shape == pos_init.shape
        noise = torch.randn_like(pos_relaxed, device=pos_relaxed.device)
        
        if self.fractional_mode:
            # fractional noise
            if self.frac_noise:
                noise = batch_matmul(noise, torch.inverse(cell), node2graph)
                
            # fractional coordination and fractional objective
            pos_init = batch_matmul(pos_init, torch.inverse(cell), node2graph)
            pos_relaxed = batch_matmul(pos_relaxed, torch.inverse(cell), node2graph)
            edge_index = get_full_edges_index(node2graph, remove_self_edge=True)
            delta = pos_init - pos_relaxed
            pos_t, objective = self.q_sample(pos_relaxed, pos_init, t, noise, delta, fixed_idx)
            
            # score function model
            objective_recon = self.model(
                atomic_numbers,
                t,
                pos_t,
                edge_index,
                mask_ads = mask_ads,
                fixed_idx = fixed_idx,
            )
        else:
            # cartesian coordination and cartesian objective
            delta_frac = batch_matmul((pos_init - pos_relaxed), torch.inverse(cell), node2graph)
            delta = batch_matmul(
                (delta_frac - torch.floor(delta_frac + 0.5)), cell, node2graph
            )
            pos_t, objective = self.q_sample(pos_relaxed, pos_init, t, noise, delta, fixed_idx)
            pos_t_frac = batch_matmul(pos_t, torch.inverse(cell), node2graph) % 1 # for PPC
            pos_t = batch_matmul(pos_t_frac, cell, node2graph)
            edge_index, dist, dist_vec = get_pbc_edges_index(
                pos_t, node2graph, cell, cutoff=self.cutoff
            )

            # score function model
            objective_recon = self.model(
                atomic_numbers,
                pos_t_frac,
                t,
                edge_index,
                dist,
                dist_vec,
                mask_ads = mask_ads,
                fixed_idx = fixed_idx
            )

        if self.fixed:
            objective_fixed = objective[~ fixed_idx]
            objective_recon_fixed = objective_recon[~ fixed_idx]
            objective_recon[fixed_idx] = 0.
            # loss_fn
            if self.loss_type == 'l1':
                recloss = (objective_fixed - objective_recon_fixed).abs().mean()
            elif self.loss_type == 'l2':
                recloss = ((objective_fixed - objective_recon_fixed) ** 2).mean()
            elif self.loss_type == 'l2mae':
                node_loss = torch.sum((objective_fixed - objective_recon_fixed) ** 2, dim=1)
                node2graph_fixed = node2graph[~ fixed_idx]
                batch_loss = scatter_add(node_loss, node2graph_fixed, dim=0, dim_size=len(cell))
                recloss = batch_loss.mean()
            else:
                raise ValueError(f"Unknown loss_type {self.loss_type}")
        else:            
            # loss_fn
            if self.loss_type == 'l1':
                recloss = (objective - objective_recon).abs().mean()
            elif self.loss_type == 'l2':
                recloss = ((objective - objective_recon) ** 2).mean()
            else:
                raise ValueError(f"Unknown loss_type {self.loss_type}")

        pos_relaxed_recon = self.predict_pos_relaxed_from_objective(
            pos_t, pos_init, t, objective_recon
        )
        log_dict = {
            'loss': recloss,
            'pos_relaxed_recon': pos_relaxed_recon,
        }
        
        return recloss, log_dict
        
    def q_sample(self, pos_relaxed, pos_init, t, noise, delta, fixed_idx):
        m_t = extract(self.m_t, t, pos_relaxed.shape)
        var_t = extract(self.variance_t, t, pos_relaxed.shape)
        sigma_t = torch.sqrt(var_t)
        
        if self.fixed:
            noise[fixed_idx] = 0.
        
        if self.fractional_mode:
            # fractional coordination and fractional objective
            if self.train_objective == 'grad':
                objective = m_t * (delta - torch.floor(delta + 0.5)) + sigma_t * noise
            elif self.train_objective == 'noise':
                objective = noise
            elif self.train_objective == 'ysubx':
                objective = delta - torch.floor(delta + 0.5)
            else:
                raise ValueError(f"Unknown train_objective {self.train_objective}")
        else:
            # periodic boundary condition
            if self.train_objective == 'grad':
                objective = m_t * delta + sigma_t * noise
            elif self.train_objective == 'noise':
                objective = noise
            elif self.train_objective == 'ysubx':
                objective = delta
            else:
                raise ValueError(f"Unknown train_objective {self.train_objective}")
        
        return(
            objective + pos_relaxed, # only for grad mode
            objective
        )
        
    def predict_pos_relaxed_from_objective(self, pos_t, pos_init, t, objective_recon):
        if self.train_objective == 'grad':
            pos_relaxed_recon = pos_t - objective_recon
        elif self.train_objective == 'noise':
            m_t = extract(self.m_t, t, pos_t.shape)
            var_t = extract(self.variance_t, t, pos_t.shape)
            sigma_t = torch.sqrt(var_t)
            pos_relaxed_recon = (pos_t - m_t * pos_init - sigma_t * objective_recon) / (1. - m_t)
        elif self.train_objective == 'ysubx':
            pos_relaxed_recon = pos_init - objective_recon
        else:
            raise ValueError(f"Unknown train_objective {self.train_objective}")
        
        return pos_relaxed_recon
        
    @torch.no_grad()
    def ground_truth_sample(self, batch):
        assert batch['pos_relaxed'].shape == batch['pos'].shape
        node2graph = batch['batch']
        fixed_idx = batch['fixed']
        cell = batch['cell']
        if self.fractional_mode:
            pos_init = batch_matmul(batch['pos'], torch.inverse(batch['cell']), node2graph)
            pos_relaxed = batch_matmul(batch['pos_relaxed'], torch.inverse(batch['cell']), node2graph)
        else:
            pos_init = batch['pos']
            pos_relaxed = batch['pos_relaxed']
        
        trajs = [batch['pos_relaxed']]
        for i in tqdm(range(self.utils_config['num_timesteps'])):
            t = torch.full((pos_init.shape[0], ), i, device=pos_init.device, dtype=torch.long)
            noise = torch.randn_like(pos_relaxed, device=pos_relaxed.device)
            if self.fractional_mode:
                if self.frac_noise:
                    noise = batch_matmul(noise, torch.inverse(batch['cell']), node2graph)
                delta = pos_init - pos_relaxed
                pos_t, _ = self.q_sample(pos_relaxed, pos_init, t, noise, delta, fixed_idx)
                pos_t = pos_t % 1
                trajs.append(batch_matmul(pos_t, batch['cell'], node2graph))
            else:
                delta_frac = batch_matmul((pos_init - pos_relaxed), torch.inverse(cell), node2graph)
                delta = batch_matmul(
                    (delta_frac - torch.floor(delta_frac + 0.5)), cell, node2graph
                )
                pos_t, _ = self.q_sample(pos_relaxed, pos_init, t, noise, delta, fixed_idx)
                pos_t_frac = batch_matmul(pos_t, torch.inverse(batch['cell']), node2graph) % 1
                pos_t = batch_matmul(pos_t_frac, batch['cell'], node2graph)
                trajs.append(pos_t)
                
        return trajs
    
    @torch.no_grad()
    def sample(self, batch):
        node2graph = batch['batch']
        cell = batch['cell']
        fixed_idx = batch['fixed']
        mask_ads = batch['mask_ads']
        
        if self.fractional_mode:
            pos_init = batch_matmul(batch['pos'], torch.inverse(cell), node2graph) 
            pos_t = pos_init.clone()
            edge_index = get_full_edges_index(node2graph, remove_self_edge=False)
        else:
            pos_init = batch['pos']
            pos_t = pos_init.clone()
            pos_t_frac = batch_matmul(pos_t, torch.inverse(cell), node2graph) % 1
        trajs = [batch['pos']]
        trajs_recon = []
        
        for i in tqdm(range(len(self.steps)), desc='Sampling BBDM'):
            noise = torch.randn_like(pos_t, device=pos_t.device)
            if self.fractional_mode:
                if self.frac_noise:
                    noise = batch_matmul(noise, torch.inverse(cell), node2graph)
                pos_t, _ = self.p_sample(
                    batch['atomic_numbers'],
                    pos_t,
                    pos_t,
                    pos_init,
                    i,
                    edge_index=edge_index,
                    noise=noise,
                    cell=cell,
                    node2graph=node2graph, 
                    fixed_idx=fixed_idx,
                    mask_ads=mask_ads,
                )
                pos_t = batch_matmul(pos_t % 1, cell, node2graph)
                trajs.append(pos_t)
            # for cartesian mode
            else:
                pos_t, recon = self.p_sample(
                    batch['atomic_numbers'],
                    pos_t,
                    pos_t_frac,
                    pos_init,
                    i,
                    edge_index=None,
                    noise=noise,
                    cell=cell,
                    node2graph=node2graph, 
                    fixed_idx=fixed_idx,
                    mask_ads=mask_ads,
                )
                pos_t_frac = batch_matmul(pos_t, torch.inverse(cell), node2graph) % 1
                pos_t = batch_matmul(pos_t_frac, cell, node2graph)
                trajs.append(pos_t)
                
                recon_frac = batch_matmul(recon, torch.inverse(cell), node2graph) % 1
                recon = batch_matmul(recon_frac, cell, node2graph)
                trajs_recon.append(recon)
                
        return pos_t, trajs, trajs_recon
            
    def p_sample(
        self,
        atomic_numbers,
        pos_t,
        pos_t_frac,
        pos_init,
        i,
        edge_index,
        noise,
        cell,
        node2graph,
        fixed_idx,
        mask_ads,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.steps[i] == 0:
            t = torch.full((pos_init.shape[0], ), 0, device=pos_init.device, dtype=torch.long)
            
            if not self.fractional_mode:
                edge_index, dist, dist_vec = get_pbc_edges_index(
                    pos_t, node2graph, cell, cutoff=self.cutoff
                )
                
                objective_recon = self.model(
                    atomic_numbers,
                    pos_t_frac,
                    t,
                    edge_index,
                    dist,
                    dist_vec,
                    mask_ads = mask_ads,
                    fixed_idx = fixed_idx
                )
            else:
                objective_recon = self.model(
                atomic_numbers,
                t,
                pos_t,
                edge_index,
                mask_ads = mask_ads,
                fixed_idx = fixed_idx,
                )
            
            if self.fixed:
                objective_recon[fixed_idx] = 0.
            pos_relaxed_recon = self.predict_pos_relaxed_from_objective(pos_t, pos_init, t, objective_recon)
            return pos_relaxed_recon, pos_relaxed_recon
        else:
            t = torch.full((pos_t.shape[0], ), self.steps[i], device=pos_t.device, dtype=torch.long)
            n_t = torch.full((pos_t.shape[0], ), self.steps[i+1], device=pos_t.device, dtype=torch.long)
            
            if not self.fractional_mode:
                edge_index, dist, dist_vec = get_pbc_edges_index(
                    pos_t, node2graph, cell, cutoff=self.cutoff
                )
                
                objective_recon = self.model(
                    atomic_numbers,
                    pos_t_frac,
                    t,
                    edge_index,
                    dist,
                    dist_vec,
                    mask_ads = mask_ads,
                    fixed_idx = fixed_idx
                )
            else:
                objective_recon = self.model(
                atomic_numbers,
                t,
                pos_t,
                edge_index,
                mask_ads = mask_ads,
                fixed_idx = fixed_idx,
                )
            
            if self.fixed:
                objective_recon[fixed_idx] = 0.
                noise[fixed_idx] = 0.
            pos_relaxed_recon = self.predict_pos_relaxed_from_objective(
                pos_t, pos_init, t, objective_recon
            )
            
            m_t = extract(self.m_t, t, pos_t.shape)
            m_nt = extract(self.m_t, n_t, pos_t.shape)
            var_t = extract(self.variance_t, t, pos_t.shape)
            var_nt = extract(self.variance_t, n_t, pos_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.utils_config['eta']
            
            # TODO(2024.08.21): check the formula, FIX Perioidic Boundary Condition BUG 
            '''
            x_tminus_mean = (1. - m_nt) * pos_relaxed_recon + m_nt * pos_init + \
                torch.sqrt((var_nt - sigma2_t) / var_t) * \
                    (pos_t - (1. - m_t) * pos_relaxed_recon - m_t * pos_init)
            
            '''
            
            cell_inv = torch.inverse(cell)
            delta_r = batch_matmul((pos_init - pos_relaxed_recon), cell_inv, node2graph)
            tilde_r = batch_matmul(delta_r - torch.floor(delta_r + 0.5), cell, node2graph)
            
            delta_xt = batch_matmul(
                (pos_t - (pos_relaxed_recon + m_t * tilde_r)),
                cell_inv,
                node2graph
            )
            tilde_xt = batch_matmul(delta_xt - torch.floor(delta_xt + 0.5), cell, node2graph)

            x_tminus_mean = pos_relaxed_recon + m_nt * tilde_r + \
                torch.sqrt((var_nt - sigma2_t) / var_t) * tilde_xt
            
        return x_tminus_mean + sigma_t * noise, pos_relaxed_recon
    
    def register_schedule(self):
        T = self.utils_config['num_timesteps']
        
        # m_t
        if self.utils_config['sample_mt_mode'] == 'linear':
            m_t = np.linspace(0.001, 0.999, T)
        elif self.utils_config['sample_mt_mode'] == 'sin':
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999    
        elif self.utils_config['sample_mt_mode'] == 'cosine':
            eps = 1e-2
            m_t = 0.5 * (np.cos(np.linspace(np.pi - eps, eps, T)) + 1)
        else:
            raise ValueError(f"Unknown mode {self.utils_config['sample_mt_mode']}")
        m_tminus = np.append(0, m_t[:-1])
        
        # variance
        variance_t = 2. * (m_t - m_t ** 2) * self.utils_config['max_var']
        variance_tminus = np.append(0, variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t
        
        # register buffer
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))
        
        # sample steps
        if self.utils_config['skip_sample']:
            if self.utils_config['sample_mode'] == 'linear':
                midsteps = torch.arange(
                    T - 1, 1, step=-((T - 1) / (self.utils_config['sample_steps'] - 2))
                ).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.utils_config['sample_mode'] == 'cosine':
                steps = np.linspace(start=0, stop=T, num=self.utils_config['sample_steps'] + 1)
                steps = (np.cos(steps / T * np.pi) + 1.) / 2. * T
                self.steps = torch.from_numpy(steps).long()
            else:
                raise ValueError(f"Unknown mode {self.utils_config['sample_mode']}")
        else:
            self.steps = torch.arange(T-1, -1, -1)

        
        
    
