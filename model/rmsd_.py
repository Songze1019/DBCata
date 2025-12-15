from rmsd import rmsd_xyz_str
from pymatgen.core import Molecule
import numpy as np
from torch import Tensor
import torch
from typing import Optional
from model.utils import batch_matmul


def rmsd_batch(relaxed_pos, mols_pos, node2graph, atomic_numbers):
    predict_mols = [
        Molecule(species=atomic_numbers[node2graph == i].cpu().numpy(),
                 coords=mols_pos[node2graph == i].cpu().numpy())
        for i in range(node2graph.max().item() + 1)
    ]
    ground_truth_mols = [
        Molecule(species=atomic_numbers[node2graph == i].cpu().numpy(),
                 coords=relaxed_pos[node2graph == i].cpu().numpy())
        for i in range(node2graph.max().item() + 1)
    ]
    rmsd_all = [
        float(rmsd_xyz_str(mol1.to(fmt='xyz'), mol2.to(fmt='xyz')))
        for mol1, mol2 in zip(predict_mols, ground_truth_mols)
    ]
    
    return np.mean(np.array(rmsd_all).clip(0., 1.0))


def rmsd_batch_pbc(
    pos_r_frac,
    pos_g_frac,
    cell,
    node2graph,
    fixed: Optional[Tensor] = None
):
    shift = pos_r_frac - pos_g_frac
    shift = shift - torch.floor(shift + 0.5)
    shift_cart = batch_matmul(shift, cell, node2graph)
    rmsd_batch = 0.
    for i in range(cell.size(0)):
        idx = (node2graph == i)
        rmsd = calc_rmsd_pbc(shift_cart, idx, fixed)
        rmsd_batch += rmsd
    return rmsd_batch / cell.size(0)
        

def calc_rmsd_pbc(pos_shift, idx, fixed):
    pos_shift = pos_shift[idx]
    if fixed is not None:
        pos_shift = pos_shift[~fixed[idx]]
    return min(torch.sqrt(torch.sum(pos_shift ** 2, dim=-1).mean()).item(), 2.0)
      
    
    
    
