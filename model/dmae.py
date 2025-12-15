from typing import Optional

import torch
from torch import Tensor
from ase import Atoms
import numpy as np

from model.utils import batch_matmul


def calc_dmae_mic_cart(pos1, pos2, cell):
    # mic: minimum image convention, borrowed from ASE
    atomic_numbers = [1] * len(pos1)
    atoms1 = Atoms(numbers=atomic_numbers, positions=pos1.cpu(), cell=cell.cpu(), pbc=True)
    atoms2 = Atoms(numbers=atomic_numbers, positions=pos2.cpu(), cell=cell.cpu(), pbc=True)
    dist1 = atoms1.get_all_distances(mic=True)
    dist2 = atoms2.get_all_distances(mic=True)
    return np.mean(np.abs(dist1 - dist2))


def calc_dmae_mic(pos1, pos2, cell):
    # mic: minimum image convention, borrowed from ASE
    atomic_numbers = [1] * len(pos1)
    pos1 = torch.matmul(pos1, cell)
    pos2 = torch.matmul(pos2, cell)
    atoms1 = Atoms(numbers=atomic_numbers, positions=pos1.cpu(), cell=cell.cpu(), pbc=True)
    atoms2 = Atoms(numbers=atomic_numbers, positions=pos2.cpu(), cell=cell.cpu(), pbc=True)
    dist1 = atoms1.get_all_distances(mic=True)
    dist2 = atoms2.get_all_distances(mic=True)
    return np.mean(np.abs(dist1 - dist2))


def calc_dmae_batch_mic(pos1, pos2, cell, batch):
    dmae = 0
    for i in range(batch.max() + 1):
        idx = batch == i
        dmae += calc_dmae_mic(pos1[idx], pos2[idx], cell[i]).item()
    return dmae / (batch.max() + 1).item()


def calc_dmae(pos1, pos2, cell, rectified: bool = True):
    EPS = 1e-4
    pos1_dist = pos1[None, :, :] - pos1[:, None, :]
    pos2_dist = pos2[None, :, :] - pos2[:, None, :]
    pos1_dist = pos1_dist - torch.floor(pos1_dist + 0.5 + EPS)
    pos2_dist = pos2_dist - torch.floor(pos2_dist + 0.5 + EPS)
    if rectified:
        e = 0.8
        shift = pos1_dist[(pos1_dist - pos2_dist).abs() > e]
        shift = shift - shift.sign()
        pos1_dist[(pos1_dist - pos2_dist).abs() > e] = shift
    cart_dist1 = torch.sqrt(torch.sum((torch.matmul(pos1_dist, cell) ** 2), dim=-1)) 
    cart_dist2 = torch.sqrt(torch.sum((torch.matmul(pos2_dist, cell) ** 2), dim=-1))
    return torch.mean((cart_dist1 - cart_dist2).abs())


def calc_dmae_batch(pos1, pos2, cell, batch, fixed: Optional[Tensor] = None):
    dmae = 0
    for i in range(batch.max() + 1):
        idx = batch == i
        if fixed is None:
            dame_ = calc_dmae(pos1[idx], pos2[idx], cell[i]).item()
        else:
            dame_ = calc_dmae_fixed(pos1[idx], pos2[idx], cell[i], fixed[idx]).item()
        dmae += min(dame_, 2.0)
    return dmae / (batch.max() + 1).item()


def calc_dmae_fixed(pos1, pos2, cell, fixed, rectified: bool = True):
    pos1 = pos1[~fixed]
    pos2 = pos2[~fixed]
    EPS = 1e-4
    pos1_dist = pos1[None, :, :] - pos1[:, None, :]
    pos2_dist = pos2[None, :, :] - pos2[:, None, :]
    pos1_dist = pos1_dist - torch.floor(pos1_dist + 0.5 + EPS)
    pos2_dist = pos2_dist - torch.floor(pos2_dist + 0.5 + EPS)
    if rectified:
        e = 0.8
        shift = pos1_dist[(pos1_dist - pos2_dist).abs() > e]
        shift = shift - shift.sign()
        pos1_dist[(pos1_dist - pos2_dist).abs() > e] = shift
    cart_dist1 = torch.sqrt(torch.sum((torch.matmul(pos1_dist, cell) ** 2), dim=-1)) 
    cart_dist2 = torch.sqrt(torch.sum((torch.matmul(pos2_dist, cell) ** 2), dim=-1))
    return torch.mean((cart_dist1 - cart_dist2).abs())


if __name__ == '__main__':
    from dataset import OCDataset
    from torch_geometric.loader import DataLoader
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = OCDataset("data/oc22/val_id.pkl", device)
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size)
    count = 0
    dae = 0
    for data in dataloader:
        pos1_frac = batch_matmul(data.pos, torch.inverse(data.cell), data.batch)
        pos2_frac = batch_matmul(data.pos_relaxed, torch.inverse(data.cell), data.batch)
        dmae = calc_dmae_batch(pos1_frac, pos2_frac, data.cell, data.batch)
        count += data.cell.size(0)
        dae += dmae.item() * data.cell.size(0)
    print(count)
    print(dae / count)
        
        