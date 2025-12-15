from pathlib import Path
import pickle

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from ase.optimize import LBFGS
from ase import Atoms
from ase.constraints import FixAtoms
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from dataset.datasets import BaseDataset
from model import calc_dmae_mic, calc_rmsd_pbc


checkpoint_path = Path("UMA") / "uma-s-1p1.pt"
atomic_refs_path = Path("UMA") / "iso_atom_elem_refs.yaml"

data_output = []

# Set up your system as an ASE atoms object
stage = 'val'
data_path = Path("data/cathub_all/clean") / f"{stage}.pkl"
dataset = BaseDataset(data_path)
dmae, rmsd, error = [], [], 0

atom_refs = OmegaConf.load(atomic_refs_path)
predictor = load_predict_unit(checkpoint_path, device="cuda", atom_refs=atom_refs)
calc = FAIRChemCalculator(predictor, task_name="oc20")

for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
    atomic_numbers = data.atomic_numbers.numpy()
    cell = data.cell.squeeze().numpy()
    positions = data.pos.numpy()
    fixed = torch.where(data.fixed != 0)[0].numpy()

    slab = Atoms(numbers=atomic_numbers, cell=cell, positions=positions, pbc=True)
    constraint = FixAtoms(indices=fixed)
    slab.set_constraint(constraint)
    slab.calc = calc

    # Set up LBFGS dynamics object
    dyn = LBFGS(slab)
    try:
        dyn.run(0.05, 200)
        
        pos1 = torch.matmul(data.pos_relaxed, data.cell.squeeze().inverse())
        pos2 = torch.matmul(
            torch.tensor(slab.positions, dtype=torch.float32), data.cell.squeeze().inverse()
        )
        dmae.append(calc_dmae_mic(pos1, pos2, data.cell.squeeze()).item())
        
        shift = pos1 - pos2
        shift = shift - torch.floor(shift + 0.5)
        shift_cart = torch.matmul(shift, data.cell.squeeze())
        index = torch.ones(data.pos_relaxed.size(0), dtype=torch.bool)
        rmsd.append(calc_rmsd_pbc(shift_cart, index, None))
        
        print('dmae_item', dmae[-1])
        print('rmsd_item', rmsd[-1])
        
        data_item = data
        data_item.pos_relaxed = torch.tensor(slab.positions, dtype=torch.float32)
        data_output.append(data_item)
    except Exception as e:
        print(e)
        error += 1
        
        pos1 = torch.matmul(data.pos_relaxed, data.cell.squeeze().inverse())
        pos2 = torch.matmul(
            data.pos, data.cell.squeeze().inverse()
        )
        dmae.append(calc_dmae_mic(pos1, pos2, data.cell.squeeze()).item())
        
        shift = pos1 - pos2
        shift = shift - torch.floor(shift + 0.5)
        shift_cart = torch.matmul(shift, data.cell.squeeze())
        index = torch.ones(data.pos_relaxed.size(0), dtype=torch.bool)
        rmsd.append(calc_rmsd_pbc(shift_cart, index, None))
        
        print('dmae_item', dmae[-1])
        print('rmsd_item', rmsd[-1])
        
        # If optimization fails, use original positions
        data_item = data
        data_item.pos_relaxed = data.pos
        data_output.append(data_item)
        
        continue

dmae_mean = np.mean(dmae)
dame_min = np.min(dmae)
dmae_max = np.max(dmae)

rmsd_mean = np.mean(rmsd)
rmsd_min = np.min(rmsd)
rmsd_max = np.max(rmsd)

print("========================Inference Results============================")
print('data:', data_path)
print('DMAE:', dmae_mean)
print('DMAE_min:', dame_min)
print('DMAE_max:', dmae_max)
print('RMSD:', rmsd_mean)
print('RMSD_min:', rmsd_min)
print('RMSD_max:', rmsd_max)
print('error:', error)
print('data_output:', len(data_output))
print("==============================END====================================")

# save structures relaxed with UMA 
data_out_path = Path(f'mlp_eval') / 'uma.pkl'
data_out_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving relaxed structures to {data_out_path}")
pickle.dump(data_output, open(data_out_path, 'wb'))
