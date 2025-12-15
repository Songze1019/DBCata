import io
from typing import Optional

from torch import Tensor

import py3Dmol
from pymatgen.core.structure import Molecule 
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.io import write


def draw_mol_xyz(file_path):
    with open(file_path, 'r') as fo:
        xyz = fo.read()
    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz, 'xyz')
    view.setStyle({'sphere': {'scale': 0.35}, 'stick': {'radius': 0.20}})
    view.zoomTo()
    view.show()
    
    
def draw_mol_pymatgen(mol: Molecule):
    xyz_str = mol.to(fmt="xyz")
    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_str, 'xyz')
    view.setStyle({'sphere': {'scale': 0.35}, 'stick': {'radius': 0.20}})
    view.zoomTo()
    view.show()

def data_to_ase(data: Tensor) -> Atoms:
    ele = data[:, -1].long().cpu().numpy()
    pos = data[:, :3].cpu().numpy()
    atoms = Atoms(positions=pos, numbers=ele)
    return atoms


def ase_to_pymatgen(data: Atoms) -> Molecule:
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_molecule(data)
    return atoms


def data_to_pymatgen(data: Tensor) -> Molecule:
    atoms = data_to_ase(data)
    atoms = ase_to_pymatgen(atoms)
    return atoms


def draw_absorbate_xyz(mol_xyz: str, cell = Optional):
    viewer = py3Dmol.view(1200, 512)
    viewer.addModel(mol_xyz, "xyz")
    viewer.setStyle({'stick': {'radius': 0.10}, "sphere": {"scale": 0.50}})
    viewer.zoomTo()
    return viewer


def draw_absorbate_pymatgen(mol: Molecule):
    xyz_str = mol.to(fmt="xyz")
    viewer = py3Dmol.view(1200, 512)
    viewer.addModel(xyz_str, "xyz")
    viewer.setStyle({'stick': {'radius': 0.10}, "sphere": {"scale": 0.50}})
    viewer.zoomTo()
    return viewer


def draw_ase_atoms(atoms: Atoms):
    f = io.StringIO()
    
    write(f, atoms, format='xyz')
    
    xyz_str = f.getvalue()
    
    viewer = py3Dmol.view(width=1200, height=512)
    viewer.addModel(xyz_str, "xyz")
    viewer.setStyle({'stick': {'radius': 0.10}, "sphere": {"scale": 0.50}})
    viewer.zoomTo()
    
    return viewer
