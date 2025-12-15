from torch_geometric.data import Data
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


ADS_ELEMENTS = [1, 6, 7, 8, 16]

def split_slab(data: Data):
    mol_idx = [x.item() in ADS_ELEMENTS for x in data.atomic_numbers]
    mol = Data(atomic_numbers=data.atomic_numbers[mol_idx], pos=data.pos[mol_idx])
    slab = Data(atomic_numbers=data.atomic_numbers[~mol_idx], pos=data.pos[~mol_idx])
    return mol, slab

def get_sites_in_order(slab: Data):
    struct = Structure(species=slab.atomic_numbers, coords=slab.pos, coords_are_cartesian=True)
    asf = AdsorbateSiteFinder(struct)
    sites = asf.find_adsorption_sites()
    return sites
    
def get_structure():
    raise NotImplementedError
