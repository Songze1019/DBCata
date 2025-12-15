import pickle

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader


ADS_ELEMENTS = torch.tensor([1, 6, 7, 8, 16])

class BaseDataset(Dataset):
    def __init__(self, path: str, device: str='cpu') -> None:
        super().__init__()
        self.path = path
        self.raw_data = pickle.load(open(path, "rb"))
        self.device = device
        
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        data_item = self.raw_data[idx].to(self.device)
        data_item.atomic_numbers = data_item.atomic_numbers.long()
        return data_item 
    
    
class AdsSlabDataset(BaseDataset):
    def __init__(self, path: str, device: str='cpu') -> None:
        super().__init__(path, device)
    
    def __getitem__(self, idx):
        data_item = self.raw_data[idx].to(self.device)
        data_item.fixed = data_item.fixed.bool()
        data_item.atomic_numbers = data_item.atomic_numbers.long()
        data_item.mask_ads = torch.isin(data_item.atomic_numbers, ADS_ELEMENTS)
        return data_item 


class DatasetFromList(Dataset):
    def __init__(self, data_list, device: str='cpu') -> None:
        super().__init__()
        self.device = device
        self.raw_data = data_list
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        data_item = self.raw_data[idx].to(self.device)
        data_item.fixed = data_item.fixed.bool()
        data_item.atomic_numbers = data_item.atomic_numbers.long()
        data_item.mask_ads = torch.isin(data_item.atomic_numbers, ADS_ELEMENTS)
        if "pos_generated" not in data_item:
            data_item.pos_generated = torch.zeros_like(data_item.pos)
        return data_item

    
if __name__ == "__main__":
    from torch.utils.data import Subset
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = AdsSlabDataset("data/cathub_all/clean/val.pkl", device)
    sub_dataset = Subset(dataset, range(100))
    print(len(sub_dataset), sub_dataset)
    dataloader = DataLoader(sub_dataset, batch_size=16)
    
    for data in dataloader:
        print(data)
        print(data.atomic_numbers)
        break    
