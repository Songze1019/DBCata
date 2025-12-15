from typing import List, Any
import os
import pickle

import torch
from tqdm import tqdm
from lightning import LightningModule
from torch_geometric.loader import DataLoader

from dataset import DatasetFromList
from model import (
    Timer,
    calc_dmae_batch,
    calc_dmae_batch_mic,
    calc_dmae_mic,
    batch_matmul,
    # rmsd_batch_pbc,
)


class InferenceModel:
    def __init__(
        self, 
        model: LightningModule, 
        data: List[Any], 
        batch_size: int,
        device: torch.device
    ):
        self.pl_model = model
        self.data_raw = data
        self.device = device
        self.batch_size = batch_size
        
        # adslabdataset -> dataloader
        self.process_dataloader(self.data_raw)
        
    # batch inference and batch processing
    def process_dataloader(self, data_list: List[Any]) -> None:
        dataset = DatasetFromList(data_list)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        
    # compute RMSD of surface atoms
    def compute_rmsd_surface(self, batch):
        pos_r_frac = batch_matmul(batch['pos_relaxed'], torch.inverse(batch['cell']), batch.batch)
        pos_g_frac = batch_matmul(batch['pos_generated'], torch.inverse(batch['cell']), batch.batch)
        return rmsd_batch_pbc(
            pos_r_frac, pos_g_frac, batch['cell'], batch.batch, batch['fixed']
        )
        
    # compute RMSD of all atoms
    def compute_rmsd(self, batch):
        pos_r_frac = batch_matmul(batch['pos_relaxed'], torch.inverse(batch['cell']), batch.batch)
        pos_g_frac = batch_matmul(batch['pos_generated'], torch.inverse(batch['cell']), batch.batch)
        return rmsd_batch_pbc(
            pos_r_frac, pos_g_frac, batch['cell'], batch.batch
        )
            
    # compute D-MAE
    def compute_dmae(self, batch):
        pos_r_frac = batch_matmul(batch['pos_relaxed'], torch.inverse(batch['cell']), batch.batch)
        pos_g_frac = batch_matmul(batch['pos_generated'], torch.inverse(batch['cell']), batch.batch)
        return calc_dmae_batch(
            pos_r_frac, pos_g_frac, batch['cell'], batch.batch
        )
        
    # compute D-MAE mic
    def compute_dmae_mic(self, batch):
        pos_r_frac = batch_matmul(batch['pos_relaxed'], torch.inverse(batch['cell']), batch.batch)
        pos_g_frac = batch_matmul(batch['pos_generated'], torch.inverse(batch['cell']), batch.batch)
        return calc_dmae_batch_mic(
            pos_r_frac, pos_g_frac, batch['cell'], batch.batch
        )
        
    # inference results of rmsd
    def compute_rmsd_results(self, surface: bool = False) -> None:
        rmsd = 0
        for idx, data in tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc='Computing RMSD of inference results'
        ): 
            data = data.to(self.device)
            if surface:
                rmsd += self.compute_rmsd_surface(data)
            else:
                rmsd += self.compute_rmsd(data)
        rmsd_mean = rmsd / len(self.dataloader)
        # RESULTS
        print("==========================RMSD Results===============================")
        print(f"RMSD mean: {rmsd_mean} (surface: {surface})")
        print("==============================END====================================")
    
    # inference results of dmae
    def compute_dmae_results(self) -> None:
        dmae = 0 
        for idx, data in tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc='Computing D-MAE of inference results'
        ):
            data = data.to(self.device)
            dmae += self.compute_dmae(data)
        dmae_mean = dmae / len(self.dataloader)
        # RESULTS
        print("==========================RMSD Results===============================")
        print(f"D-MAE mean: {dmae_mean}")
        print("==============================END====================================")
        
    # inference results of dmae
    def compute_dmae_mic_results(self) -> None:
        dmae = 0 
        for idx, data in tqdm(
            enumerate(self.dataloader),
            total=len(self.dataloader),
            desc='Computing D-MAE of inference results'
        ):
            data = data.to(self.device)
            dmae += self.compute_dmae_mic(data)
        dmae_mean = dmae / len(self.dataloader)
        # RESULTS
        print("==========================DMAE Results===============================")
        print(f"D-MAE mean: {dmae_mean}")
        print("==============================END====================================")
        
    # inference of BBDM
    def inference(self, steps: int, eta: float = 0.0, device: torch.device = 'cuda') -> List:
        timer = Timer()
        timer.start()
        data_list = []
        errors = 0
        self.pl_model.flow.utils_config['sample_steps'] = steps
        self.pl_model.flow.utils_config['eta'] = eta
        self.pl_model.flow.register_schedule()
        self.pl_model.flow.to(device)
        
        # SAMPLE
        for idx, data in tqdm(
            enumerate(self.dataloader), total=len(self.dataloader), desc='Inferencing BBDM'
        ):
            try:
                data = data.to(device)
                batch = self.sample(data)
                for i in range(data.cell.size(0)):
                    data_item = batch[i]
                    if hasattr(data, 'id'):
                        data_item.id = data[i].id
                        data_item.comp = data[i].comp
                    data_list.append(data_item.cpu())
            except Exception as e:
                print("Error in inference: ", idx)
                print(e)
                continue
        timelens = timer.stop()
        
        # RESULTS
        print("========================Inference Results============================")
        print(f"Inference time: {timelens} for {steps} steps BBDM with eta {eta}")
        print(f"END with totol data length: {len(data_list)}, error: {errors}")
        print("==============================END====================================")
        
        return data_list
    
    # sampling by different sample steps
    def sample(self, batch):
        pos_relaxed = self.pl_model.validate_sample(batch)[0]
        batch['pos_generated'] = pos_relaxed
        return batch
    
    def save_generated_data(self, dir_path: str, stage: str, data_list: list) -> None:
        for idx, data in tqdm(
            enumerate(data_list), total=len(data_list), desc='Processing generated data'
        ):
            data.pos_relaxed = data.pos_generated
        os.makedirs(dir_path, exist_ok=True)
        pickle.dump(data_list, open(f'{dir_path}/{stage}.pkl', 'wb'))
        # RESULTS
        print("=========================Save Results================================")
        print("Saved generated data to: ", dir_path + "/" +stage + ".pkl")
        print("Total data length: ", len(data_list))
        print("==============================END====================================")
        
    # label data for self-verification model according to D-MAE of the generated data
    def label_dmae(self, data_list: List[Any], dmae_threshold: float) -> List:
        num_positive = 0
        num_negative = 0
        print("Threshold of D-MAE: ", dmae_threshold)
        for data in tqdm(data_list, desc='Labeling data'):
            pos_r_frac = torch.matmul(data.pos_relaxed, torch.inverse(data.cell).squeeze())
            pos_g_frac = torch.matmul(data.pos_generated, torch.inverse(data.cell).squeeze())
            # calculate D-MAE
            dmae = calc_dmae_mic(
                pos_r_frac, pos_g_frac, data.cell.squeeze()
            ).item()
            data.dmae = dmae
            if dmae < dmae_threshold:
                data.label = 1
                num_positive += 1
            else:
                data.label = 0
                num_negative += 1
        return data_list, num_positive, num_negative
    
    def workflow_forward(self, data_list: List[Any], bc_model: LightningModule, device: torch.device) -> None:
        pred_list = []
        label_list = []
        for data in tqdm(data_list, desc='Computing metrics'):
            data = data.to(self.device)
            pred = bc_model.compute_label(data).squeeze().cpu().numpy().tolist()
            pred_list.append(pred)
            label_list.append(data.label)
        # RESULTS
        print("=========================Test Results===============================")
        print("Total data length: ", len(data_list))
        print("Prediction list: ", pred_list)
        print("Label list: ", label_list)
        print("==============================END====================================")
        

    
        

        
