import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .patch_dataset import DigiLutDataset

       

class DataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super().__init__()

        self.images_dir = _config["images_dir"]
        self.ann_dir = _config["annotation_dir"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]

        self.image_size = _config["image_size"]

        self.train_transform = _config["train_transform"]
        self.val_transform = _config["val_transform"]
        self.test_transform = _config["val_transform"]

        self.dist = dist
        
        self.num_patches =  (self.image_size // _config["model_patch_size"] ) ** 2
        
        if _config["pretrain"]:
            self.collate = self.collate_pretrain
        else:
            self.collate = self.collate

    def collate(self, batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])

        labels = torch.tensor([example["label"] for example in batch], dtype=torch.float32)
        labels = labels.unsqueeze(1)

        return {"pixel_values": pixel_values, "labels": labels} 

    def collate_pretrain(self, batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])

        labels = torch.tensor([example["label"] for example in batch],  dtype=torch.float32)
        labels = labels.unsqueeze(1)

        
        bool_masked_pos = torch.randint(low=0, high=2, size=(len(batch), self.num_patches)).bool()

        return {"pixel_values": pixel_values, "bool_masked_pos":bool_masked_pos, "labels": labels}




    @property
    def dataset_name(self):
        return "DigiLut"
    

    def train_dataloader(self):
        loader = DataLoader(
            DigiLutDataset(
            self.images_dir,
            self.ann_dir,
            self.train_transform,
            split="train",
            image_size=self.image_size,
        ),
            shuffle = True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(
            DigiLutDataset(
            self.images_dir,
            self.ann_dir,
            self.val_transform,
            split="val",
            image_size=self.image_size,
            )
         ,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(
            DigiLutDataset(
            self.images_dir,
            self.ann_dir,
            self.test_transform,
            split="val",
            image_size=self.image_size,)
         ,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader

