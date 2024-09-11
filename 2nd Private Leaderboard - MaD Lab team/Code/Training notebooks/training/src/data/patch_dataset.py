import torch
import os
import pandas as pd
from PIL import Image
from transformers import ViTImageProcessor
import torchvision.transforms.functional as F
from scipy.stats import truncnorm
import numpy as np
import zipfile
from stainlib.augmentation.augmenter import HedLighterColorAugmenter, StainAugmentor
from stainlib.utils.stain_utils import LuminosityStandardizer
from stainlib.normalization.normalizer import ReinhardStainNormalizer
import stainlib
import io

from .patch_transform import choose_transform, normalize_staining


class DigiLutDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir: str,
        ann_dir:str,
        transform_key: list,
        split: str,
        image_size: int,
    ):
        
        self.images_dir = images_dir[split]
        self.ann_dir = ann_dir[split]
        self.transform_key = transform_key
        self.split = split
        self.image_size = image_size
        
        self.zip_file_paths = []
        for filename in os.listdir(self.images_dir):
            if filename.endswith('.zip'):
                self.zip_file_paths.append(os.path.join(self.images_dir, filename))
                
        

        if split == "train":
            if self.ann_dir.split('.')[-1]=='csv':
                ann_label_file = self.ann_dir
            else:
                ann_label_file = os.path.join(self.ann_dir, 'soft_train.csv')
            self.ann_df = pd.read_csv(ann_label_file)

                

        elif split == "val":
            ann_label_file = os.path.join(self.ann_dir, 'corrected_val.csv')
            self.ann_df = pd.read_csv(ann_label_file)
        elif split == "test":
            ann_label_file = os.path.join(self.ann_dir, 'test.csv')
            self.ann_df = pd.read_csv(ann_label_file)
    

        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224", local_files_only=True)
        image_mean = self.image_processor.image_mean
        image_std = self.image_processor.image_std
        size = self.image_processor.size["height"]
        self.transform = choose_transform(transform_key[0], size, image_mean, image_std )
        
        self.hed_transform = HedLighterColorAugmenter()
        self.normalizer = ReinhardStainNormalizer()
        if  isinstance(self.transform_key[2], str):
            target =  Image.open(self.transform_key[2])
            target = F.adjust_contrast(target, contrast_factor=1.5)
            target = np.array(target)
            target = stainlib.utils.stain_utils.LuminosityStandardizer.standardize(target)
            self.normalizer.fit(target)




    def __len__(self):
        return len(self.ann_df)
    
    def read_image_patch_file(self, filename, x, y):
        image_name = f"{filename.split('.')[0]}_{x}_{y}_{self.image_size}.png"
        for zip_file_path in self.zip_file_paths:
            with zipfile.ZipFile(zip_file_path, 'r') as zip:
                # Check if the file exists in the current ZIP file
                if image_name in zip.namelist():
                    # Open and read the file
                    with zip.open(image_name) as file:
                        content = file.read()
                        return Image.open(io.BytesIO(content))
        print(filename, x, y)
        return "Not a file"

    
    def get_image_patch(self, filename, x, y):
        patch = self.read_image_patch_file(filename, x, y)

        patch = patch.convert('RGB')
        rand = np.random.rand()

        patch = F.adjust_contrast(patch, contrast_factor=1.5)

        if self.split == 'train' and rand>0.2:
            patch = np.array(patch)
            self.hed_transform.randomize()
            patch = self.hed_transform.transform(patch)

            patch = Image.fromarray(patch.astype(np.uint8))
            
        patch = self.transform(patch)

        patch = np.array(patch)
        patch = stainlib.utils.stain_utils.LuminosityStandardizer.standardize(patch)
        patch = self.normalizer.transform(patch)

        return patch
    
    def get_patch_label_and_coords(self, index):
        x, y = (self.ann_df.iloc[index]["x1"], self.ann_df.iloc[index]["y1"])
        label, filename = (self.ann_df.iloc[index]["label"], self.ann_df.iloc[index]["filename"])
        return label, filename, x, y
    
    def __getitem__(self, index):
        label, filename, x, y = self.get_patch_label_and_coords(index)
        patch = self.get_image_patch(filename, x, y)
        tensor = self.image_processor(patch, return_tensors="pt", do_rescale=True, do_normalize=False)

        return {"pixel_values": tensor["pixel_values"].squeeze(0), "label":label} 
        




