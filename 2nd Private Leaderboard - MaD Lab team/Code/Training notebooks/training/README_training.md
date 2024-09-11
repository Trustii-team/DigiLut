## DigiLut Challenge - code for training

### Structure
```
├── data                    # Folder containing data files (see more information below)
├── src                     # Source code for training
│   ├── data                # Files for dataset and dataloader creation
│   ├── models              # Files for models: 
│   │   ├── binary_model.py                     # The binary classifier used for training both level 2 and level 3
│   │   ├── pretrain_model.py                   # The pretrain model tested for pretraining but not used in the submissions
│   ├── config.py           # The configuration file used for training
├── run.py                  # File to run training 
├── predict.py              # File to run inference on whole val dataset 
├── requirements.txt        # File listing all Python package dependencies 
└── install.txt             # Instructions for setting up the environment and installing necessary packages
```

### Train models for level 2 and level 3 patches for positive/negative classification prediction

#### Getting Started
Run each line found in `install.txt` to create your own environment and download necessary packages.


##### Update the config
These models were trained on a A100 GPU.
Update the src/config.py file with your paths, chosen batch sizes and other parameters to match your GPUs and train the model. Please match the data paths with your chosen data paths for the annotation files and zip files computed previously.

The data folder is expected to have this structure :

```
├── data                     
│   ├── annotations    
│   │   ├── level_2
│   │   │   ├── more_selective_patches_7_level_2_size_224_step_200/train.csv
│   │   │   ├── more_selective_patches_7_level_2_size_224_step_200/val.csv
│   │   ├── level_3
│   │   │   ├── selective_patches_level_3_size_224_step_200/train.csv
│   │   │   ├── selective_patches_level_3_size_224_step_200/val.csv
│   ├── images              
│   │   ├── level_2     
│   │   │   ├── more_selective_patches_7_level_2_size_224_step_200.zip
│   │   ├── level_3     
│   │   │   ├── selective_patches_level_3_size_224_step_200.zip


```

#### Train the models

Use run.py to train the model and predict.py for inference.



To train for patches level 2 and level 3, the following command can be used:
```
python3 run.py with train_soft_binary_classification learning_rate=1e-5 per_gpu_batchsize=64 num_workers=8 label_threshold=0.01 loss_exponent=0.15
```

#### Train/val split

Split train/val for training:  The splits are created in the jupyter notebooks. They are a subset of the provided train set, not part of the validation images provided by the challenge