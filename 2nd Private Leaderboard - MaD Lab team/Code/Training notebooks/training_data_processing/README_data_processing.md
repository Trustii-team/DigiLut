## DigiLut Challenge - code for training

### Structure
```
├── patches_csv             # Folder: Files of the generated patch coordinates and labels will be saved here.
├── clean_train.csv         # train.csv updated to remove wrong annotations (see previous Trustii email)
├── new_val_ids.txt         # Images to validate the models (part of the training dataset provided by Trustii)
├── generating_level_2_patches_for_training.ipynb  # Notebook used to generate level 2 patches for training 
├── generating_level_3_patches_for_training.ipynb  # Notebook used to generate level 3 patches for training


```


#### Getting Started
Adjust the paths for images and csvs in the notebooks.

The 'clean_train.csv' is built from the train.csv exculding wrong annotations.

The code to generate the image patches for training the level 2 and level 3 (with respect to openslide levels) for training is in the respective jupyter notebooks for level 2 and 3.

These notebooks were run on the Trustii Jupyter hub, and do not need any additional requirement for this platform. The images will be generated as zip file to be uploaded on the high performance computing cluster. Together with the annotation files for train and val.


#### Train/val split

Split train/val for training:  The splits are created in the jupyter notebooks. They are a subset of the provided train set, not part of the validation images provided by the challenge. A csv is created for the train and val split and is saved in the patches_csv folder and contains labels and annotations for all generated png image crops.