Digilut: CVN team solution
==============================

Solution description
------------

In this challenge, we propose to use a patch based approach to predict the ROI of a given image.
We begin by extracting $512 \times 512$ patches from slides with no lesions, labelled as negatives, and $512 \times 512$ patches from the annotated bounding boxes for the slides with lesions, labelled as positives.

We then train a model to predict the correct label for each patch. We use a ResNet18 model pre-trained on histopathology images, and fine-tune it on the patches.

Finally, we extracted patches from the test slides and predict the label for each patch. We then aggregate the predictions by merging recursively positive patches that are close to each other. We then compute the bounding box of the merged patches and output the biggest bounding box as the final prediction.

Project structure
------------

    ├── README.md
    │
    ├── data
    │   ├── raw
    |   |   ├── Bounding Box IDs  
    |   |   ├── images
    |   |   ├── presence_of_lesion.csv
    |   |   ├── train.csv
    |   |   └── validation.csv  
    │   ├── processed
    |   |   ├── positive_patches <- patches extracted from the bounding boxes with presence_of_lesion=1
    |   |   ├── train_negative <-  patches extracted from the training slides with no lesions
    |   |   |   ├── contoured_images <- snapshots of the slides with contours of the tissue regions
    |   |   |   ├── patches <- patches extracted from the slides saved as .h5 files (not used here)
    |   |   |   ├── patches_as_jpg_1000 <- 1000 randomly chosen patches extracted from the slides saved as .jpg files under slide name
    |   |   |   └── stitch_images <- snapchot of the slides with the patches extracted
    |   |   ├── val <- patches extracted from the validation slides
    |   |   |   ├── contoured_images <- snapshots of the slides with contours of the tissue regions
    |   |   |   ├── patches <- patches extracted from the slides saved as .h5 files (not used here)
    |   |   |   ├── patches_as_jpg_full <- all patches extracted from the slides saved as .jpg files under slide name
    |   |   |   └── stitch_images <- snapchot of the slides with the patches extracted
    |   |   ├── dataframe_testing_selected.csv <- dataframe contains the final test data with the created 512 \times 512 images and their paths for modeling
    |   |   ├── dataframe_training_selected.csv <- dataframe contains the final train data with the created 512 \times 512 images and their paths for modeling 
    |   |   └── dataframe_validation_selected.csv <- dataframe contains all the validation data with the created 512 \times 512 images and their paths for modeling 
    │   │
    │   └── predictions
    |
    │
    ├── models <- Trained models
    │
    │
    ├── notebooks
    │   ├── preprocess_data.ipynb <- Notebook to turn raw data into features for modeling
    |   |
    │   ├── train.ipynb <- Notebook to split provided train.csv to training and testing sets.
    |   |
    │   ├── train.ipynb <- Notebook to train model(s)
    │   │
    │   └── predict.ipynb <- Notebook to make predictions using trained models
    │
    │
    ├── utils
    │   ├── LungSet.py <- Dataset module for training on patches
    │   |
    │   └── utils.py <- Utils function for notebooks

Installation
------------

To install the dependencies, you can run the following command:

```bash
python -m venv .venv
pip install -r requirements.txt
```

In case some problems are encountered with opencv, you may downgrade the version to:

```bash
pip install opencv-python==4.8.0.74
```

Preprocessing
------------

To extract the patches from the slides, and create materials for training the classification model, we provide the `preprocess_data`  jupyter notebook. You can run it with the following command:

```bash
jupyter execute notebooks/preprocess_data.ipynb
```

Training
------------

For the model, we used a pretrained ResNet18 model on histopathology images - [link to the model](https://github.com/ozanciga/self-supervised-histopathology/releases/tag/nativetenpercent). We fine-tuned the model on the extracted patches. The training process is detailed in the `train` notebook. You can run it with the following command:

```bash
jupyter execute notebooks/train.ipynb
```

Predictions
------------

To generate final predictions from model predictions, we merge the positive patches that are close to each other. The process is detailed in the `predict` notebook. You can run it with the following command:

```bash
jupyter execute notebooks/predict.ipynb
```
