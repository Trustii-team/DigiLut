import os
import copy
import torch
import pytorch_lightning as pl

from src.config import ex
from src.models import BinaryClassifier
from src.data.patch_datamodule import DataModule
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    
    exp_name = f"digilut_predict_threshold_{_config['label_threshold']}"
    model_checkpoint_path = _config["ckpt_path"]    
    
    print('\n LOADING MODEL FROM CHECKPOINT \n')
   
    model = BinaryClassifier.load_from_checkpoint(model_checkpoint_path)
    model.eval()
    datamodule = DataModule(_config)


    trainer = pl.Trainer(
        devices=_config["num_gpus"],
        accelerator="auto",
        deterministic=True,
        log_every_n_steps=10,
        sync_batchnorm=True,
        limit_predict_batches=0.5
    )
    predictions = trainer.predict(model, datamodule.val_dataloader())
    
    labels= []
    logits = []
    f1 = []
    for batch in predictions:
        labels.extend(batch["labels"]) 
        logits.extend(batch["logits"]) 
        f1.append(batch["f1"].item())

    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    preds = (logits >= 0.5).long()
    

    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels.numpy(), preds.numpy())
    print(f1)
    print("precision", np.mean(np.array(f1)))
    print("Confusion Matrix:\n", conf_matrix)
    tn, fp, fn, tp = conf_matrix.ravel()
    print("True Positive", tp)
    print("True Negative", tn)
    print("False Positive", fp)
    print("False Negative", fn)
    


