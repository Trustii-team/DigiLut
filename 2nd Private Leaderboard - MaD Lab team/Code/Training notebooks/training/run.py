import os
import copy
import pytorch_lightning as pl

from src.config import ex
from src.models import PretrainClassifier, BinaryClassifier
from src.data.patch_datamodule import DataModule

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    datamodule = DataModule(_config, dist=True)

    if _config["pretrain"]:
        model = PretrainClassifier(_config)
    elif _config["finetune"] and _config["ckpt_path"]:
        print(f'\n FINETUNING MODEL FROM CHECKPOINT {_config["ckpt_path"]}\n')
        model = BinaryClassifier.load_from_checkpoint(_config["ckpt_path"])
        model.hard_binary = _config["hard"]
        model.threshold = _config['label_threshold']
        model.loss_exponent = _config['loss_exponent']
    else:
        model = BinaryClassifier(_config)
    exp_name = f"{_config['exp_name']}_{_config['model_type']}_lr_{_config['learning_rate']}_focal_{_config['focal_loss_alpha']}_{_config['focal_loss_gamma']}_threshold_{_config['label_threshold']}_expo{_config['loss_exponent']}"

    os.makedirs(_config["log_dir"], exist_ok=True)


    logger = pl.loggers.TensorBoardLogger(
        save_dir=_config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}',
    )

    if not _config["pretrain"]:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val_f1",
            mode="max",
            save_last=True)
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=10,
            strict=False,
            verbose=False,
            mode='max'
        )
        
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
            save_last=True)
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            strict=False,
            verbose=False,
            mode='min'
        )
        

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_callback, early_stop_callback]

    if not _config["pretrain"]:
        if _config['label_threshold'] < 0.1:
            limit_val_batches = 0.2
        else:
            limit_val_batches = 1.0
        trainer = pl.Trainer(
            devices=_config["num_gpus"],
            accelerator="auto",
            max_epochs=_config["max_epoch"],
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            sync_batchnorm=True,
            limit_val_batches=limit_val_batches,
            val_check_interval=0.5,
            accumulate_grad_batches=8,
        )
    else:
        trainer = pl.Trainer(
            devices=_config["num_gpus"],
            accelerator="auto",
            max_epochs=_config["max_epoch"],
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=10,
            limit_train_batches=0.13,
            limit_val_batches=0.1,
        )
    


    if _config["pretrain"] and len(_config['ckpt_path'])> 0:
        print(f"RESUMING PRETRAINING FROM PREVIOUS CHECKPOINT {_config['ckpt_path']}")
        trainer.fit(model,  datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=_config['ckpt_path'])
    else:
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

