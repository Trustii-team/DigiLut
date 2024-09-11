import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import ViTForMaskedImageModeling, Swinv2ForMaskedImageModeling, SwinForMaskedImageModeling, get_cosine_schedule_with_warmup
from torchmetrics.classification import BinaryAccuracy

def init_weights(module):
    if isinstance(module, (nn.Linear)):
        module.weight.data.normal_(mean=0.0, std=0.02)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, first_token_tensor):
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.fc(pooled_output)
        return pooled_output.squeeze(-1)


class PretrainClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        print(config)
        #self.logger = config.logger
        self.save_hyperparameters()

        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.model = None
        self.model_type = config['model_type'] 
        if self.model_type == 'vit':

            self.model = ViTForMaskedImageModeling.from_pretrained(config['vit'], local_files_only=True,
                                                              num_labels=1, ignore_mismatched_sizes=True)
        if self.model_type == 'swin':
            self.model = SwinForMaskedImageModeling.from_pretrained(config['swin'], local_files_only=True,
                                                                    num_labels=1, ignore_mismatched_sizes=True)
            
        if self.model_type == 'swinv2':
            self.model = Swinv2ForMaskedImageModeling.from_pretrained(config['swinv2'], local_files_only=True,
                                                                    num_labels=1, ignore_mismatched_sizes=True)
        
        
        self.lr = config['learning_rate']
        self.accuracy = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        torch.set_float32_matmul_precision("medium")
        
        
        if config['cls_loss_lambda'] > 0:
            self.cls = True
            self.pooler = Pooler(config["hidden_size"])
            self.pooler.apply(init_weights)
            self.cls_loss_lambda = config["cls_loss_lambda"]



    def forward(self, pixel_values, bool_masked_pos):
        outputs = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, output_hidden_states=True)
        loss,  hidden_states = outputs.loss, outputs.hidden_states[-1]
        #print('loss', loss.shape)
        #print('hidden_states', hidden_states.shape)
        if self.cls:
            cls_output = torch.sum(hidden_states, 1, keepdim=True)
            #print('cls_output', cls_output.shape)
            cls_logits = self.pooler(cls_output)
            #print('cls_logits', cls_logits.shape)
            return loss, cls_logits
        return loss, None
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        bool_masked_pos = batch['bool_masked_pos']

        labels = batch['labels']

        loss, cls_logits = self(pixel_values, bool_masked_pos)
        #print('labels', labels.shape)

        if self.cls:
            cls_loss = self.loss(cls_logits, labels.float())
                
            cls_logits = self.sigmoid(cls_logits)
            accuracy = self.accuracy(cls_logits, labels)

            return loss, cls_loss, accuracy # logits, labels
        return loss, None, accuracy # logits, labels
    
    def training_step(self, batch, batch_idx):
        
        #loss, accuracy, _, _ = self.common_step(batch, batch_idx) 
        rec_loss, cls_loss, accuracy = self.common_step(batch, batch_idx) 
        loss = rec_loss + self.cls_loss_lambda*cls_loss
        self.log("train_reconstruction_loss", rec_loss)
        self.log("train_cls_loss", cls_loss)
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        
        return {"loss": loss} #, "accuracy": accuracy}
    
    def validation_step(self, batch, batch_idx):
        #loss, accuracy , logits, labels= self.common_step(batch, batch_idx)   
        loss, cls_loss, accuracy = self.common_step(batch, batch_idx)   
        rec_loss, cls_loss, accuracy = self.common_step(batch, batch_idx) 
        loss = rec_loss + self.cls_loss_lambda*cls_loss
        self.log("val_reconstruction_loss", rec_loss, on_epoch=True)
        self.log("val_cls_loss", cls_loss, on_epoch=True)
        self.log("val_accuracy", accuracy, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True)
        #self.log("val_accuracy", accuracy, on_epoch=True)

        return {"val_loss": loss} # "val_accuracy": accuracy,"outputs": logits, "labels": labels}


    def configure_optimizers(self):
        ## TODO maybe change ?

        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine

        
        optimizer = torch.optim.AdamW([{'params': self.parameters(), 'lr': self.lr}, ], eps=1e-8, betas=(0.9, 0.98))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10000,
            num_training_steps=100000,
        )
        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
            )
    
