import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import ViTForImageClassification, SwinForImageClassification, Swinv2ForImageClassification, get_cosine_schedule_with_warmup, Dinov2ForImageClassification
from torchmetrics.classification import BinaryAccuracy, BinaryFBetaScore, BinaryF1Score, BinaryPrecisionAtFixedRecall, MultilabelF1Score, MulticlassF1Score
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import sigmoid_focal_loss

## Soft focal loss
def soft_sigmoid_focal_loss(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    soft_label: torch.Tensor,
    ce_loss_fn,
    alpha: float = 0.25,
    gamma: float = 2,
    expo: float = 0.33,
    reduction: str = "none",

) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        ce_loss_fn : cross entropy loss function
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    quality = soft_label.detach().clone()
    quality = quality + 1e-6
    quality = torch.pow(quality, expo).float()
    p = torch.sigmoid(inputs)
    opposite_p = 1.0 - p
    opposite_quality = 1.0 - quality
    both_inputs = torch.stack((p, opposite_p), dim=1).squeeze(2)
    both_quality = torch.stack((quality, opposite_quality), dim=1).squeeze(2)
    both_inputs = torch.log(both_inputs)
    ce_loss = ce_loss_fn(both_inputs.float(), both_quality.float())
    #ce_loss = ce_loss_fn(inputs.float(), soft_label.float())
    p_t = p * labels + (1 - p) * (1 - labels)
    quality_with_zeros=quality.detach().clone()
    mask = quality_with_zeros < 0.5
    quality_with_zeros[mask] = 1.0 - quality_with_zeros[mask]

    loss = ce_loss * (torch.abs(quality_with_zeros - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * soft_label + (1 - alpha) * (1 - soft_label)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

class BinaryClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        print(config)
        self.save_hyperparameters()
        self.config = config

        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration

        self.model_type = config['model_type'] 
        if self.model_type == 'vit':

            self.model = ViTForImageClassification.from_pretrained(config['vit'], local_files_only=True,
                                                            num_labels=1, ignore_mismatched_sizes=True)
        elif self.model_type == 'swin':
            self.model = SwinForImageClassification.from_pretrained(config['swin'],   local_files_only=True,
                                                                    num_labels=1, ignore_mismatched_sizes=True)
            
        elif self.model_type == 'swinv2':
            self.model = Swinv2ForImageClassification.from_pretrained(config['swinv2'],  local_files_only=True,
                                                                    num_labels=1, ignore_mismatched_sizes=True)
            
        elif self.model_type == 'dino':
            self.model = Dinov2ForImageClassification.from_pretrained(config['dino'],  # local_files_only=True,
                                                                    num_labels=1, ignore_mismatched_sizes=True)
        
        elif self.model_type == 'resnet':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 1)

        if 'focal_loss_alpha' not in config.keys():
            self.focal_loss_alpha = 0.25
        else:
            self.focal_loss_alpha = config['focal_loss_alpha']
            
        if 'focal_loss_gamma' not in config.keys():
            self.focal_loss_gamma = 2.0
        else:
            self.focal_loss_gamma = config['focal_loss_gamma']
        
        self.lr = config['learning_rate']
        self.f_1_binary = BinaryF1Score()
        self.sigmoid = nn.Sigmoid()
        #torch.set_float32_matmul_precision("medium")
        #self.ce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss_fn = nn.KLDivLoss(size_average=False, reduction='none')
        if 'hard' in config.keys():
            self.hard_binary = config['hard']
            self.threshold = config['label_threshold']
            self.loss_exponent = config['loss_exponent']
        else:
            self.hard_binary = False
            self.threshold = 0.0
            self.loss_exponent = 0.33
        
        
       
    @classmethod 
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        hparams = checkpoint['hyper_parameters']
        print("INIT")
        model = cls(**hparams, **kwargs)
        print("LOADING")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("END LOADING")
        
        return model



    def forward(self, pixel_values):
        if self.model_type == 'resnet':
            outputs = self.model(pixel_values)
        elif self.model_type == 'dino':
            outputs = self.model(pixel_values=pixel_values).logits
        else:
            outputs = self.model(pixel_values=pixel_values).logits
        return outputs
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        
        if self.threshold != 0: 
            hard_labels = torch.where(labels < self.threshold, torch.tensor(0.0, dtype=torch.float32), labels)
        
        hard_labels = torch.where(hard_labels > 0.0, torch.tensor(1.0, dtype=torch.float32), hard_labels)
        
        if self.hard_binary:
            loss = sigmoid_focal_loss(logits, hard_labels.float(), alpha =self.focal_loss_alpha, gamma=self.focal_loss_gamma)
        else:
            loss = soft_sigmoid_focal_loss(logits, hard_labels.float(),labels.float(), self.ce_loss_fn, alpha = self.focal_loss_alpha, gamma = self.focal_loss_gamma, expo = self.loss_exponent)

        logits = self.sigmoid(logits)
        metric= self.f_1_binary(logits, hard_labels)
        return loss, metric, logits, labels
    
    def training_step(self, batch, batch_idx):
        loss, metric, _, _ = self.common_step(batch, batch_idx) 
        self.log("train_loss", loss.mean())
        self.log("train_f1", metric)
        return {"loss": loss.mean(), "train_f1": metric}
    
    def validation_step(self, batch, batch_idx):
        loss, metric , _, _= self.common_step(batch, batch_idx)     
        self.log("val_loss", loss.mean(), on_epoch=True)
        self.log("val_f1", metric, on_epoch=True)

        return {"val_loss": loss.mean(),"val_f1": metric}




    def test_step(self, batch, batch_idx):
        loss, f1 , logits, labels = self.common_step(batch, batch_idx)     

        return {"test_loss": loss.mean(), "test_f1": f1}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        hidden_states = torch.sum(hidden_states, 1, keepdim=True)
        logits = self.sigmoid(outputs.logits)
        
        
        if self.threshold != 0: 
            labels = torch.where(labels < self.threshold, torch.tensor(0.0, dtype=torch.float32), labels)
        
        hard_labels = torch.where(labels > 0.0, torch.tensor(1.0, dtype=torch.float32), labels)
     
        metric= self.f_1_binary(logits, hard_labels)
        #torch.cuda.empty_cache()
        

        #return {"logits": logits, "labels": hard_labels, "filenames": batch["filenames"], "f1": metric,
        #        "x_coords": batch["x_coords"], "y_coords": batch["y_coords"], "out_feats": hidden_states}
        return {"logits": logits, "labels": hard_labels, "f1": metric}
                
    def configure_optimizers(self):
        ## TODO maybe change ?

        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        basic_parameters = []
        classifier_parameters = []
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_parameters.append(param)
            else:
                basic_parameters.append(param)
        
        #torch.optim.AdamW([{'params': self.parameters(), 'lr': self.lr}, ], eps=1e-8, betas=(0.9, 0.98))
        #torch.optim.AdamW([{'params': basic_parameters, 'lr': self.lr*0.1}, {'params': classifier_parameters, 'lr': self.lr}], eps=1e-8, betas=(0.9, 0.98))
        
        #torch.optim.AdamW([{'params': basic_parameters, 'lr': self.lr*0.1}, {'params': classifier_parameters, 'lr': self.lr}])
    
        optimizer = torch.optim.AdamW([{'params': self.parameters(), 'lr': self.lr}, ], eps=1e-8, betas=(0.9, 0.98))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
        )
        sched = {"scheduler": scheduler, "interval": "step"}

        return (
            [optimizer],
            [sched],
            )