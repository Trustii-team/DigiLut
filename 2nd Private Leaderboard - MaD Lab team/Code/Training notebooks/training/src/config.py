from sacred import Experiment

ex = Experiment("DigiLut")

@ex.config
def config():
    num_gpus = 1
    seed = 0
    log_dir = "results"
    per_gpu_batchsize =  320

    num_workers = 16
    precision = "16-mixed"
    
    # Image setting
    train_transform = ["train_transform2", "norm", 'data/pKV3BMNUpm_a_15800_26000_224.png']
    val_transform = ["val_transform", "norm", 'data/pKV3BMNUpm_a_15800_26000_224.png']
    image_size = 224
    focal_loss_alpha = 0.05
    focal_loss_gamma = 2.0

    # Model Setting
    model_type = "dino"
    vit = "google/vit-large-patch16-224"
    swin = "microsoft/swin-tiny-patch4-window7-224"
    swinv2 = "microsoft/swinv2-base-patch4-window12-192-22k"
    model_patch_size = 4
    dino = "facebook/dinov2-small-imagenet1k-1-layer"
    cls_loss_lambda=1
    hidden_size=768
    # Optimizer Setting
    learning_rate = 5e-5
    loss_exponent=0.33
    max_epoch=20

    # PL Trainer Setting
    pretrain = False
    
    ckpt_path = ''

    hard=False
    label_threshold=0.0

    finetune=False
    pretrain=False
    images_dir = {'train': "data/images/level_3", 'val': 'data/images/level_3'}
    annotation_dir = {'train': "data/annotations/level_3", 'val': "data/annotations/level_3"} 

    exp_name="test"
 
@ex.named_config
def train_hard_binary_classification():   
    hard = True
    exp_name = f"digilut_train_hard"
 
@ex.named_config
def train_soft_binary_classification():       

    hard=False
    exp_name = f"digilut_train_soft"
       
@ex.named_config
def train_hard_binary_classification_threshold():  
    label_threshold=0.2 
    hard = True
    exp_name = f"digilut_train_hard_threshold"
 
@ex.named_config
def train_soft_binary_classification_threshold():  
    label_threshold=0.2     
    hard=False
    exp_name = f"digilut_train_soft_threshold"

@ex.named_config
def pretrain():
    model_type= 'swin'
    images_dir= {'train': "data_pretrain/images", 'val': 'data_pretrain/images'}
    annotation_dir = {'train': "data_pretrain/annotations/train.csv", 'val': "data_pretrain/annotations"} 
    cls_loss_lambda =  1
    per_gpu_batchsize =  128
    pretrain=True
    ckpt_path = ''
    exp_name = f"digilut_pretrain"

@ex.named_config
def finetune_hard_binary_classification():   
    model_type = "swin"
    hard = True    
    ckpt_path = ''
    finetune=True
    
    exp_name = f"digilut_finetune_hard"
 
@ex.named_config
def finetune_soft_binary_classification(): 
    model_type = "swin"      
    hard = False
    ckpt_path = ''
    finetune=True
    
    exp_name = f"digilut_finetune_soft"
  