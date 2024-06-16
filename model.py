import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
import torchmetrics
import gc
from torchvision.models import resnet101, ResNet101_Weights
class Model(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0001,
        batch_size: int = 256,
        seed_value: int = 1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        if seed_value != None:
          pl.seed_everything(seed_value)

        self.pretrained = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2) 
        self.pretrained.fc = nn.Linear(2048, 7)
        self.loss_f = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.softmax = nn.Softmax(dim=1)
        
        
    def calc_metric(self, logit, target):
        loss = self.loss_f(logit, target)
        f1 = torchmetrics.functional.f1_score(logit, target, 
                                              task="multiclass", num_classes=7, 
                                              average="macro")
        accuracy = torchmetrics.functional.accuracy(logit, target, 
                                                    task="multiclass", num_classes=7, 
                                                    average="macro")
        return loss, f1, accuracy
   
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        data, target = batch
        logits = self.pretrained(data)
        loss, f1, accuracy = self.calc_metric(logits, target)
        self.log('F1/Train', f1, prog_bar=True, on_epoch=True)
        self.log('Accuracy/Train', accuracy, prog_bar=True, on_epoch=True)
        self.log('Loss/Train', loss, prog_bar=True, on_epoch=True)
        loss_1.backward()
        optimizer.step()
        optimizer.zero_grad()   
        
    def validation_step(self, batch, batch_idx):
        data, target = batch
        logits = self.pretrained(data)
        loss, f1, accuracy = self.calc_metric(logits, target)
        self.log('F1/Validation', f1, prog_bar=True, on_epoch=True)
        self.log('Accuracy/Validation', accuracy, prog_bar=True, on_epoch=True)
        self.log('Loss/Validation', loss, prog_bar=True, on_epoch=True)
        
        
    def on_validation_epoch_end(self):
        gc.collect()
        self.trainer.save_checkpoint(filepath="/kaggle/working/best_checkpoint.ckpt")
        gc.collect()
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(self.parameters()),
                         lr=self.hparams.learning_rate
                         )
        return optimizer
