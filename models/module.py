import lightning as L
import torch
from torch import nn
from utils.utils import AvgMeter, get_lr
from config import CFG
from transformers import get_linear_schedule_with_warmup

# TODO : check "engine.py" pour ré-implémenter
class Pix2Seq(L.LightningModule):
    def __init__(self, cfg, encoder, decoder):
        super().__init__()
        self.encoder = encoder # Encoder composed of a backbone and a Transformer encoder
        self.decoder = decoder # Decoder composed of an autoregressive Transformer decoder

        self.cfg = cfg

        self.loss_meter_train = AvgMeter("train_loss")
        self.loss_meter_val = AvgMeter("val_loss")

        self.loss_criterion = nn.CrossEntropyLoss(ignore_index=self.cfg.pad_idx)

        self.automatic_optimization = False
    
    def forward(self, image, tgt):
        encoder_out = self.encoder(image)
        preds = self.decoder(encoder_out, tgt)
        return preds

    def training_step(self, batch, batch_idx):
        image, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        optimizer = self.optimizers()

        preds = self(image, tgt_input)
        loss = self.criterion(preds.reshape(-1, preds.shape[-1]), tgt_expected.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        self.lr_scheduler.step()

        self.loss_meter_train.update(loss.item(), image.size(0))
        lr = get_lr(optimizer)
        
        self.log('train_loss', self.loss_meter_train.avg, sync_dist=True)
        self.log('lr', lr, sync_dist=True)
        return self.loss_meter_train.avg
    
    def validation_step(self, batch, batch_idx):
        #model, valid_loader, criterion):
        image, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        preds = self(image, tgt_input)
        loss = self.criterion(preds.reshape(-1, preds.shape[-1]), tgt_expected.reshape(-1))


        self.loss_meter_val.update(loss.item(), image.size(0))
        self.log('val_loss', self.loss_meter_val.avg, sync_dist=True)
        return self.loss_meter_val.avg
    
    def configure_optimizers(self):
        # TODO : implement the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_training_steps=self.cfg.num_training_steps,
                                                    num_warmup_steps=self.cfg.num_warmup_steps)
        return [optimizer], [self.lr_scheduler]
    
    def criterion(self, preds, tgt):
        return self.loss_criterion(preds, tgt)