import lightning as L
import torch
from torch import nn
from utils.utils import AvgMeter, get_lr
from transformers import get_linear_schedule_with_warmup, top_k_top_p_filtering
import cv2
import matplotlib.pyplot as plt

import torch

GT_COLOR = (0, 255, 0) # Green
PRED_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

# TODO : check "engine.py" pour ré-implémenter
class Pix2Seq(L.LightningModule):
    def __init__(self, cfg, encoder, decoder, tokenizer):
        super().__init__()
        self.encoder = encoder # Encoder composed of a backbone and a Transformer encoder
        self.decoder = decoder # Decoder composed of an autoregressive Transformer decoder

        self.tokenizer = tokenizer

        self.cfg = cfg

        self.loss_meter_train = AvgMeter("train_loss")
        self.loss_meter_val = AvgMeter("val_loss")

        self.loss_criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.PAD_code)

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
        image, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]

        preds = self(image, tgt_input)
        loss = self.criterion(preds.reshape(-1, preds.shape[-1]), tgt_expected.reshape(-1))

        self.loss_meter_val.update(loss.item(), image.size(0))   

        obj_class, bbox = self.tokenizer.decode(preds[0])
        gt_obj_class, gt_bbox = self.tokenizer.decode(tgt[0])
        print("Prepare vis")
        vis_image = self.visualize(image[0].permute(1, 2, 0).cpu().numpy(), gt_bbox, gt_obj_class, GT_COLOR, show=True)
        vis_image  = self.visualize(vis_image, bbox, obj_class, PRED_COLOR, show=True)
        print(f"Finish visualization")

        self.log('val_loss', self.loss_meter_val.avg, sync_dist=True)
        self.logger.log_image("Ground Truth vs Prediction", [vis_image], self.global_step)
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


    def visualize_bbox(self, img, bbox, class_name, color, thickness=1):
        """Visualizes a single bounding box on the image"""
        bbox = [int(item) for item in bbox]
        x_min, y_min, x_max, y_max = bbox

        print("Create rectangle")
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        
        print("Create Text")
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
        cv2.rectangle(img, (x_min, y_min), (x_min + text_width, y_min + int(text_height * 1.3)), color, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min+ int(text_height * 1.3)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35, 
            color=TEXT_COLOR, 
            lineType=cv2.LINE_AA,
        )
        print("Finish vis")
        return img


    def visualize(self, image, bboxes, category_ids, color=PRED_COLOR, show=True):
        img = image.copy()
        print(f"Create visualization for {category_ids} with bbox {bboxes}")
        for bbox, category_id in zip(bboxes, category_ids):
            if category_id > 0:
                class_name = self.cfg.id2cls[category_id]
                img = self.visualize_bbox(img, bbox, class_name, color)
        if show:
            plt.figure(figsize=(12, 12))
            plt.axis('off')
            plt.imshow(img)
            plt.show()
        return img