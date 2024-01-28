import lightning as L
from data.dataset import YCBDataset
import albumentations as A
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data.tokenizer import Tokenizer


def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length
    """
    image_batch, seq_batch = [], []
    for image, seq in batch:
        image_batch.append(image)
        seq_batch.append(seq)

    seq_batch = pad_sequence(
        seq_batch, padding_value=pad_idx, batch_first=True)
    if max_len:
        pad = torch.ones(seq_batch.size(0), max_len -
                         seq_batch.size(1)).fill_(pad_idx).long()
        seq_batch = torch.cat([seq_batch, pad], dim=1)
    image_batch = torch.stack(image_batch)
    return image_batch, seq_batch


class YCBDatamodule(L.LightningDataModule):
    def __init__(
            self, 
            cfg
        ):
        super().__init__()
        self.data_dir = cfg.ycb_path
        self.batch_size = cfg.batch_size
        self.cfg = cfg
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(cfg.img_size, cfg.img_size),
            A.Normalize(),
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['obj_class']})
        self.val_transforms = A.Compose([
            A.Resize(cfg.img_size, cfg.img_size),
            A.Normalize(),
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['obj_class']})

        self.tokenizer = Tokenizer(
                    num_classes=cfg.num_classes, 
                    num_bins=cfg.num_bins,
                    width=cfg.img_size, 
                    height=cfg.img_size, 
                    max_len=cfg.max_len
                )

    def setup(self, stage: str):
        if stage == "fit":
            self.ycb_train = YCBDataset(
                    self.data_dir + "train_real/", 
                    self.train_transforms,
                    self.tokenizer
                )
            self.ycb_val = YCBDataset(
                    self.data_dir + "test/",
                    self.val_transforms,
                    self.tokenizer
                )
        if stage == "test":
            self.ycb_test= YCBDataset(
                self.data_dir + "test/",
                self.val_transforms,
                self.tokenizer
            )

    def train_dataloader(self):
        return DataLoader(
            self.ycb_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, max_len=self.cfg.max_len, pad_idx=self.tokenizer.PAD_code),
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ycb_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, max_len=self.cfg.max_len, pad_idx=self.tokenizer.PAD_code),
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.ycb_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, max_len=self.cfg.max_len, pad_idx=self.tokenizer.PAD_code),
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )