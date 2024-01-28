from utils.utils import seed_everything

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from models import Encoder, Decoder, Pix2Seq
from data.datamodule import YCBDatamodule
import string
import random
from config import CFG


if __name__ == '__main__':
    seed_everything(42)
    # ---------------------------------------------------------------------------------
    # Prepare config
    cfg = CFG
    
    # ---------------------------------------------------------------------------------
    # Prepare data
    datamodule = YCBDatamodule(cfg)
    CFG.pad_idx = datamodule.tokenizer.PAD_code
    datamodule.setup("fit")
    
    cfg.num_training_steps = CFG.epochs * \
        (len(datamodule.ycb_train) // CFG.batch_size)
    cfg.num_warmup_steps = int(0.05 * cfg.num_training_steps)

    # ---------------------------------------------------------------------------------
    # Prepare Lightning module
    encoder = Encoder(model_name=CFG.model_name, pretrained=True, out_dim=256)
    decoder = Decoder(cfg, vocab_size=datamodule.tokenizer.vocab_size,
                      encoder_length=CFG.num_patches, dim=256, num_heads=8, num_layers=6)

    module = Pix2Seq(cfg, encoder, decoder, datamodule.tokenizer)

    # ---------------------------------------------------------------------------------
    # Prepare logger
    letters = string.ascii_lowercase
    run_name = "".join(random.choice(letters) for i in range(8))
    wandb_logger = WandbLogger(project="Pix2Seq", offline=True, name=run_name)

    ### Monitor learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    ### Prepare the checkpointing
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        every_n_epochs=1,
        save_last=True,
        mode="min",
        monitor="train_loss",
        dirpath=f"./checkpoints/",
        filename=str(run_name)+"_ckpt_epoch_{epoch:02d}_loss_{train_loss:.2f}"
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"

    # ---------------------------------------------------------------------------------
    # Prepare trainer
    # TODO : prepare args
    trainer_args = {
        "max_epochs": cfg.epochs,
        "callbacks": [lr_monitor, checkpoint_callback],
        "logger": wandb_logger,
        "precision": cfg.precision,
        "accelerator": "gpu",
        "devices": cfg.gpus,
        "num_nodes": cfg.nodes,
        "strategy": cfg.strategy,
        "num_sanity_val_steps": 1,
        "log_every_n_steps": 10,
        "check_val_every_n_epoch": 1
    }

    trainer = L.Trainer(**trainer_args)
    trainer.fit(module, datamodule)
