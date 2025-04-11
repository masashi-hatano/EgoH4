import logging
import os
import random
import warnings

import hydra
import numpy as np
import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule.lit_egoexo4d_data_module import \
    EgoExo4DBodyHandPoseCombinedDataModule
from models.lit_EgoH4Trainer import EgoH4Trainer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg):
    # initialize random seeds
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # data module
    data_module = EgoExo4DBodyHandPoseCombinedDataModule(cfg)

    # model
    model = EgoH4Trainer(cfg)

    if torch.cuda.is_available() and len(cfg.devices):
        print(f"Using {len(cfg.devices)} GPUs !")

    train_logger = loggers.TensorBoardLogger("tensor_board", default_hp_metric=False)

    trainer = Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        max_epochs=cfg.trainer.optimizer.epochs,
        logger=train_logger,
        detect_anomaly=True,
        use_distributed_sampler=True,
    )

    if cfg.train:
        if cfg.resume_ckpt is not None:
            trainer.fit(model, data_module, ckpt_path=cfg.resume_ckpt)
        else:
            trainer.fit(model, data_module)
        print(trainer.callback_metrics)

    if cfg.test:
        logging.basicConfig(level=logging.DEBUG)
        trainer.test(model, data_module)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
