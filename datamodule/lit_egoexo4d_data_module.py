from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader

from datamodule.dataset.egoexo4d_body_and_hand_pose_combined_dataset import \
    EgoExo4DBodyHandPoseCombinedDataset
from datamodule.dataset.egoexo4d_body_pose_dataset import \
    EgoExo4DBodyPoseDataset
from datamodule.dataset.egoexo4d_hand_body_pose_dataset import \
    EgoExo4DHandBodyPoseDataset


class EgoExo4DDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(EgoExo4DDataModule, self).__init__()
        self.cfg = cfg
        self.dm_cfg = cfg.data_module

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = EgoExo4DBodyHandPoseCombinedDataset(self.dm_cfg.train, root=self.dm_cfg.root, split="train")
        elif stage == "test":
            # concatenation: manual body & manual hand + auto body
            self.test_body_dataset = EgoExo4DBodyPoseDataset(self.dm_cfg.test, root=self.dm_cfg.root, split="val")
            self.test_hand_dataset = EgoExo4DHandBodyPoseDataset(self.dm_cfg.test, root=self.dm_cfg.root, split="val")
            self.test_dataset = ConcatDataset([self.test_body_dataset, self.test_hand_dataset])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.dm_cfg.train.batch_size,
            shuffle=self.dm_cfg.train.shuffle,
            num_workers=self.dm_cfg.train.num_workers,
            drop_last=self.dm_cfg.train.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.dm_cfg.test.batch_size,
            shuffle=False,
            num_workers=self.dm_cfg.test.num_workers,
        )
