import logging
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from ema_pytorch import EMA

from utils.get_model import get_model
from utils.get_optimizer import get_optimizer

logger = logging.getLogger(__name__)


class EgoH4Trainer(pl.LightningModule):
    def __init__(self, cfg):
        super(EgoH4Trainer, self).__init__()
        self.cfg = cfg
        self.jpos_min = self.cfg.data_module.train.min_max[0]
        self.jpos_max = self.cfg.data_module.train.min_max[1]

        # model
        self.model = get_model(cfg, ckpt_pth=cfg.ckpt_pth)
        self.ema = EMA(
            self.model,
            beta=cfg.trainer.ema.decay,
            update_every=cfg.trainer.ema.update_every,
        )
        if cfg.ckpt_pth is not None:
            p = torch.load(cfg.ckpt_pth)
            od = p["module"]
            pretrained_dict = {}
            for k, v in od.items():
                if "ema." in k:
                    k = k.replace("_forward_module.ema.", "")
                    pretrained_dict[k] = v
            model_dict = self.ema.state_dict()
            model_dict.update(pretrained_dict)
            self.ema.load_state_dict(model_dict)
            logger.info(f"Model is initialized from {cfg.ckpt_pth}")

        # loss
        self.loss_fn = nn.L1Loss(reduction="none")

        # initialization
        self.training_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        self.scale_lr()
        self.trainer.fit_loop.setup_data()
        dataset = self.trainer.train_dataloader.dataset
        self.niter_per_epoch = len(dataset) // self.total_batch_size
        print("Number of training steps = %d" % self.niter_per_epoch)
        print(
            "Number of training examples per epoch = %d"
            % (self.total_batch_size * self.niter_per_epoch)
        )
        optimizer, scheduler = get_optimizer(
            self.cfg.trainer.optimizer, self.model, self.niter_per_epoch
        )
        if scheduler is not None:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def rearrange_input(self, input):
        cond_head = input["cond_head"]  # B X 2 X T X 9
        imgs_feat = input["imgs"]  # B X 2 X T X D
        body_3d = input["body_3d"]  # B X 2 X T X 17 X 3
        hand_3d = input["hand_3d"]  # B X 2 X T X 42 X 3 (right hand, left hand)
        hand_2d = input["hand_2d"]  # B X 2 X T X 42 X 2 (right hand, left hand)
        out_of_view = input["out_of_view"]  # B X 2 X T X 2
        body_valid_3d = input["body_valid_3d"]  # B X 2 X T X 17
        hand_valid_3d = input["hand_valid_3d"]  # B X 2 X T X 42
        hand_valid_2d = input["hand_valid_2d"]  # B X 2 X T X 42
        out_of_view_valid = input["out_of_view_valid"]  # B X 2 X T X 2
        extrinsics = input["extrinsics"]  # B X 2 X T X 4 X 4
        intrinsics = input["intrinsics"]  # B X 2 X 3 X 3
        T_cano_t0_world = input["T_cano_t0_world"]  # B X 2 X 4 X 4

        b, n, t, _ = cond_head.shape

        cond_head = rearrange(cond_head, "b n t c -> (b n) t c", b=b, n=n)
        imgs_feat = rearrange(imgs_feat, "b n t d -> (b n) t d", b=b, n=n)
        body_3d = rearrange(body_3d, "b n t j c -> (b n) t j c", b=b, n=n)
        hand_3d = rearrange(hand_3d, "b n t j c -> (b n) t j c", b=b, n=n)
        hand_2d = rearrange(hand_2d, "b n t j c -> (b n) t j c", b=b, n=n)
        out_of_view = rearrange(out_of_view, "b n t c -> (b n) t c", b=b, n=n)
        body_valid_3d = rearrange(body_valid_3d, "b n t j -> (b n) t j", b=b, n=n)
        hand_valid_3d = rearrange(hand_valid_3d, "b n t j -> (b n) t j", b=b, n=n)
        hand_valid_2d = rearrange(hand_valid_2d, "b n t j -> (b n) t j", b=b, n=n)
        out_of_view_valid = rearrange(
            out_of_view_valid, "b n t c -> (b n) t c", b=b, n=n
        )
        extrinsics = rearrange(extrinsics, "b n t j c -> (b n) t j c", b=b, n=n)
        intrinsics = rearrange(intrinsics, "b n j c -> (b n) j c", b=b, n=n)
        T_cano_t0_world = rearrange(T_cano_t0_world, "b n c d -> (b n) c d", b=b, n=n)

        input = {
            "cond_head": cond_head,
            "imgs": imgs_feat,
            "body_3d": body_3d,
            "hand_3d": hand_3d,
            "hand_2d": hand_2d,
            "out_of_view": out_of_view,
            "body_valid_3d": body_valid_3d,
            "hand_valid_3d": hand_valid_3d,
            "hand_valid_2d": hand_valid_2d,
            "out_of_view_valid": out_of_view_valid,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "T_cano_t0_world": T_cano_t0_world,
        }

        return input

    def training_step(self, batch, batch_idx):
        input = batch

        input = self.rearrange_input(input)
        cond_head = input["cond_head"]  # B X T X 9
        imgs_feat = input["imgs"]  # B X T X D
        body_3d = input["body_3d"]  # B X T X 17 X 3
        hand_3d = input["hand_3d"]  # B X T X 42 X 3 (right hand, left hand)
        hand_2d = input["hand_2d"]  # B X T X 42 X 2 (right hand, left hand)
        out_of_view = input["out_of_view"]  # B X T X 2
        body_valid_3d = input["body_valid_3d"]  # B X T X 17
        hand_valid_3d = input["hand_valid_3d"]  # B X T X 42
        hand_valid_2d = input["hand_valid_2d"]  # B X T X 42
        out_of_view_valid = input["out_of_view_valid"]  # B X T X 2
        extrinsics = input["extrinsics"]  # B X T X 4 X 4
        intrinsics = input["intrinsics"]  # B X 3 X 3
        T_cano_t0_world = input["T_cano_t0_world"]  # B X 4 X 4

        # need to delete wrist as they are counted twice
        body_3d = torch.cat([body_3d[:, :, :9, :], body_3d[:, :, 11:, :]], dim=2)
        body_valid_3d = torch.cat(
            [body_valid_3d[:, :, :9], body_valid_3d[:, :, 11:]], dim=2
        )
        joints_3d = torch.cat([body_3d, hand_3d], dim=2)  # B X T X 57 X 3
        joints_valid_3d = torch.cat([body_valid_3d, hand_valid_3d], dim=2)  # B X T X 57

        # 2d joints
        right_wrist_2d, left_wrist_2d = hand_2d[:, :, 0, :], hand_2d[:, :, 21, :]
        right_wrist_valid, left_wrist_valid = (
            hand_valid_2d[:, :, 0],
            hand_valid_2d[:, :, 21],
        )
        cond_wrist = torch.cat(
            [right_wrist_2d.unsqueeze(2), left_wrist_2d.unsqueeze(2)], dim=2
        )
        cond_wrist_valid = torch.cat(
            [right_wrist_valid.unsqueeze(2), left_wrist_valid.unsqueeze(2)], dim=2
        )  # B X T X 2
        cond_wrist_ = cond_wrist * cond_wrist_valid[:, :, :, None]  # B X T X 2 X 2
        cond_wrist = rearrange(cond_wrist_, "b t j c -> b t (j c)")  # B x T x 4

        cond = torch.cat([cond_head, cond_wrist], dim=-1)

        # [B, T, 57, 3] -> [B, T, 57*3]
        x_start = rearrange(joints_3d, "b t j c -> b t (j c)", j=57, c=3)

        # cond: [B, T, C]
        # imgs_feat: [B, T, D]
        # x_start: [B, T, 17*3]
        loss_output = self.model(
            x_start,
            cond,
            imgs_feat,
            joints_valid_3d,
            T_cano_t0_world,
            cond_wrist_,
            cond_wrist_valid,
            out_of_view,
            out_of_view_valid,
            extrinsics,
            intrinsics,
        )

        lambda_body = self.cfg.trainer.lambda_body
        lambda_repro = self.cfg.trainer.lambda_repro
        lambda_vis = self.cfg.trainer.lambda_vis

        loss_3d_jpos_body_obs = loss_output["loss_3d_jpos_body_obs"]
        loss_3d_jpos_hand_obs = loss_output["loss_3d_jpos_hand_obs"]
        loss_3d_jpos_body_fut = loss_output["loss_3d_jpos_body_fut"]
        loss_3d_jpos_hand_fut = loss_output["loss_3d_jpos_hand_fut"]
        loss_repro = loss_output["loss_repro"]
        loss_vis = loss_output["loss_vis"]

        loss_body = loss_3d_jpos_body_obs + loss_3d_jpos_body_fut
        loss_hand = loss_3d_jpos_hand_obs + loss_3d_jpos_hand_fut
        loss_3d_joints = (loss_hand + lambda_body * loss_body) / (1 + lambda_body)

        loss = loss_3d_joints + lambda_repro * loss_repro + lambda_vis * loss_vis

        outputs = {
            "train_loss": loss.item(),
            "loss_body_obs": loss_3d_jpos_body_obs.item(),
            "loss_hand_obs": loss_3d_jpos_hand_obs.item(),
            "loss_body_fut": loss_3d_jpos_body_fut.item(),
            "loss_hand_fut": loss_3d_jpos_hand_fut.item(),
            "loss_body": loss_body.item(),
            "loss_hand": loss_hand.item(),
            "loss_repro": loss_repro.item(),
            "loss_vis": loss_vis.item(),
        }

        # update ema
        self.ema.update()

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log_dict(outputs)
        self.training_step_outputs.append(outputs)
        return loss

    def on_train_epoch_start(self):
        # shuffle the hand data loader
        hand_take_uids = (
            self.trainer.train_dataloader.dataset.hand_dataset.poses_takes_uids
        )
        hand_start_frames = (
            self.trainer.train_dataloader.dataset.hand_dataset.start_frames
        )
        hand_take_names = self.trainer.train_dataloader.dataset.hand_dataset.take_names
        hand_seq_data = self.trainer.train_dataloader.dataset.hand_dataset.data
        lists = list(
            zip(hand_take_uids, hand_start_frames, hand_take_names, hand_seq_data)
        )
        random.shuffle(lists)
        hand_take_uids, hand_start_frames, hand_take_names, hand_seq_data = zip(*lists)
        self.trainer.train_dataloader.dataset.hand_dataset.poses_takes_uids = (
            hand_take_uids
        )
        self.trainer.train_dataloader.dataset.hand_dataset.start_frames = (
            hand_start_frames
        )
        self.trainer.train_dataloader.dataset.hand_dataset.take_names = hand_take_names
        self.trainer.train_dataloader.dataset.hand_dataset.data = hand_seq_data

    def on_train_epoch_end(self):
        train_loss = np.mean(
            [output["train_loss"] for output in self.training_step_outputs]
        )
        self.training_step_outputs.clear()

        if (self.trainer.current_epoch + 1) % self.cfg.save_ckpt_freq == 0:
            self.trainer.save_checkpoint(
                f"checkpoints/epoch={self.trainer.current_epoch:02d}-loss={train_loss:.5f}"
            )

    def validation_step(self, batch, batch_idx):
        return NotImplementedError

    def on_validation_epoch_end(self):
        return NotImplementedError

    def calc_jve(self, pred, gt, valid):
        pred_velo = torch.diff(pred, dim=1) * 10
        gt_velo = torch.diff(gt, dim=1) * 10
        valid_velo = valid[:, 1:] * valid[:, :-1]
        # Calculate per-joint velocity error
        velocity_error = torch.linalg.norm(
            pred_velo - gt_velo, axis=-1
        )  # Shape (T-1, J)
        velocity_error = velocity_error * valid_velo
        velocity_error = velocity_error[velocity_error != 0]
        return velocity_error

    def test_step(self, batch, batch_idx):
        input = batch

        cond_head = input["cond_head"]  # B X T X 9
        imgs_feat = input["imgs"]  # B X T X D
        body_3d = input["body_3d"]  # B X T X 17 X 3
        hand_3d = input["hand_3d"]  # B X T X 42 X 3 (right hand, left hand)
        hand_2d = input["hand_2d"]  # B X T X 42 X 2 (right hand, left hand)
        body_valid_3d = input["body_valid_3d"]  # B X T X 17
        hand_valid_3d = input["hand_valid_3d"]  # B X T X 42
        hand_valid_2d = input["hand_valid_2d"]  # B X T X 42
        num_out_of_view = input["num_out_of_view"]  # B X 2

        # Remove duplicate wrist entries
        body_3d = torch.cat([body_3d[:, :, :9, :], body_3d[:, :, 11:, :]], dim=2)
        body_valid_3d = torch.cat(
            [body_valid_3d[:, :, :9], body_valid_3d[:, :, 11:]], dim=2
        )
        # Concatenate body and hand joints
        joints_3d = torch.cat([body_3d, hand_3d], dim=2)  # B X T X 57 X 3
        joints_valid_3d = torch.cat([body_valid_3d, hand_valid_3d], dim=2)  # B X T X 57

        # 2d joints
        right_wrist_2d, left_wrist_2d = hand_2d[:, :, 0, :], hand_2d[:, :, 21, :]
        right_wrist_valid, left_wrist_valid = (hand_valid_2d[:, :, 0], hand_valid_2d[:, :, 21],)
        cond_wrist = torch.cat([right_wrist_2d.unsqueeze(2), left_wrist_2d.unsqueeze(2)], dim=2)
        cond_wrist_valid = torch.cat([right_wrist_valid.unsqueeze(2), left_wrist_valid.unsqueeze(2)], dim=2)  # B X T X 2
        cond_wrist_ = cond_wrist * cond_wrist_valid[:, :, :, None]  # B X T X 2 X 2
        cond_wrist = rearrange(cond_wrist_, "b t j c -> b t (j c)")  # B x T x 4

        # Prepare conditioning input
        cond = torch.cat([cond_head, cond_wrist], dim=-1)

        # Flatten joint: [B, T, 57, 3] -> [B, T, 57*3]
        x_start = rearrange(joints_3d, "b t j c -> b t (j c)", j=57, c=3)

        # Generate predictions
        # cond: [B, T, C]
        # x_start: [B, T, 17*3]
        # pred: [B, T, 17*3]
        pred = self.ema.ema_model.sample(x_start, cond, imgs_feat)

        # [B, T, 17*3] -> [B, T, 17, 3]
        pred = rearrange(pred, "b t (j c) -> b t j c", j=57, c=3)
        x_start = rearrange(x_start, "b t (j c) -> b t j c", j=57, c=3)

        # De-normalize jpos to original scale
        pred = self.de_normalize_min_max(pred)
        x_start = self.de_normalize_min_max(x_start)

        # Compute metrics
        outputs = {}
        per_timestep_ranges = {f"F={x + 1}" : slice(x + 20, x + 21) for x in range(10)}
        timestep_ranges = {
            "obs_ade": slice(0, 20),
            "fut_ade": slice(20, 30),
            "obs_fde": slice(19, 20),
            "fut_fde": slice(29, 30),
        }
        part_ranges = {
            "full": [slice(0, 57)],
            "body": [slice(0, 15), slice(15, 16), slice(36, 37)],  # body, right wrist, left wrist
            "wrist": [slice(15, 16), slice(36, 37)],  # right wrist, left wrist
            "hand": [slice(15, 36), slice(36, 57)],  # right hand, left hand
        }
        hand_side_list = ["right", "left"]

        for part, part_range in part_ranges.items():
            pred_part = torch.cat([pred[:, :, slice, :] for slice in part_range], dim=2)
            x_start_part = torch.cat([x_start[:, :, slice, :] for slice in part_range], dim=2)
            joints_valid_3d_part = torch.cat([joints_valid_3d[:, :, slice] for slice in part_range], dim=2)
            for timestep, timestep_range in timestep_ranges.items():
                outputs[f"test_{part}_mpjpe_{timestep}"] = self.calc_jpe(pred_part[:, timestep_range], x_start_part[:, timestep_range], joints_valid_3d_part[:, timestep_range]).detach()
                # Compute MPJVE
                if part == "body" and "fde" not in timestep:
                    outputs[f"test_{part}_mpjve_{timestep}"] = self.calc_jve(pred_part[:, timestep_range], x_start_part[:, timestep_range], joints_valid_3d_part[:, timestep_range]).detach()

        # Subset evaluation: in-view, out-of-view
        for b in range(num_out_of_view.shape[0]):
            for i, _ in enumerate(hand_side_list):
                pred_wrist_side = pred[b, :, part_ranges["wrist"][i], :]
                x_start_wrist_side = x_start[b, :, part_ranges["wrist"][i], :]
                joints_valid_3d_wrist_side = joints_valid_3d[b, :, part_ranges["wrist"][i]]
                pred_hand_side = pred[b, :, part_ranges["hand"][i], :]
                x_start_hand_side = x_start[b, :, part_ranges["hand"][i], :]
                joints_valid_3d_hand_side = joints_valid_3d[b, :, part_ranges["hand"][i]]
                for timestep, timestep_range in timestep_ranges.items():
                    view, ratio = self.get_view_ratio(num_out_of_view[b, i])
                    key_wrist_ratio = f"test_wrist_mpjpe_{timestep}_{ratio:.1f}"
                    key_wrist_view = f"test_wrist_mpjpe_{timestep}_{view}"
                    key_hand_view = f"test_hand_mpjpe_{timestep}_{view}"
                    jpe_wrist = self.calc_jpe(pred_wrist_side[timestep_range], x_start_wrist_side[timestep_range], joints_valid_3d_wrist_side[timestep_range]).detach()
                    jpe_hand = self.calc_jpe(pred_hand_side[timestep_range], x_start_hand_side[timestep_range], joints_valid_3d_hand_side[timestep_range]).detach()
                    outputs.setdefault(key_wrist_ratio, []).extend(jpe_wrist)
                    outputs.setdefault(key_wrist_view, []).extend(jpe_wrist)
                    outputs.setdefault(key_hand_view, []).extend(jpe_hand)

        # Subset evaluation: per-timestep
        for part, part_range in part_ranges.items():
            if part == "full" or part == "body":
                continue
            for timestep, timestep_range in per_timestep_ranges.items():
                pred_part = torch.cat([pred[:, :, slice, :] for slice in part_range], dim=2)
                x_start_part = torch.cat([x_start[:, :, slice, :] for slice in part_range], dim=2)
                joints_valid_3d_part = torch.cat([joints_valid_3d[:, :, slice] for slice in part_range], dim=2)
                outputs[f"test_{part}_mpjpe_{timestep}"] = self.calc_jpe(pred_part[:, timestep_range], x_start_part[:, timestep_range], joints_valid_3d_part[:, timestep_range]).detach()

        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        # Aggregate losses
        key_metric_name_dict = {}
        losses = {}
        for output in self.test_step_outputs:
            for key, value in output.items():
                if key not in key_metric_name_dict.keys():
                    if len(key.split("_")) == 4:
                        part, metric, timestep = key.split("_")[1:]
                        metric_name = f"{part.upper()}: {metric.upper()} {timestep.upper()}"
                    elif len(key.split("_")) == 5:
                        part, metric, timestep, avg = key.split("_")[1:]
                        metric_name = f"{part.upper()}: {metric.upper()} {avg.upper()} {timestep.upper()}"
                    elif len(key.split("_")) == 6:
                        part, metric, timestep, avg, subset = key.split("_")[1:]
                        metric_name = f"{part.upper()}: {metric.upper()} {avg.upper()} {timestep.upper()} {subset}"
                    key_metric_name_dict[key] = metric_name

                losses.setdefault(key, []).extend(value)

        # Convert lists to tensors
        losses = {k: torch.tensor(v).view(-1) for k, v in losses.items()}

        # Logging results
        for key, metric in key_metric_name_dict.items():
            self.log(metric, losses[key].mean(), on_step=False)

        # Clear test step outputs
        self.test_step_outputs.clear()

    def de_normalize_min_max(self, jpos):
        jpos = (jpos + 1) * 0.5  # [0, 1] range
        de_jpos = jpos * (self.jpos_max - self.jpos_min) + self.jpos_min
        return de_jpos  # B X T X 17 X 3

    def calc_jpe(self, pred, gt, visible):
        # pred: B X T X 17 X 3 or T X 17 X 3
        # gt: B X T X 17 X 3 or T X 17 X 3
        # visible: B X T X 17 or T X 17
        diff = pred - gt
        diff = diff**2
        diff = diff.sum(dim=-1)
        diff = diff**0.5
        jpe = diff * visible
        jpe = jpe[jpe != 0]
        return jpe

    def get_view_ratio(self, num_out_of_view):
        if num_out_of_view == 0:
            return "in-view", 0.0
        for threshold, ratio in zip(range(4, 21, 4), [0.2, 0.4, 0.6, 0.8, 1.0]):
            if num_out_of_view <= threshold:
                return "out-of-view", ratio

        raise ValueError("Unexpected Out-of-view value")

    def scale_lr(self):
        self.total_batch_size = self.cfg.data_module.train.batch_size * len(
            self.cfg.devices
        )
        self.cfg.trainer.optimizer.lr = (
            self.cfg.trainer.optimizer.lr * self.total_batch_size / 128
        )
        print("LR = %.8f" % self.cfg.trainer.optimizer.lr)
        print("Batch size = %d" % self.total_batch_size)
