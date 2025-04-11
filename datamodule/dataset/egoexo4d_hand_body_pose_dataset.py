import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms as transforms_3d
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from datamodule.utils.transform import ToTensor


class EgoExo4DHandBodyPoseDataset(Dataset):
    def __init__(self, cfg, root, split="train"):
        self.cfg = cfg
        self.root = Path(root)
        self.gt_output_dir = Path(root, "gt_output")
        self.split = split
        self.window_size = cfg.window_size
        self.slice_size = cfg.slice_size
        self.interval = cfg.interval
        self.min_jpos = np.inf if cfg.min_max is None else cfg.min_max[0]
        self.max_jpos = -np.inf if cfg.min_max is None else cfg.min_max[1]
        self.mean = cfg.mean
        self.std = cfg.std

        # Image transformation
        self.transform = transforms.Compose([
            ToTensor(),
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ])

        # initialize the data
        self.take_names = []
        self.start_frames = []
        self.data = []
        self.poses_takes_uids = []

        self.out_of_view_obs = 0
        self.oov_frame = 0
        self.out_of_view = 0
        self.frame_num = 0

        self._load_data()

    def _load_data(self):
        # Load all takes metadata
        takes = json.load(open(Path(self.root, "takes.json")))

        # Load GT annotation
        gt_anno_path = Path(
            self.gt_output_dir,
            "annotation",
            "manual",
            f"ego_pose_gt_anno_{self.split}_combined.json",
        )
        # Check gt-anno file existence
        assert gt_anno_path.exists()
        gt_anno = json.load(open(gt_anno_path))
        # Extract frames with annotations for all takes
        for take_uid, take_anno in tqdm(gt_anno.items()):
            # Get current take's metadata
            take = [t for t in takes if t["take_uid"] == take_uid]
            assert len(take) == 1, f"Take: {take_uid} does not exist"
            take = take[0]
            # Get current take's name and aria camera name
            take_name = take["take_name"]

            # for loop to create sequences of poses with a window size
            pose_frames = list(take_anno.keys())
            pose_frames.remove("intrinsics")

            for seq_idx in range(0, len(pose_frames) - self.window_size + 1, self.slice_size):
                seq_frames = pose_frames[seq_idx:seq_idx + self.window_size]
                if not self.check_sequence(seq_frames):
                    continue
                seq_data = self.sequence_canonicalization(
                    take_anno,
                    seq_frames,
                    take_uid
                )

                out_right, out_left = self.count_out_of_view_obs(seq_data)
                seq_data["num_out_of_view"] = (out_right, out_left)
                if out_right != 0:
                    self.out_of_view_obs += 1
                if out_left != 0:
                    self.out_of_view_obs += 1
                if out_right != 0 or out_left != 0:
                    self.out_of_view += 1

                self.start_frames.append(seq_frames[0])
                self.data.append(seq_data)
                self.poses_takes_uids.append(take_uid)
                self.take_names.append(take_name)

        print("Dataset lenght: {}".format(len(self.data)))
        print("Split: {}".format(self.split))
        print(self.min_jpos)
        print(self.max_jpos)
        print("Out of view observations: {}".format(self.out_of_view_obs))
        print(f"{self.out_of_view=}")
        print("Total observations: {}".format(self.frame_num))
        print("Total out of view frames: {}".format(self.oov_frame))

    def count_out_of_view_obs(self, seq_data):
        out_of_view = seq_data["out_of_view"]
        out_right = 0
        out_left = 0
        for i in range(20):
            # right
            if out_of_view[i][0]:
                out_right += 1
            # left
            if out_of_view[i][1]:
                out_left += 1
        return out_right, out_left

    def check_sequence(self, seq):
        for i in range(len(seq) - 1):
            if int(seq[i+1]) - int(seq[i]) == self.interval:
                pass
            else:
                return False
        return True

    def rz(self, angle_rad):
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                  0,                  1]
        ])

    def sequence_canonicalization(self, anno, seq_frames, take_uid):
        # initialize the sequence data
        extrinsics_seq = []
        T_cano_t0_cam_seq = []
        T_world_camera_seq = []
        body_3d_pose_seq = []
        hand_3d_pose_seq = []
        hand_2d_pose_seq = []
        body_valid_3d_seq = []
        hand_valid_3d_seq = []
        hand_valid_2d_seq = []
        out_of_view_seq = []
        out_of_view_valid_seq = []

        # assert len(seq_frames) == 30

        # camera pose for the first frame of the sequence: t=frame
        extrinsic_t0 = anno[seq_frames[0]]["extrinsics"]
        T_camera_t0_world = np.eye(4)
        T_camera_t0_world[:3, :] = np.array(extrinsic_t0)
        T_world_camera_t0 = np.linalg.inv(T_camera_t0_world)

        # calc canonical to world transformation at the first frame
        t_world_cano_t0 = np.array([T_world_camera_t0[0, 3], T_world_camera_t0[1, 3], T_world_camera_t0[2, 3]])
        forward_t0_world = np.dot(T_world_camera_t0[:3, :3], np.array([0.0, 0.0, 1]).T).T
        R_cano_t0_world = self.rz(-np.arctan2(forward_t0_world[1], forward_t0_world[0]))  # align with x-axis
        t_cano_t0_world = -np.dot(R_cano_t0_world, t_world_cano_t0.T).T
        T_cano_t0_world = np.eye(4)
        T_cano_t0_world[:3, :3] = R_cano_t0_world
        T_cano_t0_world[:3, 3] = t_cano_t0_world

        # camera pose for the first frame of the sequence: t=frame
        extrinsic_tobs = anno[seq_frames[19]]["extrinsics"]
        T_camera_tobs_world = np.eye(4)
        T_camera_tobs_world[:3, :] = np.array(extrinsic_tobs)
        T_world_camera_tobs = np.linalg.inv(T_camera_tobs_world)

        # calc canonical to world transformation at the first frame
        t_world_cano_tobs = np.array([T_world_camera_tobs[0, 3], T_world_camera_tobs[1, 3], T_world_camera_tobs[2, 3]])
        forward_tobs_world = np.dot(T_world_camera_tobs[:3, :3], np.array([0.0, 0.0, 1]).T).T
        R_cano_tobs_world = self.rz(-np.arctan2(forward_tobs_world[1], forward_tobs_world[0]))  # align with x-axis
        t_cano_tobs_world = -np.dot(R_cano_tobs_world, t_world_cano_tobs.T).T
        T_cano_tobs_world = np.eye(4)
        T_cano_tobs_world[:3, :3] = R_cano_tobs_world
        T_cano_tobs_world[:3, 3] = t_cano_tobs_world

        # camera pose for the second last frame of the sequence: t=tobs-1
        extrinsic_tobs_m1 = anno[seq_frames[18]]["extrinsics"]
        T_camera_tobs_m1_world = np.eye(4)
        T_camera_tobs_m1_world[:3, :] = np.array(extrinsic_tobs_m1)
        T_world_camera_tobs_m1 = np.linalg.inv(T_camera_tobs_m1_world)

        # calc canonical to world transformation at the second last frame
        t_world_cano_tobs_m1 = np.array([T_world_camera_tobs_m1[0, 3], T_world_camera_tobs_m1[1, 3], T_world_camera_tobs_m1[2, 3]])
        forward_tobs_m1_world = np.dot(T_world_camera_tobs_m1[:3, :3], np.array([0.0, 0.0, 1]).T).T
        R_cano_tobs_m1_world = self.rz(-np.arctan2(forward_tobs_m1_world[1], forward_tobs_m1_world[0]))  # align with x-axis
        t_cano_tobs_m1_world = -np.dot(R_cano_tobs_m1_world, t_world_cano_tobs_m1.T).T
        T_cano_tobs_m1_world = np.eye(4)
        T_cano_tobs_m1_world[:3, :3] = R_cano_tobs_m1_world
        T_cano_tobs_m1_world[:3, 3] = t_cano_tobs_m1_world

        for frame in seq_frames:
            # perform the transformation for each joint in the frame if available
            current_anno = anno[frame]
            assert len(current_anno) != 0

            # camera pose for the current frame
            extrinsic_t = current_anno["extrinsics"]  # 3 * 4
            T_camera_world_t = np.eye(4)
            T_camera_world_t[:3, :] = np.array(extrinsic_t)
            extrinsics_seq.append(T_camera_world_t)

            # T_world_camera_t
            T_world_camera_t = np.linalg.inv(T_camera_world_t)
            T_world_camera_seq.append(T_world_camera_t[:3, :])

            # T_cano_0_cam
            T_cano_t0_cam_t = T_cano_t0_world @ T_world_camera_t
            T_cano_t0_cam_seq.append(T_cano_t0_cam_t[:3, :])

            # body 3d pose
            body_3d = np.array(current_anno["body_3d"])  # 17 joints X 3 dimensions
            body_3d = np.concatenate([body_3d, np.ones((17, 1))], axis=1)  # 17 joints X 4 dimensions
            body_pose = np.dot(T_cano_t0_world, body_3d.T).T[:, :3]  # 17 joints X 3 dimensions

            body_3d_pose_seq.append(body_pose)
            body_valid_3d = np.array(current_anno["body_valid_3d"])
            body_valid_3d_seq.append(body_valid_3d)

            # hand 2d pose
            right_hand_2d = np.array(current_anno["right_hand_2d"]) / self.cfg.img_size  # 21 joints * 2 dimensions for right hand
            left_hand_2d = np.array(current_anno["left_hand_2d"]) / self.cfg.img_size  # 21 joints * 2 dimensions for left hand
            hand_2d_pose = np.concatenate([right_hand_2d, left_hand_2d], axis=0)
            hand_2d_pose_seq.append(hand_2d_pose)
            # validity of hand 2d pose
            right_hand_2d_placement = np.array(current_anno["right_hand_2d_placement"])
            left_hand_2d_placement = np.array(current_anno["left_hand_2d_placement"])
            right_hand_valid_2d = np.array(current_anno["right_hand_valid_2d"])
            left_hand_valid_2d = np.array(current_anno["left_hand_valid_2d"])
            right_hand_manual_valid_2d = np.where(np.logical_and(right_hand_2d_placement=='manual', right_hand_valid_2d), True, False)
            left_hand_manual_valid_2d = np.where(np.logical_and(left_hand_2d_placement=='manual', left_hand_valid_2d), True, False)
            hand_valid_2d = np.concatenate([right_hand_manual_valid_2d, left_hand_manual_valid_2d], axis=0)
            hand_valid_2d_seq.append(hand_valid_2d)

            # hand 3d pose
            right_hand_pose = np.array(current_anno["right_hand_3d"])  # 21 joints * 3 dimensions for right hand
            left_hand_pose = np.array(current_anno["left_hand_3d"])  # 21 joints * 3 dimensions for left hand
            hand_pose = np.concatenate([right_hand_pose, left_hand_pose], axis=0)  # 42 joints * 3 dimensions
            hand_pose = np.concatenate([hand_pose, np.ones((42, 1))], axis=1)  # 42 joints * 4 dimensions
            hand_pose = np.dot(T_cano_t0_world, hand_pose.T).T[:, :3]  # 42 joints * 3 dimensions
            hand_3d_pose_seq.append(hand_pose)
            right_hand_valid_3d = np.array(current_anno["right_hand_valid_3d"])  # 21 joints for right hand
            left_hand_valid_3d = np.array(current_anno["left_hand_valid_3d"])  # 21 joints for left hand
            hand_valid_3d = np.concatenate([right_hand_valid_3d, left_hand_valid_3d], axis=0)  # 42 joints
            hand_valid_3d_seq.append(hand_valid_3d)

            # out of view flag
            right_out = np.array(current_anno["out-of-view"]["right"])
            left_out = np.array(current_anno["out-of-view"]["left"])
            right_out_valid = False if right_out == np.nan else True
            left_out_valid = False if left_out == np.nan else True
            out_of_view_seq.append([right_out, left_out])
            out_of_view_valid_seq.append([right_out_valid, left_out_valid])

        seq_data = {
            "T_world_camera": T_world_camera_seq,
            "T_cano_cam": T_cano_t0_cam_seq,
            "body_3d_pose": body_3d_pose_seq,
            "hand_3d_pose": hand_3d_pose_seq,
            "hand_2d_pose": hand_2d_pose_seq,
            "out_of_view": out_of_view_seq,
            "body_valid_3d": body_valid_3d_seq,
            "hand_valid_3d": hand_valid_3d_seq,
            "hand_valid_2d": hand_valid_2d_seq,
            "out_of_view_valid": out_of_view_valid_seq,
            "take_uid": take_uid,
            "extrinsics": extrinsics_seq,
            "intrinsics": anno["intrinsics"],
            "T_cano_world": T_cano_t0_world,
            "T_cano_tobs_world": T_cano_tobs_world,
            "T_cano_tobs_m1_world": T_cano_tobs_m1_world
        }
        return seq_data

    def min_max(self, pose, valid):
        pose = pose[valid]
        if len(pose):
            min_jpos = np.min(pose)
            max_jpos = np.max(pose)
            if min_jpos < self.min_jpos:
                self.min_jpos = min_jpos
            if max_jpos > self.max_jpos:
                self.max_jpos = max_jpos

    def normalize_jpos_min_max(self, ori_jpos):
        # ori_jpos: T X J X 3
        normalized_jpos = (ori_jpos - self.min_jpos) / (self.max_jpos - self.min_jpos)
        normalized_jpos = normalized_jpos * 2 - 1  # [-1, 1] range
        return normalized_jpos  # T X J X 3

    def __getitem__(self, index):
        take_uid = self.poses_takes_uids[index]
        start_frame = self.start_frames[index]
        take_name = self.take_names[index]
        seq_data = self.data[index]

        # 3d auto body pose
        body_3d_pose = torch.Tensor(np.array(seq_data["body_3d_pose"]))  # T X 17 X 3
        body_valid_3d = torch.Tensor(np.array(seq_data["body_valid_3d"]))  # T X 17

        # 3d hand pose
        hand_3d_pose = torch.Tensor(np.array(seq_data["hand_3d_pose"]))  # T X 42 X 3
        hand_valid_3d = torch.Tensor(np.array(seq_data["hand_valid_3d"]))  # T X 42

        # use the wrist annotations from 3d hand pose
        body_3d_pose[:, 9] = hand_3d_pose[:, 21]  # T X 3 (left)
        body_3d_pose[:, 10] = hand_3d_pose[:, 0]  # T X 3 (right)
        body_valid_3d[:, 9] = hand_valid_3d[:, 21]  # T (left)
        body_valid_3d[:, 10] = hand_valid_3d[:, 0]  # T (right)

        # 2d hand pose
        hand_2d_pose = torch.Tensor(np.array(seq_data["hand_2d_pose"]))  # T X 42 X 2
        hand_valid_2d = torch.Tensor(np.array(seq_data["hand_valid_2d"]))  # T X 42

        # out of view flag
        out_of_view = torch.Tensor(np.array(seq_data["out_of_view"]))  # T X 2
        out_of_view_valid = torch.Tensor(np.array(seq_data["out_of_view_valid"]))  # T X 2

        # Min-max normalization
        body_3d_pose = self.normalize_jpos_min_max(body_3d_pose)
        hand_3d_pose = self.normalize_jpos_min_max(hand_3d_pose)

        T_cano_t0_cam = torch.Tensor(np.array(seq_data["T_cano_cam"]))  # T X 4 X 4
        translation = T_cano_t0_cam[:, :3, 3]  # T X 3
        rotation_mat = T_cano_t0_cam[:, :3, :3]  # T X 3 X 3
        rotation_6d = transforms_3d.matrix_to_rotation_6d(rotation_mat)  # T X 6
        cond_head = torch.cat([translation, rotation_6d], dim=1)
        T_cano_t0_world = torch.Tensor(np.array(seq_data["T_cano_world"]))  # 4 X 4
        T_cano_tobs_world = torch.Tensor(np.array(seq_data["T_cano_tobs_world"]))  # 4 X 4
        T_cano_tobs_m1_world = torch.Tensor(np.array(seq_data["T_cano_tobs_m1_world"]))  # 4 X 4
        assert cond_head.shape == (self.window_size, 9)
        extrinsics = torch.Tensor(np.array(seq_data["extrinsics"]))  # T X 4 X 4
        intrinsics = torch.Tensor(np.array(seq_data["intrinsics"]))  # 3 X 3

        num_out_of_view = torch.Tensor(np.array(seq_data["num_out_of_view"]))  # 2

        cond_head = torch.cat([translation, rotation_6d], dim=1)  # T X 9
        assert cond_head.shape == (self.window_size, 9)

        # filter out the invalid poses
        body_3d_pose[body_valid_3d == 0] = -1
        hand_3d_pose[hand_valid_3d == 0] = -1
        hand_2d_pose[hand_valid_2d == 0] = -1

        # load RGB imgs features
        imgs = []
        for i in range(20):
            frame_number = int(start_frame) + self.interval * i
            img_path = Path(self.gt_output_dir, "imgs_feat", self.split, take_name, f"{frame_number:06d}.pt")
            img = torch.load(img_path)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)  # T X D

        hand_data = {
            "cond_head": cond_head,  # T X 9
            "imgs": imgs,  # T X D
            "body_3d": body_3d_pose,  # T X 17 X 3
            "body_valid_3d": body_valid_3d,  # T X 17
            "hand_3d": hand_3d_pose,  # T X 42 X 3
            "hand_valid_3d": hand_valid_3d,  # T X 42
            "hand_2d": hand_2d_pose,  # T X 42 X 2
            "hand_valid_2d": hand_valid_2d,  # T X 42
            "out_of_view": out_of_view,  # T X 2
            "out_of_view_valid": out_of_view_valid,  # T X 2
            "T_cano_t0_world": T_cano_t0_world,  # 4 X 4
            "T_cano_tobs_world": T_cano_tobs_world,  # 4 X 4
            "T_cano_tobs_m1_world": T_cano_tobs_m1_world,  # 4 X 4
            "extrinsics": extrinsics,  # T X 4 X 4
            "intrinsics": intrinsics,  # 3 X 3
            "num_out_of_view": num_out_of_view,  # 2
            "start_frame": start_frame,
            "take_uid": take_uid,
            "take_name": take_name,
        }
        return hand_data

    def __len__(self):
        return len(self.data)
