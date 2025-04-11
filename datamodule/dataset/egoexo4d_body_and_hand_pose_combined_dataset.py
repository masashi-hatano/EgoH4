import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms as transforms_3d
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from datamodule.dataset.egoexo4d_hand_body_pose_dataset import \
    EgoExo4DHandBodyPoseDataset
from datamodule.utils.transform import ToTensor


class EgoExo4DBodyHandPoseCombinedDataset(Dataset):
    def __init__(self, cfg, root, split="train"):
        self.cfg = cfg
        self.root = Path(root)
        self.gt_output_dir = Path(root, "gt_output")
        self.split = split
        self.undist_img_dim = (512, 512)
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
        self.frame_num = 0
        self.statistics = {}
        self.statistics_right = {}
        self.statistics_left = {}

        self._load_body_data()
        self._load_hand_data()

    def _load_body_data(self):
        # Load all takes metadata
        takes = json.load(open(Path(self.root, "takes.json")))

        # Load GT annotation
        gt_anno_path = Path(
            self.gt_output_dir,
            "annotation",
            "manual",
            f"ego_body_pose_gt_anno_{self.split}.json",
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
                    take_uid,
                )
                self.start_frames.append(seq_frames[0])
                self.data.append(seq_data)
                self.poses_takes_uids.append(take_uid)
                self.take_names.append(take_name)

        print("Dataset lenght: {}".format(len(self.data)))
        print("Split: {}".format(self.split))
        print(self.min_jpos)
        print(self.max_jpos)

    def _load_hand_data(self):
        self.hand_dataset = EgoExo4DHandBodyPoseDataset(self.cfg, self.root, split=self.split)

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
        body_2d_pose_seq = []
        body_valid_3d_seq = []
        body_valid_2d_seq = []
        out_of_view_seq = []
        out_of_view_valid_seq = []

        assert len(seq_frames) == 30

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

            # body 2d pose
            body_2d = np.array(current_anno["body_2d"]) / self.cfg.img_size  # 17 joints X 2 dimensions for body
            body_2d_pose_seq.append(body_2d)

            # validity of hand 2d pose
            body_valid_2d = np.array(current_anno["body_valid_2d"])
            body_valid_2d_seq.append(body_valid_2d)

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
            "body_2d_pose": body_2d_pose_seq,
            "out_of_view": out_of_view_seq,
            "body_valid_3d": body_valid_3d_seq,
            "body_valid_2d": body_valid_2d_seq,
            "out_of_view_valid": out_of_view_valid_seq,
            "take_uid": take_uid,
            "extrinsics": extrinsics_seq,
            "intrinsics": anno["intrinsics"],
            "T_cano_world": T_cano_t0_world
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

    def _get_body_data(self, seq_data, take_uid, take_name, start_frame):
        # 3d body pose
        body_3d_pose = torch.Tensor(np.array(seq_data["body_3d_pose"]))  # T X 17 X 3
        body_valid_3d = torch.Tensor(np.array(seq_data["body_valid_3d"]))  # T X 17

        # 3d hand pose dummy
        hand_3d_pose = torch.zeros((self.window_size, 42, 3))
        hand_valid_3d = torch.zeros((self.window_size, 42))

        # 2d hand pose dummy
        hand_2d_pose = torch.zeros((self.window_size, 42, 2))
        hand_valid_2d = torch.zeros((self.window_size, 42))

        # 2d body pose
        body_2d_pose = torch.Tensor(np.array(seq_data["body_2d_pose"]))  # T X 17 X 2
        body_valid_2d = torch.Tensor(np.array(seq_data["body_valid_2d"]))  # T X 17

        # use the wrist annotations from 2d body pose
        hand_3d_pose[:, 21, :] = body_3d_pose[:, 9, :]  # T X 3 (left)
        hand_3d_pose[:, 0, :] = body_3d_pose[:, 10, :]  # T X 3 (right)
        hand_valid_3d[:, 21] = body_valid_3d[:, 9]  # T (left)
        hand_valid_3d[:, 0] = body_valid_3d[:, 10]  # T (right)
        hand_2d_pose[:, 21, :] = body_2d_pose[:, 9, :]  # T X 2 (left)
        hand_2d_pose[:, 0, :] = body_2d_pose[:, 10, :]  # T X 2 (right)
        hand_valid_2d[:, 21] = body_valid_2d[:, 9]  # T (left)
        hand_valid_2d[:, 0] = body_valid_2d[:, 10]  # T (right)

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
        assert cond_head.shape == (self.window_size, 9)
        extrinsics = torch.Tensor(np.array(seq_data["extrinsics"]))  # T X 4 X 4
        intrinsics = torch.Tensor(np.array(seq_data["intrinsics"]))  # 3 X 3

        # filter out the invalid poses
        body_3d_pose[body_valid_3d == 0] = -1
        hand_3d_pose[hand_valid_3d == 0] = -1
        hand_2d_pose[hand_valid_2d == 0] = -1

        # load RGB imgs features
        imgs = []
        for i in range(20):
            frame_number = int(start_frame) + self.interval * i
            img_path = Path(self.gt_output_dir, "imgs_feat", self.split, take_name, f"{frame_number:06d}.pt")
            img = torch.load(img_path).flatten()
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)  # T X D

        body_data = {
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
            "extrinsics": extrinsics,  # T X 4 X 4
            "intrinsics": intrinsics,  # 3 X 3
            "start_frame": start_frame,
            "take_uid": take_uid,
            "take_name": take_name,
        }
        return body_data

    def _get_hand_data(self, seq_data, take_uid, take_name, start_frame):
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
        assert cond_head.shape == (self.window_size, 9)
        extrinsics = torch.Tensor(np.array(seq_data["extrinsics"]))  # T X 4 X 4
        intrinsics = torch.Tensor(np.array(seq_data["intrinsics"]))  # 3 X 3

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
            "extrinsics": extrinsics,  # T X 4 X 4
            "intrinsics": intrinsics,  # 3 X 3
            "start_frame": start_frame,
            "take_uid": take_uid,
            "take_name": take_name,
        }
        return hand_data

    def __getitem__(self, index):
        # manual body data
        take_uid = self.poses_takes_uids[index]
        start_frame = self.start_frames[index]
        take_name = self.take_names[index]
        seq_data = self.data[index]

        body_data = self._get_body_data(seq_data, take_uid, take_name, start_frame)

        # manual hand & auto body data
        hand_index = index % len(self.hand_dataset)
        take_uid = self.hand_dataset.poses_takes_uids[hand_index]
        start_frame = self.hand_dataset.start_frames[hand_index]
        take_name = self.hand_dataset.take_names[hand_index]
        seq_data = self.hand_dataset.data[hand_index]

        hand_data = self._get_hand_data(seq_data, take_uid, take_name, start_frame)

        # combine two datasets: manual body dataset & manual hand dataset with auto body
        cond_head = torch.cat([body_data["cond_head"].unsqueeze(0), hand_data["cond_head"].unsqueeze(0)], dim=0)  # 2 X T X 9
        imgs = torch.cat([body_data["imgs"].unsqueeze(0), hand_data["imgs"].unsqueeze(0)], dim=0)  # 2 X T X D
        body_3d = torch.cat([body_data["body_3d"].unsqueeze(0), hand_data["body_3d"].unsqueeze(0)], dim=0)  # 2 X T X 17 X 3
        body_valid_3d = torch.cat([body_data["body_valid_3d"].unsqueeze(0), hand_data["body_valid_3d"].unsqueeze(0)], dim=0)  # 2 X T X 17
        hand_3d = torch.cat([body_data["hand_3d"].unsqueeze(0), hand_data["hand_3d"].unsqueeze(0)], dim=0)  # 2 X T X 40 X 3
        hand_valid_3d = torch.cat([body_data["hand_valid_3d"].unsqueeze(0), hand_data["hand_valid_3d"].unsqueeze(0)], dim=0)  # 2 X T X 40
        hand_2d = torch.cat([body_data["hand_2d"].unsqueeze(0), hand_data["hand_2d"].unsqueeze(0)], dim=0)  # 2 X T X 40 X 2
        hand_valid_2d = torch.cat([body_data["hand_valid_2d"].unsqueeze(0), hand_data["hand_valid_2d"].unsqueeze(0)], dim=0)  # 2 X T X 40
        out_of_view = torch.cat([body_data["out_of_view"].unsqueeze(0), hand_data["out_of_view"].unsqueeze(0)], dim=0)  # 2 X T X 2
        out_of_view_valid = torch.cat([body_data["out_of_view_valid"].unsqueeze(0), hand_data["out_of_view_valid"].unsqueeze(0)], dim=0)  # 2 X T X 2
        T_cano_t0_world = torch.cat([body_data["T_cano_t0_world"].unsqueeze(0), hand_data["T_cano_t0_world"].unsqueeze(0)], dim=0)  # 2 X 4 X 4
        extrinsics = torch.cat([body_data["extrinsics"].unsqueeze(0), hand_data["extrinsics"].unsqueeze(0)], dim=0)  # 2 X T X 4 X 4
        intrinsics = torch.cat([body_data["intrinsics"].unsqueeze(0), hand_data["intrinsics"].unsqueeze(0)], dim=0)  # 2 X 3 X 3
        start_frame = [body_data["start_frame"], hand_data["start_frame"]]
        take_uid = [body_data["take_uid"], hand_data["take_uid"]]
        take_name = [body_data["take_name"], hand_data["take_name"]]

        return {
            "cond_head": cond_head,  # 2 X T X 9
            "imgs": imgs,  # 2 X T X D
            "body_3d": body_3d,  # 2 X T X 17 X 3
            "body_valid_3d": body_valid_3d,  # 2 X T X 17
            "hand_3d": hand_3d,  # 2 X T X 42 X 3
            "hand_valid_3d": hand_valid_3d,  # 2 X T X 42
            "hand_2d": hand_2d,  # 2 X T X 42 X 2
            "hand_valid_2d": hand_valid_2d,  # 2 X T X 42
            "out_of_view": out_of_view,  # 2 X T X 2
            "out_of_view_valid": out_of_view_valid,  # 2 X T X 2
            "T_cano_t0_world": T_cano_t0_world,  # 2 X 4 X 4
            "extrinsics": extrinsics,  # 2 X T X 4 X 4
            "intrinsics": intrinsics,  # 2 X 3 X 3
            "start_frame": start_frame,
            "take_uid": take_uid,
            "take_name": take_name,
        }

    def __len__(self):
        return len(self.data)
