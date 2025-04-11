import math
from inspect import isfunction

import torch
import torch.nn.functional as F
# from data.amass_diffusion_dataset import quat_ik_torch
from einops import rearrange, reduce
from torch import nn
from tqdm.auto import tqdm

from models.transformer_module import Decoder


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_feats,
        d_cond,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()

        self.d_feats = d_feats
        self.d_cond = d_cond
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k
        self.d_v = d_v
        self.max_timesteps = max_timesteps

        # Input: BS X D X T
        # Output: BS X T X D'
        self.motion_transformer = Decoder(
            d_feats=self.d_feats,
            d_cond=self.d_cond,
            d_model=self.d_model,
            n_layers=self.n_dec_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            max_timesteps=self.max_timesteps,
            use_full_attention=True,
        )

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # visiblity output
        self.vis_out = nn.Linear(self.d_model, 2)

        # For noise level t embedding
        dim = 64
        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model),
        )

    def forward(self, src, imgs_feat, noise_t, padding_mask=None):
        # src: BS X T X D
        # imgs_feat: BS X T X C
        # noise_t: int

        noise_t_embed = self.time_mlp(noise_t)  # BS X d_model
        noise_t_embed = noise_t_embed[:, None, :]  # BS X 1 X d_model

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            # In training, no need for masking
            padding_mask = (
                torch.ones(bs, 1, num_steps).to(src.device).bool()
            )  # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps) + 1  # timesteps
        pos_vec = (
            pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1)
        )  # BS X 1 X timesteps

        data_input = src.transpose(1, 2).detach()  # BS X D X T
        feat_pred, _ = self.motion_transformer(
            data_input, imgs_feat, padding_mask, pos_vec, obj_embedding=noise_t_embed
        )

        output = self.linear_out(feat_pred[:, 1:])  # BS X T X D
        vis_e = self.vis_out(feat_pred[:, 1:])  # BS X T X 2

        return output, vis_e


class CondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        d_feats,
        d_cond,
        d_model,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps=1000,
        loss_type="l1",
        objective="pred_noise",
        beta_schedule="cosine",
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        jpos_min=None,
        jpos_max=None,
    ):
        super().__init__()

        self.jpos_min = jpos_min
        self.jpos_max = jpos_max

        self.denoise_fn = TransformerDiffusionModel(
            d_feats=d_feats,
            d_cond=d_cond,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            n_dec_layers=n_dec_layers,
            max_timesteps=max_timesteps,
        )
        # Input condition and noisy motion, noise level t, predict gt motion

        self.objective = objective

        self.seq_len = max_timesteps - 1
        self.out_dim = out_dim

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, imgs_feat, clip_denoised):
        x_all = torch.cat((x, x_cond), dim=-1)
        model_output, vis_e = self.denoise_fn(x_all, imgs_feat, t)

        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == "pred_x0":
            x_start = model_output
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, imgs_feat, clip_denoised=True):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            x_cond=x_cond,
            imgs_feat=imgs_feat,
            clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, imgs_feat):
        # x_start: BS X T X D (D = J X 3)
        # x_cond: BS X T X Dc
        # imgs_feat: BS X T X C (C = 512)
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            x = self.p_sample(
                x,
                torch.full((b,), i, device=device, dtype=torch.long),
                x_cond,
                imgs_feat,
            )

        return x  # BS X T X D

    @torch.no_grad()
    def sample(self, x_start, cond, imgs_feat):
        # naive conditional sampling by replacing the noisy prediction with input target data.
        self.denoise_fn.eval()
        sample_res = self.p_sample_loop(x_start.shape, cond, imgs_feat)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, x_start, x_cond, imgs_feat, t, target_2d, valid_2d, extrinsics, intrinsics, noise=None, visible=None, out_of_view=None, out_of_view_valid=None):
        # x_start: BS X T X D (D = J X 3)
        # x_cond: BS X T X Dc
        # imgs_feat: BS X T X C
        # visible: BS X T X J
        output = {}
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(
            x_start=x_start, t=t, noise=noise
        )  # noisy motion in noise level t.

        x_all = torch.cat((x, x_cond), dim=-1)  # BS X T X (D + Dc)
        pred, vis_e = self.denoise_fn(x_all, imgs_feat, t)  # BS X T X D

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # Predicting joints' pose.
        # BS X T X J X C
        pred = rearrange(
            pred, "b t (j c) -> b t j c", j=57, c=3
        )
        target = rearrange(
            target, "b t (j c) -> b t j c", j=57, c=3
        )

        # visiblitiy
        vis_e = rearrange(
            vis_e, "b t (j c) -> b t j c", j=2, c=1
        )

        # 3D joint loss
        loss_3d_jpos = self.loss_fn(pred[:, :, :, :], target[:, :, :, :], reduction="none")
        if visible is not None:
            loss_3d_jpos = loss_3d_jpos * visible[:, :, :, None]
        loss_3d_jpos_body_obs = loss_3d_jpos[:, :20, :15, :]
        loss_3d_jpos_hand_obs = loss_3d_jpos[:, :20, 15:, :]
        loss_3d_jpos_body_fut = loss_3d_jpos[:, 20:, :15, :]
        loss_3d_jpos_hand_fut = loss_3d_jpos[:, 20:, 15:, :]
        loss_3d_jpos_body_obs = reduce(loss_3d_jpos_body_obs, "b ... -> b (...)", "mean")
        loss_3d_jpos_hand_obs = reduce(loss_3d_jpos_hand_obs, "b ... -> b (...)", "mean")
        loss_3d_jpos_body_fut = reduce(loss_3d_jpos_body_fut, "b ... -> b (...)", "mean")
        loss_3d_jpos_hand_fut = reduce(loss_3d_jpos_hand_fut, "b ... -> b (...)", "mean")
        loss_3d_jpos_body_obs = loss_3d_jpos_body_obs * extract(self.p2_loss_weight, t, loss_3d_jpos_body_obs.shape)
        loss_3d_jpos_hand_obs = loss_3d_jpos_hand_obs * extract(self.p2_loss_weight, t, loss_3d_jpos_hand_obs.shape)
        loss_3d_jpos_body_fut = loss_3d_jpos_body_fut * extract(self.p2_loss_weight, t, loss_3d_jpos_body_fut.shape)
        loss_3d_jpos_hand_fut = loss_3d_jpos_hand_fut * extract(self.p2_loss_weight, t, loss_3d_jpos_hand_fut.shape)

        # append
        output["loss_3d_jpos_body_obs"] = loss_3d_jpos_body_obs[loss_3d_jpos_body_obs != 0].mean()
        output["loss_3d_jpos_hand_obs"] = loss_3d_jpos_hand_obs[loss_3d_jpos_hand_obs != 0].mean()
        output["loss_3d_jpos_body_fut"] = loss_3d_jpos_body_fut[loss_3d_jpos_body_fut != 0].mean()
        output["loss_3d_jpos_hand_fut"] = loss_3d_jpos_hand_fut[loss_3d_jpos_hand_fut != 0].mean()

        # Visiblity loss
        loss_vis = self.balanced_ce_loss(vis_e[:, :20, :, :], out_of_view[:, :20, :], out_of_view_valid[:, :20, :])
        loss_vis = loss_vis * extract(self.p2_loss_weight, t, loss_vis.shape)

        output["loss_vis"] = loss_vis.mean()

        # 2D reprojection loss
        if target_2d is not None:
            loss_repro = self.loss_reprojection(pred[:, :20, :, :], target_2d[:, :20, :, :], extrinsics[:, :20, :, :], intrinsics)
            loss_repro = loss_repro * valid_2d[:, :20, :, None]
            loss_repro = reduce(loss_repro, "b ... -> b (...)", "mean")
            loss_repro = loss_repro * extract(self.p2_loss_weight, t, loss_repro.shape)
            if torch.isnan(loss_repro).any():
                print("loss_repro nan")
                output["loss_repro"] = torch.tensor(0.0).to(loss_repro.device)
                return output
            output["loss_repro"] = loss_repro.mean()
            return output
        else :
            output["loss_repro"] = torch.tensor(0.0).to(loss_3d_jpos_body_obs.device)
            return output

    def loss_reprojection(self, pred, target_2d, extrinsics, intrinsics):
        # pred: BS X T X 19 X 3
        # target: BS X T X J X 2
        # extrinsics: BS X T X 4 X 4
        # intrinsics: BS X 3 X 3
        # return: BS X T X 19

        BS, T, _, _ = pred.shape

        # pred = pred[:, :, 15:57]
        pred = torch.cat((pred[:, :, 15:16], pred[:, :, 36:37]), dim=2) # BS X T X 2 X 3
        J = pred.shape[2]

        # denormalize
        pred = self.de_normalize_min_max(pred)

        # transform from canonical to world coordinates
        # pred: BS X T X 2 X 3 -> BS X T X 2 X 4 -> (BS*T*2) X 4
        pred = torch.cat((pred, torch.ones(pred.shape[0], pred.shape[1], pred.shape[2], 1).to(pred.device)), dim=-1)
        pred = rearrange(pred, "b t j c -> (b t j) c", b=BS, t=T, j=J, c=4)
        pred = pred.unsqueeze(-1) # (BS*T*2) X 4 X 1
        # canonical to world: BS X 4 X 4 -> (BS*T*2) X 4 X 4
        T_cano_t0_world = self.T_cano_t0_world.unsqueeze(1).unsqueeze(1).repeat(1, T, J, 1, 1)
        T_cano_t0_world = rearrange(T_cano_t0_world, "b t j c1 c2 -> (b t j) c1 c2", b=BS, t=T, j=J)
        # Perform batched matrix multiplication
        pred = torch.bmm(torch.linalg.inv(T_cano_t0_world), pred)  # Result shape is [(BS*T*2), 4, 1]

        # Extrinsics: world to camera
        extrinsics = extrinsics.unsqueeze(2).repeat(1, 1, J, 1, 1)
        extrinsics = rearrange(extrinsics, "b t j c1 c2 -> (b t j) c1 c2", b=BS, t=T, j=J)
        # Perform batched matrix multiplication
        pred_camera_3d = torch.bmm(extrinsics, pred)  # Result shape is [(BS*T*2), 4, 1]
        pred_camera_3d = pred_camera_3d[:, :3, :]  # (BS*T*2) X 3 X 1

        # Intrinsics: camera to image coordinates
        # intrinsics: BS X 3 X 3 -> (BS*T*2) X 3 X 3
        intrinsics = intrinsics.unsqueeze(1).unsqueeze(2).repeat(1, T, J, 1, 1)
        intrinsics = rearrange(intrinsics, "b t j c1 c2 -> (b t j) c1 c2", b=BS, t=T, j=J)
        # Perform batched matrix multiplication
        pred_image_2d = torch.bmm(intrinsics, pred_camera_3d)  # Result shape is [(BS*T*2), 3, 1]
        scale = pred_image_2d[:, 2].unsqueeze(1)  # (BS*T*2) X 1 X 1
        pred_image_2d = pred_image_2d / scale  # scale to 1
        pred_image_2d = pred_image_2d.squeeze(-1)[:, :2]  # (BS*T*2) X 2
        # rotate from landscape to portrait view
        pred_image_2d = self.aria_landscape_to_portrait(pred_image_2d, img_shape=(512, 512))
        pred_image_2d = rearrange(pred_image_2d, "(b t j) c -> b t j c", b=BS, t=T, j=J)
        pred_image_2d /= 512.0
        loss_repro = F.l1_loss(pred_image_2d, target_2d, reduction="none")

        # filtering
        filter = torch.where(scale < 0.1, 0., 1.)
        filter = torch.where(scale > 1.5, 0., filter)
        filter = rearrange(filter, "(b t j) c1 c2 -> b t j (c1 c2)", b=BS, t=T, j=J, c1=1, c2=1)
        loss_repro = loss_repro * filter

        return loss_repro

    def de_normalize_min_max(self, jpos):
        jpos = (jpos + 1) * 0.5  # [0, 1] range
        de_jpos = jpos * (self.jpos_max - self.jpos_min) + self.jpos_min
        return de_jpos  # B X T X 17 X 3

    def balanced_ce_loss(self, pred, gt, valid):
        # pred: BS X T X 2 X 1
        # gt: BS X T X 2
        # valid: BS X T X 2

        gt = gt.unsqueeze(-1)  # BS X T X 2 X 1

        # pred and gt are the same shape
        for (a, b) in zip(pred.size(), gt.size()):
            assert (a == b)  # some shape mismatch!

        pos = (gt > 0.95).float()
        neg = (gt < 0.05).float()

        label = pos * 2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        pos_loss = self.reduce_masked_mean(loss, pos)
        neg_loss = self.reduce_masked_mean(loss, neg)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss

    def reduce_masked_mean(self, x, mask):
        # x and mask are the same shape, or at least broadcastably so < actually it's safer if you disallow broadcasting
        # returns shape-1
        # axis can be a list of axes
        for (a, b) in zip(x.size(), mask.size()):
            # if not b==1:
            assert (a == b)  # some shape mismatch!
        prod = x * mask
        numer = reduce(prod, "b ... -> b (...)", "sum")
        denom = 1e-6 + reduce(mask, "b ... -> b (...)", "sum")

        mean = numer / denom
        return mean

    def aria_landscape_to_portrait(self, kpts, img_shape=(512, 512)):
        """
        Rotate kpts coordinates from landscape view to portrait view
        img_shape is the shape of landscape image
        """
        H, _ = img_shape
        new_kpts = kpts.clone()
        new_kpts[:, 0] = H - kpts[:, 1] - 1
        new_kpts[:, 1] = kpts[:, 0]
        return new_kpts

    def forward(self, x_start, cond, imgs_feat, visible, T_cano_t0_world, target_2d=None, valid_2d=None, out_of_view=None, out_of_view_valid=None, extrinsics=None, intrinsics=None):
        # x_start: BS X T X D
        # cond: BS X T X Dc
        # imgs_feat: BS X T X C or None
        # visible: BS X T X J
        # T_cano_t0_world: BS X 4 X 4
        # target_2d: BS X T X J X 2
        # valid_2d: BS X T X J
        # out_of_view: BS X T X 2
        # out_of_view_valid: BS X T X 2
        # extrinsics: BS X T X 4 X 4
        # intrinsics: BS X 3 X 3
        bs = x_start.shape[0]
        self.T_cano_t0_world = T_cano_t0_world
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()
        loss_output = self.p_losses(x_start, cond, imgs_feat, t, target_2d, valid_2d, extrinsics, intrinsics, None, visible, out_of_view, out_of_view_valid)

        return loss_output
