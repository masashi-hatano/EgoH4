import logging

import torch

from models.transformer_cond_diffusion_model import CondGaussianDiffusion

logger = logging.getLogger(__name__)


def get_model(cfg, ckpt_pth=None):
    print(f"Creating model: {cfg.trainer.model}")
    if cfg.trainer.model == "egoh4":
        model = CondGaussianDiffusion(
            d_feats=cfg.trainer.d_feats,
            d_cond=cfg.trainer.d_cond,
            d_model=cfg.trainer.embed_dim,
            n_dec_layers=cfg.trainer.num_layer,
            n_head=cfg.trainer.nhead,
            d_k=cfg.trainer.embed_dim // cfg.trainer.nhead,
            d_v=cfg.trainer.embed_dim // cfg.trainer.nhead,
            max_timesteps=cfg.trainer.max_timesteps+1,
            out_dim=cfg.trainer.d_feats,
            timesteps=1000,
            objective='pred_x0',
            loss_type='l1',
            jpos_min=cfg.data_module.train.min_max[0],
            jpos_max=cfg.data_module.train.min_max[1],
        )
        if ckpt_pth is not None:
            p = torch.load(ckpt_pth)
            od = p["module"]
            pretrained_dict = {}
            for k, v in od.items():
                if ".model." in k:
                    k = k.replace("_forward_module.model.", "")
                    pretrained_dict[k] = v
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Model is initialized from {ckpt_pth}")
    return model
