import json

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from timm.scheduler import CosineLRScheduler
from torch import optim as optim
from torch.optim.lr_scheduler import MultiStepLR


def create_optimizer(cfg, model):
    opt_lower = cfg.opt.lower()
    weight_decay = cfg.weight_decay
    # parameters = model.parameters()
    
    if type(model) == list:
        for m in model:
            parameters = [p for p in m.parameters() if p.requires_grad]
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]

    opt_args = dict(lr=cfg.lr, weight_decay=weight_decay)
    if hasattr(cfg, "opt_eps") and cfg.opt_eps is not None:
        opt_args["eps"] = cfg.opt_eps
    if hasattr(cfg, "opt_betas") and cfg.opt_betas is not None:
        opt_args["betas"] = cfg.opt_betas

    # print("optimizer settings:", opt_args)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "sgd" or opt_lower == "nesterov":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(
            parameters, momentum=cfg.momentum, nesterov=True, **opt_args
        )
    elif opt_lower == "momentum":
        opt_args.pop("eps", None)
        optimizer = optim.SGD(
            parameters, momentum=cfg.momentum, nesterov=False, **opt_args
        )
    elif opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "nadam":
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == "adamp":
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == "sgdp":
        optimizer = SGDP(parameters, momentum=cfg.momentum, nesterov=True, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "adafactor":
        if not cfg.lr:
            opt_args["lr"] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == "adahessian":
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == "rmsprop":
        optimizer = optim.RMSprop(
            parameters, alpha=0.9, momentum=cfg.momentum, **opt_args
        )
    elif opt_lower == "rmsproptf":
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=cfg.momentum, **opt_args)
    elif opt_lower == "nvnovograd":
        optimizer = NvNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == "lookahead":
            optimizer = Lookahead(optimizer)

    return optimizer


def get_optimizer(cfg, model, niter_per_epoch):
    optimizer = create_optimizer(cfg, model)
    if cfg.scheduler == 'cosine':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=cfg.epochs * niter_per_epoch,
            lr_min=cfg.min_lr,
            warmup_t=cfg.warmup_epochs * niter_per_epoch,
            warmup_lr_init=cfg.warmup_lr,
            warmup_prefix=True,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    else:
        scheduler = None

    return optimizer, scheduler
