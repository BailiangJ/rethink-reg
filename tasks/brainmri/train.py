import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import gc
import logging
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import wandb
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import DiceHelper, PSNRMetric, SSIMMetric
from models import (
    Warp,
    build_loss,
    build_metrics,
    build_registration_head,
    build_flow_estimator,
)
from run_iter import run_iter as train_iter
from val_iter import run_iter as val_iter
from utils import load_data_LUMIR, load_valdata_LUMIR, worker_init_fn

torch.backends.cudnn.deterministic = True
import types

def find_modules_in_config(cfg, prefix=''):
    """Recursively find module references in config."""
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, types.ModuleType):
            print(f"Found module at: {full_key} = {value}")
        elif isinstance(value, dict):
            find_modules_in_config(value, full_key)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, types.ModuleType):
                    print(f"Found module at: {full_key}[{i}] = {item}")
                elif isinstance(item, dict):
                    find_modules_in_config(item, f"{full_key}[{i}]")


def train(config_file: str, random_seed: int = 42):
    cfg = Config.fromfile(config_file)
    cfg.update(dict(random_seed=random_seed))
    # Debug before wandb.init
    find_modules_in_config(cfg)
    wandb.init(project=cfg.project, name=cfg.name, config=dict(cfg))

    # define output directory
    model_dir = os.path.join(cfg.out_path, cfg.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.out_path, "train_configs.py"))
    cfg.update(dict(amp_dtype=getattr(torch, cfg.amp_dtype)))

    # load model
    model = build_flow_estimator(cfg.model_cfg)

    if cfg.load_model:
        model.load_state_dict(
            torch.load(cfg.load_model, map_location=torch.device(cfg.device)),
            strict=False
        )

    model.to(cfg.device)
    wandb.watch(model, log="gradients", log_freq=100, log_graph=True)

    if cfg.get("scale_pyramid", None) is None:
        cfg.update(dict(pyramid=False))
    else:
        cfg.update(dict(pyramid=True))
    reg_head = []
    if cfg.pyramid:
        for scale in cfg.scale_pyramid:
            reg_head_down = build_registration_head(
                dict(
                    type="DownSizeRegistrationHead",
                    image_size=cfg.image_size,
                    scale=scale,
                    interp_mode="bilinear",
                )
            )
            reg_head_down.to(cfg.device)
            reg_head.append(reg_head_down)
    reg_head.append(build_registration_head(cfg.registration_cfg).to(cfg.device))

    dataset = load_data_LUMIR(
        cfg.lumir_cfg,
        cache_rate=cfg.lumir_cfg.cache_rate,
        num_workers=cfg.lumir_cfg.num_workers,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    len_trainset = len(dataloader)

    val_dataset = load_valdata_LUMIR(
        cfg.lumir_val_cfg,
        cache_rate=cfg.lumir_val_cfg.cache_rate,
        num_workers=cfg.lumir_val_cfg.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    len_valset = len(val_dataloader)

    # define loss
    sim_loss = build_loss(cfg.sim_loss_cfg).to(cfg.device)
    reg_loss = build_loss(cfg.reg_loss_cfg).to(cfg.device)
    gradicon_loss = build_loss(cfg.gradicon_loss_cfg).to(cfg.device)
    dice_loss = build_loss(cfg.dice_loss_cfg)
    jacdet_loss = build_loss(cfg.np_jacdet_loss_cfg)
    loss_funcs = dict(
        sim=sim_loss,
        dice=dice_loss,
        reg=reg_loss,
        gradicon=gradicon_loss,
        jacdet=jacdet_loss,
    )

    # define metric
    compute_dice = DiceHelper(include_background=False, get_not_nans=True)
    compute_ssim = SSIMMetric(
        data_range=torch.tensor(1.0), spatial_dims=3
    )._compute_metric
    compute_psnr = PSNRMetric(max_val=1.0)._compute_metric
    compute_jacdet = build_metrics(dict(type="sdlogjac"))
    metric_funcs = dict(
        dice=compute_dice, ssim=compute_ssim, psnr=compute_psnr, jacdet=compute_jacdet
    )

    # optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=0, amsgrad=True
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    current_iter=0
    # epoch loop
    for epoch in range(cfg.start_epoch, cfg.max_epochs):
        if epoch % cfg.val_interval == 0:
            model.eval()
            with torch.no_grad():
                current_iter = val_iter(
                    cfg,
                    model,
                    reg_head,
                    val_dataloader,
                    metric_funcs,
                    current_iter,
                    phase='val',
                )

        model.train()
        train_iter(
            cfg,
            model,
            reg_head,
            dataloader,
            loss_funcs,
            metric_funcs,
            scaler,
            optimizer,
            current_iter=len_trainset*epoch,
            phase='train'
        )
        lr_scheduler.step()
        print("current_iter",len_trainset*epoch)

        if epoch % cfg.save_interval == 0 and epoch != cfg.start_epoch:
            torch.save(model.state_dict(), os.path.join(model_dir, "%04d.pth" % epoch))

    torch.save(model.state_dict(), os.path.join(model_dir, "%04d.pth" % cfg.max_epochs))

    max_mem_mb = torch.cuda.max_memory_allocated() / (1024**3)
    print("[+] Maximum memory:\t{:.2f}GB".format(max_mem_mb))
    max_mem_re = torch.cuda.max_memory_reserved() / (1024**3)
    print("[+] Maximum memory:\t{:.2f}GB".format(max_mem_re))


if __name__ == '__main__':
    import pathlib
    import configargparse
    from utils import set_seed

    p = configargparse.ArgParser()
    p.add_argument('--train-config',
                   required=True,
                   type=lambda f: pathlib.Path(f).absolute(),
                   help='path of train configure file')
    p.add_argument('--random-seed',
                   '-seed',
                   required=True,
                   type=int,
                   help='random seed')
    args = p.parse_args()
    set_seed(args.random_seed)
    train(args.train_config, args.random_seed)
