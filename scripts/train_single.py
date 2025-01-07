import os
import sys

sys.path.append('../')

import gc
import logging
import random
import sys
import time

import numpy as np
import torch
import wandb
from utils import (worker_init_fn, load_data_adni, load_data_oasis, load_data_ixi)
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import DiceMetric, PSNRMetric, SSIMMetric
from models import Warp, build_loss, build_metrics, build_registration_head, build_flow_estimator
from run_iter_single import run_iter

torch.backends.cudnn.deterministic = True


def train(train_cfg_file: str, random_seed: int = 42):
    cfg = Config.fromfile(train_cfg_file)
    cfg.update(dict(random_seed=random_seed))
    wandb.init(project=cfg.project, name=cfg.name, config=dict(cfg))

    # define output directory
    model_dir = os.path.join(cfg.out_path, cfg.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.out_path, 'train_configs.py'))
    cfg.update(dict(amp_dtype=getattr(torch, cfg.amp_dtype)))

    # load model
    model = build_flow_estimator(cfg.model_cfg)
    if cfg.load_model:
        model.load_state_dict(
            torch.load(cfg.load_model, map_location=torch.device(cfg.device)))
    model.to(cfg.device)
    wandb.watch(model, log='gradients', log_freq=100, log_graph=True)

    # build registration head module
    reg_head = build_registration_head(cfg.registration_cfg)
    reg_head.to(cfg.device)

    # load data
    datasets = [
        load_data_oasis(cfg.oasis_cfg,
                        cache_rate=cfg.oasis_cfg.cache_rate,
                        num_workers=cfg.oasis_cfg.num_workers,
                        ),
        load_data_adni(cfg.adni_cfg,
                       cache_rate=cfg.adni_cfg.cache_rate,
                       num_workers=cfg.adni_cfg.num_workers,
                       ),
        load_data_ixi(cfg.ixi_cfg,
                      cache_rate=cfg.ixi_cfg.cache_rate,
                      num_workers=cfg.ixi_cfg.num_workers, )
    ]
    dataloaders = [DataLoader(dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              drop_last=True,
                              worker_init_fn=worker_init_fn
                              )
                   for dataset in datasets]
    len_dataloader = sum(len(d) for d in dataloaders)
    max_len = max(len(d) for d in dataloaders)

    # define loss
    sim_loss = build_loss(cfg.sim_loss_cfg).to(cfg.device)
    reg_loss = build_loss(cfg.reg_loss_cfg).to(cfg.device)
    dice_loss = build_loss(cfg.dice_loss_cfg) if cfg.dice_loss_cfg else None
    loss_funcs = dict(sim=sim_loss, reg=reg_loss, dice=dice_loss)

    # define metric
    compute_dice = DiceMetric(include_background=False, reduction='mean')
    compute_ssim = SSIMMetric(data_range=torch.tensor(1.0),
                              spatial_dims=3)._compute_metric
    compute_psnr = PSNRMetric(max_val=1.0)._compute_metric
    compute_jacdet = build_metrics(dict(type='sdlogjac'))
    metric_funcs = dict(dice=compute_dice,
                        ssim=compute_ssim,
                        psnr=compute_psnr,
                        jacdet=compute_jacdet)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=0,
                                 amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                          gamma=cfg.lr_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # epoch loop
    for epoch in range(cfg.start_epoch, cfg.max_epochs):
        phase = 'train'
        model.train()
        run_iter(cfg,
                 model,
                 reg_head,
                 dataloaders,
                 loss_funcs,
                 metric_funcs,
                 scaler,
                 optimizer,
                 len_dataloader * epoch,
                 max_len,
                 phase)
        lr_scheduler.step()

        if epoch % cfg.save_interval == 0 and epoch != cfg.start_epoch:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, '%04d.pth' % epoch))

    torch.save(model.state_dict(),
               os.path.join(model_dir, '%04d.pth' % cfg.max_epochs))

    max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print("[+] Maximum memory:\t{:.2f}GB".format(max_mem_mb))
    max_mem_re = torch.cuda.max_memory_reserved() / (1024 ** 3)
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
