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
from torch.utils.data import ConcatDataset
import wandb
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import DiceMetric, PSNRMetric, SSIMMetric
from utils import (load_data_AbdomenCT, load_data_AbdCTCT,
                    load_data_AmosMR, load_data_AbdomenMR,
                    load_data_AbdomenMRCT_pair, worker_init_fn,
                    ABDOMENMRCT_SPLITS)
from models import Warp, build_loss, build_metrics, build_registration_head, build_flow_estimator
from run_iter import run_iter as run_iter_unpaired
from run_iter_pair import run_iter as run_iter_paired
from prealign import RigidPreAlign, TranslationPreAlign

torch.backends.cudnn.deterministic = True


def train(config_file: str,
          random_seed: int = 42,
          use_rigid: bool = False,
          split: str = 'ts'):
    cfg = Config.fromfile(config_file)
    cfg.update(dict(random_seed=random_seed, split=split))
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

    if cfg.get("scale_pyramid", None) is None:
        cfg.update(dict(pyramid=False))
    else:
        cfg.update(dict(pyramid=True))
    reg_head = []
    if cfg.pyramid:
        for scale in cfg.scale_pyramid:
            reg_head_down = build_registration_head(
                dict(
                    type='DownSizeRegistrationHead',
                    image_size=cfg.image_size,
                    scale=scale,
                    interp_mode='bilinear',
                )
            )
            reg_head_down.to(cfg.device)
            reg_head.append(reg_head_down)
    reg_head.append(build_registration_head(cfg.registration_cfg).to(cfg.device))

    if cfg.get('pairwise', False):
        run_iter = run_iter_paired
        print(f'[train] AbdomenMRCT split={split} '
              f'(indices {ABDOMENMRCT_SPLITS[split]["indices"][0]}-'
              f'{ABDOMENMRCT_SPLITS[split]["indices"][-1]}); '
              f'evaluate.py must be given the other half.')
        dataset = load_data_AbdomenMRCT_pair(cfg.pair_cfg,
                                             split=split,
                                             cache_rate=cfg.cache_rate,
                                             num_workers=cfg.num_workers)
        dataloader = DataLoader(dataset,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                worker_init_fn=worker_init_fn)
        print(len(dataset))
        max_iters = len(dataset) * cfg.max_epochs
        dataloaders = [dataloader]
    else:
        run_iter = run_iter_unpaired
        # load data
        ct_dataset = load_data_AbdomenCT(cfg.ct_cfg,
                                        cache_rate=cfg.cache_rate,
                                        num_workers=cfg.num_workers)
        ctct_dataset = load_data_AbdCTCT(cfg.ctct_cfg,
                                        cache_rate=cfg.cache_rate,
                                        num_workers=cfg.num_workers)
        ct_dataset = ConcatDataset([ct_dataset, ctct_dataset])
        abd_mr_dataset = load_data_AbdomenMR(cfg.abd_mr_cfg,
                                            cache_rate=cfg.cache_rate,
                                            num_workers=cfg.num_workers)
        amos_mr_dataset = load_data_AmosMR(cfg.amos_mr_cfg,
                                        cache_rate=cfg.cache_rate,
                                        num_workers=cfg.num_workers)
        mr_dataset = ConcatDataset([abd_mr_dataset, amos_mr_dataset])
        ct_dataloader = DataLoader(ct_dataset,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                worker_init_fn=worker_init_fn)
        mr_dataloader = DataLoader(mr_dataset,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                worker_init_fn=worker_init_fn)
        print(len(ct_dataset), len(mr_dataset))
        len_dataset = max(len(ct_dataset), len(mr_dataset))
        max_iters = len_dataset * cfg.max_epochs
        dataloaders = (ct_dataloader, mr_dataloader)

    # define loss
    sim_loss = build_loss(cfg.sim_loss_cfg).to(cfg.device)
    reg_loss = build_loss(cfg.reg_loss_cfg).to(cfg.device)
    gradicon_loss = build_loss(cfg.gradicon_loss_cfg).to(cfg.device)
    dice_loss = build_loss(cfg.dice_loss_cfg)
    loss_funcs = dict(sim=sim_loss,
                      dice=dice_loss,
                      reg=reg_loss,
                      gradicon=gradicon_loss, )

    # define metric
    # labels don't have background
    compute_dice = DiceMetric(include_background=True, get_not_nans=True, reduction='mean')
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

    # pre-alignment; translation is what the paper uses, --rigid is the alternative
    if use_rigid:
        prealigner = RigidPreAlign(compute_dice=compute_dice,
                                   device=cfg.device)
    else:
        prealigner = TranslationPreAlign(compute_dice=compute_dice,
                                         device=cfg.device,
                                         randomize=False)

    # epoch loop
    current_iter = 0
    for epoch in range(cfg.start_epoch, cfg.max_epochs):
        phase = 'train'
        model.train()
        current_iter = run_iter(cfg,
                 model,
                 reg_head,
                 dataloaders,
                 prealigner,
                 loss_funcs,
                 metric_funcs,
                 scaler,
                 optimizer,
                 current_iter,
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
    p.add_argument('--rigid',
                   action='store_true',
                   help='use rigid pre-alignment instead of translation-only '
                        '(the paper uses translation)')
    p.add_argument('--split', choices=['ts', 'tr'], default='ts',
                   help="Which half of AbdomenMRCT to use. "
                        "'ts' = indices 9-16 (imagesTs/TSlabelsTs, labels [5,2,3,1]); "
                        "'tr' = indices 1-8 (imagesTr/labelsTr, labels [1,2,3,4]). "
                        "train.py and evaluate.py must be given opposite values.")
    args = p.parse_args()
    set_seed(args.random_seed)
    train(args.train_config, args.random_seed, args.rigid, args.split)
