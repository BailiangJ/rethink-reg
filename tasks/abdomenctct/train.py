import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import wandb
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import DiceMetric, PSNRMetric, SSIMMetric
from utils import load_data_PSMAReg_pair, worker_init_fn
from models import build_loss, build_metrics, build_registration_head, build_flow_estimator
from run_iter import run_iter

torch.backends.cudnn.deterministic = True


def _sanitize_wandb_config(value):
    if isinstance(value, ModuleType):
        return None
    if isinstance(value, Mapping):
        return {k: _sanitize_wandb_config(v) for k, v in value.items()
                if not isinstance(v, ModuleType)}
    if isinstance(value, (list, tuple)):
        return [_sanitize_wandb_config(v) for v in value]
    return value


def _resolve_out_path(config_file, out_path):
    if os.path.isabs(out_path):
        return out_path
    config_dir = os.path.dirname(os.path.abspath(config_file))
    task_dir = (os.path.dirname(config_dir)
                if os.path.basename(config_dir) in {'configs', 'half_res_configs'}
                else config_dir)
    return os.path.abspath(os.path.join(task_dir, out_path))


def train(config_file: str, random_seed: int = 42):
    cfg = Config.fromfile(config_file)
    cfg.update(dict(random_seed=random_seed,
                    out_path=_resolve_out_path(config_file, cfg.out_path)))
    os.makedirs(cfg.out_path, exist_ok=True)
    wandb_dir = cfg.get('wandb_dir', os.path.join(cfg.out_path, 'wandb'))
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(project=cfg.project,
               group=cfg.get('group', None),
               name=cfg.name,
               dir=wandb_dir,
               config=_sanitize_wandb_config(dict(cfg)))

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
        assert len(cfg.scale_pyramid) == len(cfg.scale_loss_weights)
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

    # load data
    dataset = load_data_PSMAReg_pair(cfg,
                                     split='train',
                                     cache_rate=cfg.cache_rate,
                                     num_workers=cfg.num_workers)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=cfg.get('dataloader_num_workers', 0),
                            worker_init_fn=worker_init_fn)
    len_dataset = len(dataset)

    # define loss
    sim_loss = build_loss(cfg.sim_loss_cfg).to(cfg.device)
    reg_loss = build_loss(cfg.reg_loss_cfg).to(cfg.device)
    dice_loss = build_loss(cfg.dice_loss_cfg)
    loss_funcs = dict(sim=sim_loss, reg=reg_loss, dice=dice_loss)

    # define metric
    compute_dice = DiceMetric(include_background=True, reduction='mean')
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
                 dataloader,
                 loss_funcs,
                 metric_funcs,
                 scaler,
                 optimizer,
                 len_dataset * epoch,
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
