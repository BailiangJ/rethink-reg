"""Evaluate a lung-CT model trained on NLST, on NLST or on Lung250M-4B.

The architecture is not re-specified here: train.py dumps the resolved config to
<method_folder>/exp<N>/train_configs.py, and this script reads the model_cfg and
the registration head back from it. Only the evaluation data comes from
eval_configs/.

Lung250M-4B is the cross-dataset generalisation set: models are never trained on
it, and it ships landmarks but no lung masks, so the mask-based Dice/HD95 and the
lung-region non-positive-Jacobian rates are NLST-only.

Examples:
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 100
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 100 --dataset lung250m
    python evaluate.py -m ./pwc_iter_outputs -exp 0 -epoch 100 --flow-idx 0 --scale 16
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 100 --tre-only
"""

import os
import sys
from functools import partial
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import (DiceMetric, PSNRMetric, SSIMMetric,
                           compute_hausdorff_distance)

from models import (Warp, build_flow_estimator, build_metrics,
                    build_registration_head)
from utils import load_data_Lung250M, load_data_NLST, worker_init_fn

# dataset key -> (loader, eval config, ships lung masks)
DATASETS = {
    'nlst': (load_data_NLST, './eval_configs/nlst.py', True),
    'lung250m': (partial(load_data_Lung250M, val=False),
                 './eval_configs/lung250m.py', False),
}


def get_flow_lists(model, source, target):
    """Every flow a pyramid model produces, coarse to fine, both directions.

    Iterative models expose forward_eval(), which additionally returns the
    intermediate flow of each refinement iteration, not just the last one per
    level. Either way the final entry is the model's actual output, so
    flow_idx=-1 scores what the model would be used for in practice.
    """
    if hasattr(model, 'forward_eval'):
        result = model.forward_eval(source, target)
        if result['bck'] is None:
            raise ValueError('scoring a pyramid needs a bidirectional model.')
        return result['fwd'], result['bck']
    return model(source, target)


def infer(cfg):
    print(cfg.load_model)
    model = build_flow_estimator(cfg.model_cfg)
    model.load_state_dict(
        torch.load(cfg.load_model, map_location=torch.device(cfg.device)))
    model.to(cfg.device)
    model.eval()

    # build registration head module. The default cfg.registration_cfg upsamples
    # by 2 (the final flow is at half resolution); scoring a coarser pyramid
    # level needs the matching factor, e.g. --scale 16 with --flow-idx 0.
    reg_cfg = cfg.registration_cfg
    if cfg.scale is not None:
        reg_cfg = dict(type='RegistrationHead',
                       image_size=cfg.image_size,
                       spatial_scale=cfg.scale,
                       flow_scale=cfg.scale,
                       interp_mode='bilinear')
    reg_head = build_registration_head(reg_cfg).to(cfg.device)
    reg_head.eval()
    warp_nearest = Warp(cfg.image_size, interp_mode='nearest').to(cfg.device)

    # load data
    loader, _, has_masks = DATASETS[cfg.dataset]
    dataset = loader(cfg,
                     cache_rate=cfg.cache_rate,
                     num_workers=cfg.num_workers,
                     )
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            worker_init_fn=worker_init_fn)

    # define metric
    compute_dice = DiceMetric(include_background=False, reduction='mean')
    compute_hd95 = partial(compute_hausdorff_distance, include_background=False, percentile=95)
    compute_ssim = SSIMMetric(data_range=torch.tensor(1.0),
                              spatial_dims=3)._compute_metric
    compute_psnr = PSNRMetric(max_val=1.0)._compute_metric
    compute_jacdet = build_metrics(dict(type='jacdet'))
    compute_tre = build_metrics(cfg.tre_cfg)
    metric_funcs = dict(dice=compute_dice,
                        hd=compute_hd95,
                        ssim=compute_ssim,
                        psnr=compute_psnr,
                        jacdet=compute_jacdet,
                        tre=compute_tre)
    # the TRE metric reports millimetres, so the initial (unregistered) TRE has
    # to be scaled by the voxel spacing to be comparable with it
    spacing = torch.tensor(cfg.tre_cfg.spacing).view(1, 1, -1).to(cfg.device)

    metrics = {k: [] for k in (
        'init_tre', 'fwd_tre', 'bck_tre',
        'init_ssim', 'fwd_ssim', 'bck_ssim',
        'init_psnr', 'fwd_psnr', 'bck_psnr',
        'init_dice', 'fwd_dice', 'bck_dice',
        'init_hd95', 'fwd_hd95', 'bck_hd95',
        'fwd_log_jacdet', 'bck_log_jacdet',
        'fwd_np_jacdet', 'bck_np_jacdet',
        'fwd_lung_np_jacdet', 'bck_lung_np_jacdet')}

    with torch.no_grad():
        for data in dataloader:
            source = data['inp'].float().to(cfg.device)
            target = data['exp'].float().to(cfg.device)
            src_keypnt = data['inp_keypnt'].float().to(cfg.device)
            tgt_keypnt = data['exp_keypnt'].float().to(cfg.device)
            if has_masks:
                src_mask = data['inp_mask'].float().to(cfg.device)
                tgt_mask = data['exp_mask'].float().to(cfg.device)

            init_tre = torch.linalg.norm(spacing * (tgt_keypnt - src_keypnt), dim=-1)

            if cfg.pyramid:
                fwd_py_flows, bck_py_flows = get_flow_lists(model, source, target)
                if not -len(fwd_py_flows) <= cfg.flow_idx < len(fwd_py_flows):
                    raise IndexError(
                        f'--flow-idx {cfg.flow_idx} is out of range for '
                        f'{len(fwd_py_flows)} pyramid levels.')
                fwd_flow = fwd_py_flows[cfg.flow_idx]
                bck_flow = bck_py_flows[cfg.flow_idx]
            else:
                if 'Dual' in cfg.model_cfg.type and cfg.model_cfg.get('bidirectional', cfg.model_cfg.get('config', {}).get('bidirectional', False)):
                    fwd_flow, bck_flow = model(source, target)
                else:
                    fwd_flow = model(source, target)
                    bck_flow = model(target, source)

            fwd_flow, y_source, _ = reg_head(fwd_flow, source)
            bck_flow, y_target, _ = reg_head(bck_flow, target)

            fwd_tre = metric_funcs['tre'](fwd_flow, tgt_keypnt, src_keypnt)
            bck_tre = metric_funcs['tre'](bck_flow, src_keypnt, tgt_keypnt)
            print('init_tre:', init_tre.mean())
            print('fwd_tre:', fwd_tre.mean())
            print('bck_tre:', bck_tre.mean())

            if cfg.tre_only:
                # keep the full per-keypoint distribution instead of one mean
                # per case, for the TRE box plots
                metrics['init_tre'].append(init_tre.squeeze().cpu().numpy())
                metrics['fwd_tre'].append(fwd_tre.squeeze().cpu().numpy())
                metrics['bck_tre'].append(bck_tre.squeeze().cpu().numpy())
                continue

            metrics['init_tre'].append(init_tre.mean().squeeze().cpu().numpy())
            metrics['fwd_tre'].append(fwd_tre.mean().squeeze().cpu().numpy())
            metrics['bck_tre'].append(bck_tre.mean().squeeze().cpu().numpy())

            # intensity agreement
            for name, (a, b) in dict(
                    init=(source, target),
                    fwd=(y_source, target),
                    bck=(y_target, source)).items():
                metrics[f'{name}_ssim'].append(
                    metric_funcs['ssim'](a, b).squeeze().cpu().numpy())
                metrics[f'{name}_psnr'].append(
                    metric_funcs['psnr'](a, b).squeeze().cpu().numpy())

            # dice and hd95 on the lung masks
            if has_masks:
                y_src_mask = warp_nearest(src_mask, fwd_flow)
                y_tgt_mask = warp_nearest(tgt_mask, bck_flow)
                for name, (a, b) in dict(
                        init=(src_mask, tgt_mask),
                        fwd=(y_src_mask, tgt_mask),
                        bck=(y_tgt_mask, src_mask)).items():
                    metrics[f'{name}_dice'].append(
                        metric_funcs['dice'](a, b).squeeze().cpu().numpy())
                    metrics[f'{name}_hd95'].append(
                        metric_funcs['hd'](a, b).squeeze().cpu().numpy())
                print('init_dice:', metrics['init_dice'][-1],
                      'fwd_dice:', metrics['fwd_dice'][-1])
                print('init_hd95:', metrics['init_hd95'][-1],
                      'fwd_hd95:', metrics['fwd_hd95'][-1])

            # jacdet
            fwd_jacdet = metric_funcs['jacdet'](fwd_flow.detach().cpu().numpy())
            bck_jacdet = metric_funcs['jacdet'](bck_flow.detach().cpu().numpy())
            metrics['fwd_log_jacdet'].append(
                np.std(np.log((fwd_jacdet + 3).clip(1e-9, 1e9)), axis=(1, 2, 3)))
            metrics['bck_log_jacdet'].append(
                np.std(np.log((bck_jacdet + 3).clip(1e-9, 1e9)), axis=(1, 2, 3)))

            # non-positive Jacobian rate, restricted to the body (and, on NLST,
            # to the lungs). The 2-voxel crop drops the finite-difference border.
            src_fg = torch.where(source > 0.0, 1.0, 0.0).cpu().numpy().squeeze(1)[:, 2:-2, 2:-2, 2:-2]
            tgt_fg = torch.where(target > 0.0, 1.0, 0.0).cpu().numpy().squeeze(1)[:, 2:-2, 2:-2, 2:-2]
            metrics['fwd_np_jacdet'].append(
                (np.sum((fwd_jacdet <= 0) * tgt_fg) / np.sum(tgt_fg)) * 100)
            metrics['bck_np_jacdet'].append(
                (np.sum((bck_jacdet <= 0) * src_fg) / np.sum(src_fg)) * 100)
            if has_masks:
                src_lung = src_mask.cpu().numpy().squeeze(1)[:, 2:-2, 2:-2, 2:-2]
                tgt_lung = tgt_mask.cpu().numpy().squeeze(1)[:, 2:-2, 2:-2, 2:-2]
                metrics['fwd_lung_np_jacdet'].append(
                    (np.sum((fwd_jacdet <= 0) * tgt_lung) / np.sum(tgt_lung)) * 100)
                metrics['bck_lung_np_jacdet'].append(
                    (np.sum((bck_jacdet <= 0) * src_lung) / np.sum(src_lung)) * 100)

    results = {k: np.hstack(v) for k, v in metrics.items() if v}

    os.makedirs(cfg.save_dir, exist_ok=True)
    # non-default pyramid levels get their own file, so a sweep over --flow-idx
    # does not overwrite the headline numbers
    suffix = '' if cfg.flow_idx == -1 and cfg.scale is None \
        else f'_s{cfg.scale}_f{cfg.flow_idx}'
    kind = 'tre' if cfg.tre_only else 'metrics'
    np.savez(os.path.join(cfg.save_dir,
                          f'{cfg.prefix}_{kind}_{cfg.epoch_id:03d}{suffix}.npz'),
             **results)


if __name__ == '__main__':
    import pathlib

    import configargparse

    from utils import set_seed

    p = configargparse.ArgParser()
    p.add_argument('--method-folder',
                   '-m',
                   required=True,
                   type=lambda f: pathlib.Path(f).absolute(),
                   help='path of method folder')
    p.add_argument('--exp-id',
                   '-exp',
                   required=True,
                   type=int)
    p.add_argument('--epoch-id',
                   '-epoch',
                   required=True,
                   type=int)
    p.add_argument('--dataset',
                   '-d',
                   default='nlst',
                   choices=list(DATASETS),
                   help='evaluation dataset (default: nlst). lung250m is the '
                        'cross-dataset generalisation set and has no lung masks.')
    p.add_argument('--flow-idx',
                   '-i',
                   default=-1,
                   type=int,
                   help='pyramid level to score: -1 = final flow (default, and '
                        'what the paper reports), 0 = coarsest. Ignored for '
                        'non-pyramid models.')
    p.add_argument('--scale',
                   '-s',
                   default=None,
                   type=int,
                   help='override the registration-head upsampling factor. '
                        'Needed when --flow-idx selects a coarser level, whose '
                        'flow is at 1/scale resolution (e.g. -s 16 -i 0). '
                        'Default: take it from the training config.')
    p.add_argument('--tre-only',
                   action='store_true',
                   help='score landmarks only, and keep every keypoint rather '
                        'than one mean per case (for the TRE distribution plots)')
    args = p.parse_args()
    save_dir = os.path.join(args.method_folder, f'exp{args.exp_id}/eval')
    train_cfg = Config.fromfile(os.path.join(args.method_folder, f'exp{args.exp_id}/train_configs.py'))
    if train_cfg.get('model_cfg', False):
        model_cfg = train_cfg.model_cfg
        load_model = os.path.join(args.method_folder, f'exp{args.exp_id}/saved_models/{args.epoch_id:04d}.pth')
    else:
        model_cfg = train_cfg.reg_model_cfg
        load_model = os.path.join(args.method_folder, f'exp{args.exp_id}/saved_models/reg_{args.epoch_id:04d}.pth')

    config = Config.fromfile(DATASETS[args.dataset][1])
    config.update(dict(
        epoch_id=args.epoch_id,
        save_dir=save_dir,
        load_model=load_model,
        model_cfg=model_cfg,
        registration_cfg=train_cfg.registration_cfg,
        # a model is pyramidal iff it was trained with a scale pyramid
        pyramid=train_cfg.get('scale_pyramid', None) is not None,
        dataset=args.dataset,
        flow_idx=args.flow_idx,
        scale=args.scale,
        tre_only=args.tre_only,
    ))
    print(f'[eval] dataset={config.dataset} pyramid={config.pyramid} '
          f'flow_idx={config.flow_idx} scale={config.scale or "from config"} '
          f'tre_only={config.tre_only}')

    set_seed(2023)
    infer(config)
