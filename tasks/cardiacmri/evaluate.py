"""Evaluate a 2D cardiac-MRI model trained on ACDC, on ACDC or on M&Ms.

Registration is between the end-diastolic and end-systolic frame of one patient.
Every valid short-axis slice is stacked into the batch dimension, so the metrics
below are per-slice, not per-patient.

The architecture is not re-specified here: train.py dumps the resolved config to
<method_folder>/exp<N>/train_configs.py, and this script reads the model_cfg and
the registration head back from it. Only the evaluation data comes from
eval_configs/.

Examples:
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 200
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 200 --dataset mms
    python evaluate.py -m ./pwc_iter_outputs -exp 0 -epoch 200 --flow-idx 0 --scale 16
"""

import os
import sys
from functools import partial
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import nibabel as nib
import numpy as np
import torch
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import (DiceMetric, PSNRMetric, SurfaceDistanceMetric,
                           compute_hausdorff_distance, compute_surface_dice)
from monai.networks.utils import one_hot

from models import (Warp, build_flow_estimator, build_metrics,
                    build_registration_head)
from utils import load_data_ACDC_2d, load_data_MMs_2d, worker_init_fn

# dataset key -> (loader, eval config)
DATASETS = {
    'acdc': (load_data_ACDC_2d, './eval_configs/acdc.py'),
    'mms': (load_data_MMs_2d, './eval_configs/mms.py'),
}

# ACDC and M&Ms share the same label protocol
LABEL_MAP = {
    0: 'background',
    1: 'left ventricle',
    2: 'myocardium',
    3: 'right ventricle',
}


def set_image_size(cfg, image_size):
    """Rewrite every nested image_size key of a config, in place.

    The pyramidal decoders and the registration head each hold their own copy of
    the full-resolution spatial size, so evaluating an ACDC-trained model on
    M&Ms at its native 256x256 means updating all of them, not just the
    top-level one.
    """
    for key, value in list(cfg.items()):
        if key == 'image_size':
            cfg[key] = image_size
        elif isinstance(value, dict):
            set_image_size(value, image_size)


def get_flow_lists(model, source, target):
    """Every flow a pyramid model produces, coarse to fine, both directions.

    Iterative models expose forward_eval(), which additionally returns the
    intermediate flow of each refinement iteration, not just the last one per
    level. Single-stream models return no backward flow, so it is obtained by
    swapping the inputs. Either way the final entry is the model's actual
    output, so flow_idx=-1 scores what the model would be used for in practice.
    """
    if hasattr(model, 'forward_eval'):
        result = model.forward_eval(source, target)
        bck_flows = result['bck']
        if bck_flows is None:
            bck_flows = model.forward_eval(target, source)['fwd']
        return result['fwd'], bck_flows

    fwd_flows, bck_flows = model(source, target)
    if bck_flows is None:
        bck_flows, _ = model(target, source)
    return fwd_flows, bck_flows


def save_pair_as_nifti(cfg, data, tensors):
    """Dump the images and segmentations of one patient for visual inspection."""
    save_path = os.path.join(cfg.save_dir, f'images_{cfg.epoch_id:03d}')
    os.makedirs(save_path, exist_ok=True)

    patient_id = data['patient_id']
    if isinstance(patient_id, (list, tuple)):
        patient_id = patient_id[0]

    # the slice axis was folded into the batch dimension, so put it back last
    affine = np.diag([*cfg.spacing, 1.0])
    for name, tensor in tensors.items():
        volume = tensor.permute(dims=(1, 2, 3, 0)).cpu().numpy().squeeze()
        nib.save(nib.Nifti1Image(volume, affine),
                 os.path.join(save_path, f'{patient_id}_{name}.nii.gz'))


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
    loader, _ = DATASETS[cfg.dataset]
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
    compute_psnr = PSNRMetric(max_val=1.0)._compute_metric
    compute_mae = lambda x, y: torch.mean(torch.abs(x - y), dim=(1, 2, 3))
    compute_jacdet = build_metrics(dict(type='jacdet2d'))
    compute_hd95 = partial(compute_hausdorff_distance, include_background=False, percentile=95)
    compute_assd = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction='mean')
    compute_nsd = partial(compute_surface_dice, include_background=False, spacing=1.5, use_subvoxels=False)

    metric_funcs = dict(dice=compute_dice,
                        psnr=compute_psnr,
                        mae=compute_mae,
                        jacdet=compute_jacdet,
                        hd=compute_hd95,
                        assd=compute_assd,
                        nsd=compute_nsd)
    overlap_metrics = ('dice', 'hd95', 'assd', 'nsd')

    metrics = {f'{d}_{m}': [] for d in ('init', 'fwd', 'bck')
               for m in ('psnr', 'mae') + overlap_metrics}
    metrics.update({f'{d}_{m}': [] for d in ('fwd', 'bck')
                    for m in ('log_jacdet', 'np_jacdet')})

    def score_overlap(direction, moved_oh, fixed_oh, class_thresholds):
        metrics[f'{direction}_dice'].append(
            metric_funcs['dice'](moved_oh, fixed_oh).squeeze().cpu().numpy())
        metrics[f'{direction}_hd95'].append(
            metric_funcs['hd'](moved_oh, fixed_oh).squeeze().cpu().numpy())
        metrics[f'{direction}_assd'].append(
            metric_funcs['assd'](moved_oh, fixed_oh).squeeze().cpu().numpy())
        metrics[f'{direction}_nsd'].append(
            metric_funcs['nsd'](moved_oh, fixed_oh, class_thresholds).squeeze().cpu().numpy())

    with torch.no_grad():
        for data in dataloader:
            source = data['ED'].float().to(cfg.device)
            target = data['ES'].float().to(cfg.device)
            source_seg = data['ED_seg'].float().to(cfg.device)
            target_seg = data['ES_seg'].float().to(cfg.device)

            # the loader stacks a patient's valid short-axis slices along the
            # channel axis; move them into the batch axis for 2D registration
            B, C, H, W = source.shape
            assert B == 1
            source = source.view(B * C, 1, H, W)
            target = target.view(B * C, 1, H, W)
            source_seg = source_seg.view(B * C, 1, H, W)
            target_seg = target_seg.view(B * C, 1, H, W)

            source_oh = one_hot(source_seg, num_classes=len(LABEL_MAP),
                                dtype=torch.float, dim=1).to(cfg.device)
            target_oh = one_hot(target_seg, num_classes=len(LABEL_MAP),
                                dtype=torch.float, dim=1).to(cfg.device)
            class_thresholds = [1] * (source_oh.shape[1] - 1)

            metrics['init_psnr'].append(
                metric_funcs['psnr'](source, target).squeeze().cpu().numpy())
            metrics['init_mae'].append(
                metric_funcs['mae'](source, target).squeeze().cpu().numpy())
            score_overlap('init', source_oh, target_oh, class_thresholds)

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

            fwd_flow, y_source, y_source_oh = reg_head(fwd_flow, source, source_oh)
            bck_flow, y_target, y_target_oh = reg_head(bck_flow, target, target_oh)

            metrics['fwd_psnr'].append(
                metric_funcs['psnr'](y_source, target).squeeze().cpu().numpy())
            metrics['bck_psnr'].append(
                metric_funcs['psnr'](y_target, source).squeeze().cpu().numpy())
            metrics['fwd_mae'].append(
                metric_funcs['mae'](y_source, target).squeeze().cpu().numpy())
            metrics['bck_mae'].append(
                metric_funcs['mae'](y_target, source).squeeze().cpu().numpy())
            score_overlap('fwd', y_source_oh, target_oh, class_thresholds)
            score_overlap('bck', y_target_oh, source_oh, class_thresholds)

            # jacdet
            fwd_jacdet = metric_funcs['jacdet'](fwd_flow.detach().cpu().numpy())
            bck_jacdet = metric_funcs['jacdet'](bck_flow.detach().cpu().numpy())
            metrics['fwd_log_jacdet'].append(
                np.std(np.log((fwd_jacdet + 3).clip(1e-9, 1e9)), axis=(1, 2)))
            metrics['bck_log_jacdet'].append(
                np.std(np.log((bck_jacdet + 3).clip(1e-9, 1e9)), axis=(1, 2)))

            # non-positive Jacobian rate, restricted to the body. The 2-voxel
            # crop drops the finite-difference border.
            source_fg = torch.where(source > 0.0, 1.0, 0.0).cpu().numpy().squeeze(1)[:, 2:-2, 2:-2]
            target_fg = torch.where(target > 0.0, 1.0, 0.0).cpu().numpy().squeeze(1)[:, 2:-2, 2:-2]
            metrics['fwd_np_jacdet'].append(
                (np.sum((fwd_jacdet <= 0) * target_fg, axis=(1, 2))
                 / np.sum(target_fg, axis=(1, 2))) * 100)
            metrics['bck_np_jacdet'].append(
                (np.sum((bck_jacdet <= 0) * source_fg, axis=(1, 2))
                 / np.sum(source_fg, axis=(1, 2))) * 100)

            print('init_dice:', np.mean(metrics['init_dice'][-1]),
                  'fwd_dice:', np.mean(metrics['fwd_dice'][-1]))

            if cfg.save_images:
                save_pair_as_nifti(cfg, data, dict(
                    source=source, target=target,
                    y_source=y_source, y_target=y_target,
                    source_seg=source_seg, target_seg=target_seg,
                    y_source_seg=warp_nearest(source_seg, fwd_flow),
                    y_target_seg=warp_nearest(target_seg, bck_flow)))

    results = {k: np.concatenate(v, axis=0) for k, v in metrics.items()}

    os.makedirs(cfg.save_dir, exist_ok=True)
    # non-default pyramid levels get their own file, so a sweep over --flow-idx
    # does not overwrite the headline numbers
    suffix = '' if cfg.flow_idx == -1 and cfg.scale is None \
        else f'_s{cfg.scale}_f{cfg.flow_idx}'
    np.savez(os.path.join(cfg.save_dir,
                          f'{cfg.prefix}_metrics_{cfg.epoch_id:03d}{suffix}.npz'),
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
                   default='acdc',
                   choices=list(DATASETS),
                   help='evaluation dataset (default: acdc). mms is the '
                        'cross-dataset generalisation set.')
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
    p.add_argument('--image-size',
                   default=None,
                   type=int,
                   help='override the evaluation crop size, e.g. 128 to score '
                        'M&Ms at the ACDC training resolution instead of its '
                        'native 256.')
    p.add_argument('--save-images',
                   action='store_true',
                   help='save original and registered images/segmentations')
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
    if args.image_size is not None:
        config.image_size = [args.image_size, args.image_size]
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
        save_images=args.save_images,
    ))
    # the model, the registration head and the warping grid are all sized from
    # the evaluation crop, which may differ from the training crop when a model
    # trained on ACDC (128) is applied to M&Ms at its native 256
    set_image_size(config, config.image_size)
    config.prefix = f'{config.prefix}_{config.image_size[0]}'
    print(f'[eval] dataset={config.dataset} image_size={config.image_size} '
          f'pyramid={config.pyramid} flow_idx={config.flow_idx} '
          f'scale={config.scale or "from config"}')

    set_seed(2023)
    infer(config)
