"""Evaluate abdomen MR-CT registration on AbdomenMRCT (Learn2Reg).

MR and CT are acquired separately, so every pair is pre-aligned before the
deformable model sees it -- centroid translation by default, rigid with
``--rigid``. Metrics are therefore reported at three stages: initial (no
alignment at all), after pre-alignment, and after deformable registration.

The dataset's 16 pairs are split in half and the two halves use different
labelling schemes, so ``--split`` selects which half to score and must be the
opposite of what ``train.py --split`` was given. Organs missing from either
modality of a pair are left as NaN rather than scored as zero.

The architecture is not re-specified here: train.py dumps the resolved config to
<method_folder>/exp<N>/train_configs.py, and this script reads the model_cfg and
the registration head back from it.

Examples:
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 100 --split tr
    python evaluate.py -m ./pwc_iter_outputs -exp 0 -epoch 100 --split tr -i 0 -s 16
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
import pandas as pd
import torch
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import (DiceMetric, SurfaceDistanceMetric,
                           compute_hausdorff_distance, compute_surface_dice)
from utils import (ABDOMENMRCT_ORGAN_NAMES, ABDOMENMRCT_SPLITS,
                   load_data_AbdomenMRCT_pair, resolve_path, worker_init_fn)
from models import build_metrics, build_registration_head, build_flow_estimator
from prealign import RigidPreAlign, TranslationPreAlign

torch.backends.cudnn.deterministic = True


def get_flow_lists(model, source, target):
    """Every flow a pyramid model produces, coarse to fine, both directions.

    Iterative models expose forward_eval(), which additionally returns the
    intermediate flow of each refinement iteration, not just the last one per
    level. Either way the final entry is the model's actual output, so
    flow_idx=-1 scores what the model would be used for in practice.
    """
    if hasattr(model, 'forward_eval'):
        result = model.forward_eval(source, target)
        fwd_flows, bck_flows = result['fwd'], result['bck']
        if bck_flows is None:
            bck_flows = model.forward_eval(target, source)['fwd']
        return fwd_flows, bck_flows

    fwd_flows, bck_flows = model(source, target)
    if bck_flows is None:
        bck_flows, _ = model(target, source)
    return fwd_flows, bck_flows


def build_one_hot(mask, labels):
    """One-hot stack of `labels` out of a raw label map, in the given order."""
    one_hot = torch.zeros((1, len(labels), *mask.shape[1:]), device=mask.device)
    for idx, label in enumerate(labels):
        one_hot[:, idx] = (mask == label).float()
    return one_hot


def metric_values(metric):
    """Flatten a batch-one per-organ metric without collapsing one-organ cases."""
    return metric.detach().cpu().reshape(-1).numpy()


def eval(cfg):
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

    # raw label maps, so that organs absent from a pair can be excluded below
    dataset = load_data_AbdomenMRCT_pair(cfg,
                                         split=cfg.split,
                                         one_hot=False,
                                         cache_rate=cfg.cache_rate,
                                         num_workers=cfg.num_workers)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            worker_init_fn=worker_init_fn)

    # define metric
    compute_dice = DiceMetric(include_background=True, reduction='mean')
    compute_hd95 = partial(compute_hausdorff_distance, include_background=True, percentile=95)
    compute_assd = SurfaceDistanceMetric(include_background=True, symmetric=True, reduction='mean')
    compute_nsd = partial(compute_surface_dice,
                          include_background=True,
                          spacing=2.0,
                          use_subvoxels=False)
    compute_jacdet = build_metrics(dict(type='jacdet'))
    metric_funcs = dict(dice=compute_dice,
                        hd=compute_hd95,
                        assd=compute_assd,
                        nsd=compute_nsd,
                        jacdet=compute_jacdet)

    # the same pre-alignment training used; translation is the paper's setting
    if cfg.rigid:
        prealigner = RigidPreAlign(compute_dice=compute_dice,
                                   device=cfg.device)
    else:
        prealigner = TranslationPreAlign(compute_dice=compute_dice,
                                         device=cfg.device)

    # Initialize list to store metrics for each subject
    metrics_list = []

    # label id -> organ name for this half of the dataset
    label_map = ABDOMENMRCT_ORGAN_NAMES[cfg.split]
    split_labels = ABDOMENMRCT_SPLITS[cfg.split]['label_list']
    image_dir = ABDOMENMRCT_SPLITS[cfg.split]['image_dir']
    all_possible_organs = [label_map[label] for label in split_labels]

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            ct = data['ct'].float().to(cfg.device)
            mr = data['mr'].float().to(cfg.device)
            ct_mask = data['ct_label'].float().to(cfg.device)
            mr_mask = data['mr_label'].float().to(cfg.device)

            # Keep only the organs this split labels *and* both modalities have
            ct_labels = torch.unique(ct_mask[ct_mask > 0]).cpu().numpy()
            mr_labels = torch.unique(mr_mask[mr_mask > 0]).cpu().numpy()
            common_labels = np.array([label for label in split_labels
                                      if label in ct_labels and label in mr_labels])
            if common_labels.size == 0:
                print(f'subject {data["id"].item()}: no organ present in both '
                      f'modalities, skipping')
                continue

            ct_oh = build_one_hot(ct_mask, common_labels)
            mr_oh = build_one_hot(mr_mask, common_labels)

            with torch.enable_grad():
                processed = prealigner.process_pair(
                    src_img=mr,
                    src_oh=mr_oh,  # now one-hot encoded
                    tgt_img=ct,
                    tgt_oh=ct_oh   # now one-hot encoded
                )
            source = processed['src_image'].float().to(cfg.device)
            target = processed['tgt_image'].float().to(cfg.device)
            src_oh = processed['src_seg'].float().to(cfg.device)
            tgt_oh = processed['tgt_seg'].float().to(cfg.device)

            # Deformable registration
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
            flow, y_source, y_source_oh = reg_head(fwd_flow, source, src_oh)
            # Backward registration
            bck_flow, y_target, y_target_oh = reg_head(bck_flow, target, tgt_oh)

            # Compute dice for all three stages
            init_dice = metric_funcs['dice'](ct_oh, mr_oh)
            trans_dice = metric_funcs['dice'](src_oh, tgt_oh)
            reg_dice = metric_funcs['dice'](y_source_oh, tgt_oh)
            # Backward registration dice
            bck_reg_dice = metric_funcs['dice'](y_target_oh, src_oh)

            # compute hd95
            init_hd95 = metric_funcs['hd'](ct_oh, mr_oh)
            trans_hd95 = metric_funcs['hd'](src_oh, tgt_oh)
            reg_hd95 = metric_funcs['hd'](y_source_oh, tgt_oh)
            # Backward registration hd95
            bck_reg_hd95 = metric_funcs['hd'](y_target_oh, src_oh)

            # compute assd average symmetric surface distance
            init_assd = metric_funcs['assd'](ct_oh, mr_oh)
            trans_assd = metric_funcs['assd'](src_oh, tgt_oh)
            reg_assd = metric_funcs['assd'](y_source_oh, tgt_oh)
            # backward
            bck_reg_assd = metric_funcs['assd'](y_target_oh, src_oh)

            class_thresholds = [1] * ct_oh.shape[1]
            # compute normalized surface dice
            init_nsd = metric_funcs['nsd'](ct_oh, mr_oh, class_thresholds)
            trans_nsd = metric_funcs['nsd'](src_oh, tgt_oh, class_thresholds)
            reg_nsd = metric_funcs['nsd'](y_source_oh, tgt_oh, class_thresholds)
            # backward
            bck_reg_nsd = metric_funcs['nsd'](y_target_oh, src_oh, class_thresholds)

            # compute jacdet for fwd_flow
            ct_fg = torch.where(ct > 0.0, 1.0, 0.0).cpu().numpy().squeeze(1)[:, 2:-2, 2:-2, 2:-2]
            mr_fg = torch.where(mr > 0.0, 1.0, 0.0).cpu().numpy().squeeze(1)[:, 2:-2, 2:-2, 2:-2]

            flow_jacdet = metric_funcs['jacdet'](flow.detach().cpu().numpy())
            log_jacdet = np.std(np.log((flow_jacdet + 3).clip(1e-9, 1e9)), axis=(1, 2, 3))
            np_jacdet = (np.sum((flow_jacdet <= 0) * ct_fg) / np.sum(ct_fg)) * 100
            # compute jacdet for bck_flow
            bck_flow_jacdet = metric_funcs['jacdet'](bck_flow.detach().cpu().numpy())
            bck_log_jacdet = np.std(np.log((bck_flow_jacdet + 3).clip(1e-9, 1e9)), axis=(1, 2, 3))
            bck_np_jacdet = (np.sum((bck_flow_jacdet <= 0) * mr_fg) / np.sum(mr_fg)) * 100

            # Create a dictionary for this subject's metrics
            subject_metrics = {
                'subject_id': data['id'].item(),
                'log_jacdet': log_jacdet.item(),
                'np_jacdet': np_jacdet.item(),
                'bck_log_jacdet': bck_log_jacdet.item(),
                'bck_np_jacdet': bck_np_jacdet.item()
            }

            # Initialize all organ metrics with NaN
            for organ in all_possible_organs:
                subject_metrics.update({
                    f'init_dice_{organ}': np.nan,
                    f'trans_dice_{organ}': np.nan,
                    f'reg_dice_{organ}': np.nan,
                    f'init_hd95_{organ}': np.nan,
                    f'trans_hd95_{organ}': np.nan,
                    f'reg_hd95_{organ}': np.nan,
                    f'bck_reg_dice_{organ}': np.nan,
                    f'bck_reg_hd95_{organ}': np.nan,
                    f'init_assd_{organ}': np.nan,
                    f'trans_assd_{organ}': np.nan,
                    f'reg_assd_{organ}': np.nan,
                    f'bck_reg_assd_{organ}': np.nan,
                    f'init_nsd_{organ}': np.nan,
                    f'trans_nsd_{organ}': np.nan,
                    f'reg_nsd_{organ}': np.nan,
                    f'bck_reg_nsd_{organ}': np.nan
                })

            # Save images if requested
            if cfg.save_image:
                ref_img_path = os.path.join(resolve_path(cfg.data_dir), image_dir,
                                            f'AbdomenMRCT_{data["id"].item():04d}_0000.nii.gz')
                ref_img = nib.load(ref_img_path)
                ref_affine = ref_img.affine
                ref_header = ref_img.header

                # Create directory for this subject
                subject_dir = os.path.join(cfg.save_dir, f'subject_{data["id"].item()}')
                os.makedirs(subject_dir, exist_ok=True)

                # Convert one-hot encoded segmentation back to label map
                # First get the original masks to identify background
                y_source_bg = (torch.sum(y_source_oh, dim=1) == 0)
                src_bg = (torch.sum(src_oh, dim=1) == 0)
                tgt_bg = (torch.sum(tgt_oh, dim=1) == 0)

                # Get organ labels from argmax
                y_source_label = torch.argmax(y_source_oh, dim=1).float().cpu().numpy()
                src_label = torch.argmax(src_oh, dim=1).float().cpu().numpy()
                tgt_label = torch.argmax(tgt_oh, dim=1).float().cpu().numpy()

                # Map the labels back to original values
                label_mapping = common_labels
                y_source_label = label_mapping[y_source_label.astype(int)]
                src_label = label_mapping[src_label.astype(int)]
                tgt_label = label_mapping[tgt_label.astype(int)]

                # Set background to 0
                y_source_label[y_source_bg.cpu().numpy()] = 0
                src_label[src_bg.cpu().numpy()] = 0
                tgt_label[tgt_bg.cpu().numpy()] = 0

                # Save images and labels
                for name, img in [
                    ('source', source.squeeze().cpu().numpy()),
                    ('target', target.squeeze().cpu().numpy()),
                    ('warped_source', y_source.squeeze().cpu().numpy()),
                    ('source_label', src_label.squeeze()),
                    ('target_label', tgt_label.squeeze()),
                    ('warped_source_label', y_source_label.squeeze())
                ]:
                    # Remove batch dimension if present
                    if img.ndim == 5:  # [B, C, D, H, W]
                        img = img[0]
                    elif img.ndim == 4:  # [B, D, H, W]
                        img = img[0]

                    # Create NIfTI image
                    nii_img = nib.Nifti1Image(img, ref_affine, ref_header)
                    nib.save(nii_img, os.path.join(subject_dir, f'{name}.nii.gz'))

            presented_organs = [label_map[label] for label in common_labels]

            # Update metrics only for present organs
            for organ, init_d, trans_d, reg_d, init_h, trans_h, reg_h, bck_d, bck_h, \
                init_a, trans_a, reg_a, bck_a, \
                init_n, trans_n, reg_n, bck_n in zip(
                presented_organs,
                metric_values(init_dice),
                metric_values(trans_dice),
                metric_values(reg_dice),
                metric_values(init_hd95),
                metric_values(trans_hd95),
                metric_values(reg_hd95),
                metric_values(bck_reg_dice),
                metric_values(bck_reg_hd95),
                metric_values(init_assd),
                metric_values(trans_assd),
                metric_values(reg_assd),
                metric_values(bck_reg_assd),
                metric_values(init_nsd),
                metric_values(trans_nsd),
                metric_values(reg_nsd),
                metric_values(bck_reg_nsd)
            ):
                subject_metrics.update({
                    f'init_dice_{organ}': init_d,
                    f'trans_dice_{organ}': trans_d,
                    f'reg_dice_{organ}': reg_d,
                    f'init_hd95_{organ}': init_h,
                    f'trans_hd95_{organ}': trans_h,
                    f'reg_hd95_{organ}': reg_h,
                    f'bck_reg_dice_{organ}': bck_d,
                    f'bck_reg_hd95_{organ}': bck_h,
                    f'init_assd_{organ}': init_a,
                    f'trans_assd_{organ}': trans_a,
                    f'reg_assd_{organ}': reg_a,
                    f'bck_reg_assd_{organ}': bck_a,
                    f'init_nsd_{organ}': init_n,
                    f'trans_nsd_{organ}': trans_n,
                    f'reg_nsd_{organ}': reg_n,
                    f'bck_reg_nsd_{organ}': bck_n
                })

            metrics_list.append(subject_metrics)

            print("=================================================")
            print(f'subject {data["id"].item()}: initial -> pre-aligned -> registered Dice')
            for init_dsc, trans_dsc, reg_dsc, organ in zip(metric_values(init_dice),
                                                           metric_values(trans_dice),
                                                           metric_values(reg_dice),
                                                           presented_organs):
                print(f'{organ}: {init_dsc:.4f} -> {trans_dsc:.4f} -> {reg_dsc:.4f}')
            print("=================================================")

    # Convert to DataFrame
    df = pd.DataFrame(metrics_list)

    # Save metrics. The split is always part of the name: the two halves are
    # different evaluations, and without it a `--split ts` run would silently
    # overwrite a `--split tr` one. The scale/level suffix is added only when
    # they were overridden, so the plain run keeps a stable name.
    suffix = '' if cfg.flow_idx == -1 and cfg.scale is None \
        else f'_s{cfg.scale}_f{cfg.flow_idx}'
    os.makedirs(cfg.save_dir, exist_ok=True)
    df.to_csv(os.path.join(cfg.save_dir,
                           f'metrics_{cfg.split}_{cfg.epoch_id:03d}{suffix}.csv'),
              index=False)

    # Print overall statistics
    print(f"\nOverall Statistics (split={cfg.split}):")
    # Get all organ names from column names
    organ_names = set()
    for col in df.columns:
        if col.startswith('init_dice_'):
            organ_names.add(col.replace('init_dice_', ''))

    # Print jacdet statistics first
    print("\nJacobian Statistics:")
    print(f"log_jacdet: {df['log_jacdet'].mean():.4f} ± {df['log_jacdet'].std():.4f}")
    print(f"np_jacdet: {df['np_jacdet'].mean():.4f} ± {df['np_jacdet'].std():.4f}")

    # Print organ-specific statistics
    print("\nOrgan-specific Statistics:")
    for organ in sorted(organ_names):
        init_col = f'init_dice_{organ}'
        trans_col = f'trans_dice_{organ}'
        reg_col = f'reg_dice_{organ}'
        init_assd_col = f'init_assd_{organ}'
        trans_assd_col = f'trans_assd_{organ}'
        reg_assd_col = f'reg_assd_{organ}'
        init_nsd_col = f'init_nsd_{organ}'
        trans_nsd_col = f'trans_nsd_{organ}'
        reg_nsd_col = f'reg_nsd_{organ}'
        init_hd95_col = f'init_hd95_{organ}'
        trans_hd95_col = f'trans_hd95_{organ}'
        reg_hd95_col = f'reg_hd95_{organ}'
        if all(col in df.columns for col in [init_col, trans_col, reg_col]):
            n_subjects = df[init_col].count()
            print(f"\n{organ} (n={n_subjects}):")
            print(f"Initial Dice: {df[init_col].mean():.4f} ± {df[init_col].std():.4f}")
            print(f"Pre-aligned Dice: {df[trans_col].mean():.4f} ± {df[trans_col].std():.4f}")
            print(f"Registered Dice: {df[reg_col].mean():.4f} ± {df[reg_col].std():.4f}")
            if all(col in df.columns for col in [init_hd95_col, trans_hd95_col, reg_hd95_col]):
                print(f"Initial HD95: {df[init_hd95_col].mean():.4f} ± {df[init_hd95_col].std():.4f}")
                print(f"Pre-aligned HD95: {df[trans_hd95_col].mean():.4f} ± {df[trans_hd95_col].std():.4f}")
                print(f"Registered HD95: {df[reg_hd95_col].mean():.4f} ± {df[reg_hd95_col].std():.4f}")
            if all(col in df.columns for col in [init_assd_col, trans_assd_col, reg_assd_col]):
                print(f"Initial ASSD: {df[init_assd_col].mean():.4f} ± {df[init_assd_col].std():.4f}")
                print(f"Pre-aligned ASSD: {df[trans_assd_col].mean():.4f} ± {df[trans_assd_col].std():.4f}")
                print(f"Registered ASSD: {df[reg_assd_col].mean():.4f} ± {df[reg_assd_col].std():.4f}")
            if all(col in df.columns for col in [init_nsd_col, trans_nsd_col, reg_nsd_col]):
                print(f"Initial NSD: {df[init_nsd_col].mean():.4f} ± {df[init_nsd_col].std():.4f}")
                print(f"Pre-aligned NSD: {df[trans_nsd_col].mean():.4f} ± {df[trans_nsd_col].std():.4f}")
                print(f"Registered NSD: {df[reg_nsd_col].mean():.4f} ± {df[reg_nsd_col].std():.4f}")


if __name__ == '__main__':
    import pathlib
    import configargparse
    from utils import set_seed

    script_dir = os.path.dirname(os.path.abspath(__file__))

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
    p.add_argument('--split', choices=['ts', 'tr'], default='tr',
                   help="Which half of AbdomenMRCT to use. "
                        "'ts' = indices 9-16 (imagesTs/TSlabelsTs, labels [5,2,3,1]); "
                        "'tr' = indices 1-8 (imagesTr/labelsTr, labels [1,2,3,4]). "
                        "train.py and evaluate.py must be given opposite values.")
    p.add_argument('--save-images',
                   action='store_true',
                   help='save output images and label maps as NIfTI files')
    p.add_argument('--rigid',
                   '-r',
                   action='store_true',
                   help='use rigid pre-alignment instead of translation-only '
                        '(the paper uses translation); must match how the '
                        'model was trained')
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
    args = p.parse_args()

    save_dir = os.path.join(args.method_folder, f'exp{args.exp_id}/eval')
    train_cfg = Config.fromfile(os.path.join(args.method_folder, f'exp{args.exp_id}/train_configs.py'))
    load_model = os.path.join(args.method_folder, f'exp{args.exp_id}/saved_models/{args.epoch_id:04d}.pth')

    config = Config.fromfile(os.path.join(script_dir, 'eval_cfg.py'))
    if train_cfg.get('pair_cfg', None) is not None:
        config.data_dir = train_cfg.pair_cfg.data_dir
    config.image_size = train_cfg.image_size
    config.update(dict(
        epoch_id=args.epoch_id,
        split=args.split,
        save_dir=save_dir,
        load_model=load_model,
        model_cfg=train_cfg.model_cfg,
        registration_cfg=train_cfg.registration_cfg,
        # a model is pyramidal iff it was trained with a scale pyramid
        pyramid=train_cfg.get('scale_pyramid', None) is not None,
        save_image=args.save_images,
        rigid=args.rigid,
        flow_idx=args.flow_idx,
        scale=args.scale,
    ))
    trained_on = train_cfg.get('split', None)
    if trained_on == args.split:
        print(f'[eval] WARNING: this model was trained on split={trained_on}, '
              f'which is the half being scored.')
    print(f'[eval] split={config.split} pyramid={config.pyramid} '
          f'prealign={"rigid" if config.rigid else "translation"} '
          f'flow_idx={config.flow_idx} scale={config.scale or "from config"}')

    set_seed(2023)
    eval(config)
