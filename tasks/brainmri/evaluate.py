"""Evaluate a brain-MRI model trained on LUMIR against the held-out datasets.

The architecture is not re-specified here: train.py dumps the resolved config to
<method_folder>/exp<N>/train_configs.py, and this script reads the model_cfg
back from it. Only the evaluation data and the registration head come from
eval_configs/.

Examples:
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 100
    python evaluate.py -m ./pwc_outputs -exp 0 -epoch 100 --dataset oasis
    python evaluate.py -m ./pwc_iter_outputs -exp 0 -epoch 100 --flow-idx 0
"""

import gc
import os
import sys
import time
from functools import partial
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
from mmengine import Config
from monai.data import DataLoader
from monai.metrics import DiceHelper, compute_hausdorff_distance

from models import build_flow_estimator, build_metrics, build_registration_head
from models.metrics import calc_jac_dets, calc_measurements, get_identity_grid
from utils import (PairDataset, load_data_adni, load_data_ixi, load_data_LPBA,
                   load_data_Mindboggle, load_data_oasis, worker_init_fn)

# dataset key -> (display name, loader, config attribute)
DATASETS = {
    'oasis': ('OASIS', load_data_oasis, 'oasis_cfg'),
    'adni': ('ADNI', load_data_adni, 'adni_cfg'),
    'ixi': ('IXI', load_data_ixi, 'ixi_cfg'),
    'lpba': ('LPBA', load_data_LPBA, 'lpba_cfg'),
    'mindboggle': ('Mindboggle', load_data_Mindboggle, 'mindboggle_cfg'),
}


def get_fwd_flows(model, source, target):
    """Every flow a pyramid model produces, coarse to fine.

    Iterative models expose forward_eval(), which additionally returns the
    intermediate flow of each refinement iteration, not just the last one per
    level. Either way the final entry is the model's actual output, so
    flow_idx=-1 scores what the model would be used for in practice.
    """
    if hasattr(model, 'forward_eval'):
        return model.forward_eval(source, target)['fwd']
    fwd_flows, _ = model(source, target)
    return fwd_flows


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
    register = build_registration_head(reg_cfg)
    register.to(cfg.device)
    register.eval()

    # load data
    dataset_names, datasets = [], []
    for key in cfg.datasets:
        name, loader, cfg_attr = DATASETS[key]
        ds_cfg = cfg[cfg_attr]
        kwargs = dict(cache_rate=ds_cfg.cache_rate, num_workers=ds_cfg.num_workers)
        if key == 'oasis':
            kwargs['val'] = True
        dataset_names.append(name)
        datasets.append(loader(ds_cfg, **kwargs))
    datasets = [PairDataset(dataset, length=cfg.num_pairs) for dataset in datasets]
    for d in datasets:
        print(len(d))
    dataloaders = [DataLoader(dataset,
                              batch_size=1,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              ) for dataset in datasets]
    # metrics: sdlogjacdet, nonposjacdet, dice, hd95, ndv
    compute_jacdet = build_metrics(dict(type='sdlogjac'))
    compute_dice = DiceHelper(include_background=False, get_not_nans=True)
    compute_hausdorff = partial(compute_hausdorff_distance, include_background=False, percentile=95, )
    metric_funcs = dict(dice=compute_dice,
                        jacdet=compute_jacdet,
                        hd=compute_hausdorff)
    # get_identity_grid((3,H,W,D)) -> voxel-wise grid
    # calc_jac_dets(deformation)
    # calc_measurements(jac_dets, mask(image>0))
    id_grid = get_identity_grid(np.empty((3, *cfg.image_size)))
    os.makedirs(cfg.save_dir, exist_ok=True)

    for ds_name, dataloader in zip(dataset_names, dataloaders):
        print(ds_name)
        per_np_jacdet_list = []
        non_pos_jacdet_list = []
        sd_logjacdet_list = []
        ndv_list = []
        per_ndv_list = []
        reg_dice_list = []
        init_dice_list = []
        hd95_list = []
        init_hd95_list = []
        runtime_list = []
        torch.cuda.empty_cache()
        for data in dataloader:
            with torch.no_grad():
                source = data[0]['image'].float().to(cfg.device)
                target = data[1]['image'].float().to(cfg.device)
                source_oh = data[0]['label'].float().to(cfg.device)
                target_oh = data[1]['label'].float().to(cfg.device)

                target_fg = torch.where(target > 0.0, 1.0, 0.0)

                start = time.time()
                if cfg.pyramid:
                    fwd_flows = get_fwd_flows(model, source, target)
                    if not -len(fwd_flows) <= cfg.flow_idx < len(fwd_flows):
                        raise IndexError(
                            f'--flow-idx {cfg.flow_idx} is out of range for '
                            f'{len(fwd_flows)} pyramid levels.')
                    half_res_flow = fwd_flows[cfg.flow_idx]
                else:
                    if 'Dual' in cfg.model_cfg.type and cfg.model_cfg.get('bidirectional', cfg.model_cfg.get('config', {}).get('bidirectional', False)):
                        half_res_flow, _ = model(source, target)
                    else:
                        half_res_flow = model(source, target)
                full_res_flow, y_source, y_source_oh = register(half_res_flow, source, source_oh)
                runtime_list.append(time.time() - start)

                # dice
                init_dice, not_nans = metric_funcs['dice'](source_oh, target_oh)
                reg_dice, not_nans = metric_funcs['dice'](y_source_oh, target_oh)
                init_dice_list.append(init_dice.detach().cpu().numpy())
                reg_dice_list.append(reg_dice.detach().cpu().numpy())
                print('init_dsc:', np.mean(init_dice.detach().cpu().numpy()))
                print('reg_dsc:', np.mean(reg_dice.detach().cpu().numpy()))

                # jacdet
                log_jacdet, non_pos_jacdet = compute_jacdet(
                    full_res_flow.detach().cpu().numpy(),
                    target_fg.detach().cpu().numpy())
                non_pos_jacdet_list.append(non_pos_jacdet)
                sd_logjacdet_list.append(log_jacdet)
                per_np_jacdet = non_pos_jacdet / target_fg.sum(dim=(2, 3, 4)).detach().cpu().numpy()
                per_np_jacdet_list.append(per_np_jacdet)
                print('per_np_jacdet:', per_np_jacdet)

                # hd95
                init_hd = metric_funcs['hd'](source_oh, target_oh)
                init_hd95_list.append(init_hd.detach().cpu().numpy())
                reg_hd = metric_funcs['hd'](y_source_oh, target_oh)
                hd95_list.append(reg_hd.detach().cpu().numpy())
                print('hd95:', np.nanmean(reg_hd.detach().cpu().numpy()))
                # NOTE: hausdorff distance metric has memory leakage
                gc.collect()

                # ndv
                # B=1
                target_fg = target_fg.squeeze().detach().cpu().numpy()[1:-1, 1:-1, 1:-1]
                # full_res_flow: (1,3,H,W,D)
                disp_field = full_res_flow.squeeze().detach().cpu().numpy()
                def_field = disp_field + id_grid
                jac_dets = calc_jac_dets(def_field)
                _, _, ndv, _ = calc_measurements(jac_dets, target_fg)
                total_voxels = np.sum(target_fg)
                per_ndv = ndv / total_voxels
                ndv_list.append(ndv)
                per_ndv_list.append(per_ndv)
                print('ndv:', ndv, 'per_ndv:', per_ndv)

        init_dice_list = np.vstack(init_dice_list)
        reg_dice_list = np.vstack(reg_dice_list)
        init_hd95_list = np.vstack(init_hd95_list)
        hd95_list = np.vstack(hd95_list)
        non_pos_jacdet_list = np.concatenate(non_pos_jacdet_list, axis=0)
        sd_logjacdet_list = np.concatenate(sd_logjacdet_list, axis=0)
        per_np_jacdet_list = np.concatenate(per_np_jacdet_list, axis=0)
        runtime_list = np.array(runtime_list)
        ndv_list = np.array(ndv_list)
        per_ndv_list = np.array(per_ndv_list)
        print(hd95_list.shape, per_np_jacdet_list.shape, ndv_list.shape, per_ndv_list.shape, runtime_list.shape)

        # non-default pyramid levels get their own file, so a sweep over
        # --flow-idx does not overwrite the headline numbers
        suffix = '' if cfg.flow_idx == -1 and cfg.scale is None \
            else f'_s{cfg.scale}_f{cfg.flow_idx}'
        np.savez(
            file=os.path.join(cfg.save_dir, ds_name + suffix + '.npz'),
            init_dice=init_dice_list,
            reg_dice=reg_dice_list,
            init_hd95=init_hd95_list,
            hd95=hd95_list,
            non_pos_jacdet=non_pos_jacdet_list,
            sd_logjacdet=sd_logjacdet_list,
            per_np_jacdet=per_np_jacdet_list,
            ndv=ndv_list,
            per_ndv=per_ndv_list,
            runtime=runtime_list
        )

        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print("[+] Maximum memory:\t{:.2f}GB".format(max_mem_mb))
        max_mem_re = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print("[+] Maximum memory:\t{:.2f}GB".format(max_mem_re))


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
                   default='all',
                   choices=list(DATASETS) + ['all'],
                   help='evaluation dataset (default: all five)')
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
    p.add_argument('--eval-config',
                   default='./eval_configs/_base_.py',
                   type=str,
                   help='evaluation data config (use local_base_.py for a '
                        'low-cache-rate local run)')
    args = p.parse_args()
    save_dir = os.path.join(args.method_folder, f'exp{args.exp_id}/eval_200pair')
    load_model = os.path.join(args.method_folder, f'exp{args.exp_id}/saved_models/{args.epoch_id:04d}.pth')
    train_cfg = Config.fromfile(os.path.join(args.method_folder, f'exp{args.exp_id}/train_configs.py'))
    config = Config.fromfile(args.eval_config)
    config.update(dict(
        save_dir=save_dir,
        load_model=load_model,
        model_cfg=train_cfg.model_cfg,
        # a model is pyramidal iff it was trained with a scale pyramid
        pyramid=train_cfg.get('scale_pyramid', None) is not None,
        flow_idx=args.flow_idx,
        scale=args.scale,
        datasets=list(DATASETS) if args.dataset == 'all' else [args.dataset],
    ))
    print(f'[eval] datasets={config.datasets} pyramid={config.pyramid} '
          f'flow_idx={config.flow_idx} scale={config.scale or "from config"}')
    set_seed(2023)
    infer(config)
