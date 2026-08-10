from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union, Literal

import torch
import numpy as np
from monai.config import DtypeLike, IndexSelection, KeysCollection
from monai.transforms.transform import MapTransform, Transform
from monai.config.type_definitions import NdarrayOrTensor
import SimpleITK as sitk
import nibabel as nib


def load_flow(flow_path: str, pkg: Literal["demons", "ants", "niftyreg", "greedy"]):
    # displacement fields of ANTs and SimpleITK are both in Physical Point coordinate
    disp = sitk.ReadImage(flow_path)
    direction = torch.tensor(disp.GetDirection()).reshape(3, 3)
    spacing = torch.diag(torch.tensor(disp.GetSpacing()))
    # the computed Affine matrix exclude the Origin
    # since we are transforming the displacement vector in Physical Point coordinate
    # to Image Index coordinate, the Origin is not needed
    affine = torch.matmul(direction, spacing)
    # mapping from Image Index coordinate to Physical Point coordinate, so we need the inverse
    affine_inv = torch.linalg.inv(affine)
    # print(affine)
    # print(affine_inv)

    # sitk: (x,y,z) -> numpy:(z,y,x)
    disp_arr = sitk.GetArrayFromImage(disp)
    disp_arr = np.transpose(disp_arr, axes=(3, 2, 1, 0))  # (3,H,W,D)
    disp_tensor = torch.from_numpy(disp_arr).float()

    if pkg == "niftyreg":
        nifty_to_sitk = torch.tensor([-1.0, 0, 0, 0, -1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        # from nifty space to sitk space
        # the x, y axes are mirrored
        disp_tensor = torch.einsum("ij,jhwd->ihwd", nifty_to_sitk, disp_tensor)

    # Physical Point space displacement to Image Index space displacement
    disp_tensor = torch.einsum("ij,jhwd->ihwd", affine_inv, disp_tensor)
    return disp_tensor


class LoadFlowd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 pkg: Literal["demons", "ants", "niftyreg", "greedy"],
                 allow_missing_keys: bool = False,
                 ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.pkg = pkg

    def __call__(
            self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            flow = load_flow(d[key], self.pkg)
            d[key] = flow
        return d


class FlowRAS2LIA(MapTransform):
    # resign and reorder the flow field component after reorientation
    # from 'RAS' to 'LIA'.
    def __init__(self,
                 keys: KeysCollection,
                 src_axcodes: str = 'RAS',
                 tar_axcodes: str = 'LIA',
                 labels: Sequence[tuple[str, str]] = (("L", "R"), ("P", "A"), ("I", "S")),
                 allow_missing_keys: bool = False,
                 ):
        super().__init__(keys, allow_missing_keys)
        src_index = []
        tar_index = []
        neg_src_index = []
        for labs in labels:
            for i, (src_ax, tar_ax) in enumerate(zip(src_axcodes, tar_axcodes)):
                if src_ax in labs:
                    src_index.append(i)
                if tar_ax in labs:
                    tar_index.append(i)

        for src_i, tar_i in zip(src_index, tar_index):
            if not src_axcodes[src_i] == tar_axcodes[tar_i]:
                neg_src_index.append(src_i)

        self.src_index = src_index
        self.tar_index = tar_index
        self.neg_src_index = neg_src_index

    def __call__(
            self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            flow = d[key]
            # print(flow.shape)
            if len(self.neg_src_index) > 0:
                flow[self.neg_src_index] = -flow[self.neg_src_index]
            flow[self.src_index] = flow[self.tar_index]
            d[key] = flow
        return d
