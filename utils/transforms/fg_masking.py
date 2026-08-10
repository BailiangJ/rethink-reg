from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union
import torch
import numpy as np
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform


class ForegroundMaskingd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 mask_keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.mask_keys = mask_keys

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, mask_key in zip(self.keys, self.mask_keys):
            if key in d and mask_key in d:
                d[key] = d[key] * (d[mask_key] > 0).astype(d[key].dtype)
        return d
