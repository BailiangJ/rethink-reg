from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union
import torch
import numpy as np
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.networks.utils import one_hot

class BrainOneHotd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            label = one_hot(d[key], num_classes=36, dim=0)
            d[key] = label
        return d
