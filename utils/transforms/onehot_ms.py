from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.transforms.utils_pytorch_numpy_unification import unique
from copy import deepcopy


class LesionOneHotd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        # Temporal global lesions mask
        lesions_mask = None

        for key in self.key_iterator(d):
            mask = d[key].squeeze()
            unique_labels = unique(mask)
            num_classes = len(unique_labels)

            one_hot = np.zeros((num_classes, *mask.shape), dtype=np.uint8)

            for i, c in enumerate(unique_labels):
                one_hot[i][mask == c] = 1

            if lesions_mask is None:
                lesions_mask = deepcopy(one_hot[[0]])
                lesions_mask = 1 - lesions_mask
                # print(lesions_mask.sum())
            else:
                # print((1 - one_hot[[0]]).sum())
                lesions_mask += (1 - one_hot[[0]])

            d[key] = one_hot

        d['global_lesions'] = lesions_mask
        return d
