from typing import Dict, Hashable, Mapping

import numpy as np
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks.utils import one_hot
from monai.transforms.transform import MapTransform
from monai.transforms.utils_pytorch_numpy_unification import unique


class OASISOneHotd(MapTransform):
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


class OneHotd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
            self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key in self.key_iterator(d):
            mask = d[key].squeeze()
            unique_labels = unique(mask)
            num_classes = len(unique_labels)

            one_hot = np.zeros((num_classes, *mask.shape), dtype=np.uint8)

            for i, c in enumerate(unique_labels):
                one_hot[i][mask == c] = 1

            d[key] = one_hot

        return d


class FreesurferOneHotd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
            self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        # Freesurfer label map
        # that corresponds to OASIS:
        classes_list = [
            0, 2, 3, 4, 5, 7, 8, 10,
            11, 12, 13, 14, 15, 16, 17, 18,
            26,
            28, 30, 31,
            41, 42, 43, 44, 46, 47, 49,
            50, 51, 52, 53, 54, 58,
            60, 62, 63
        ]
        num_classes = len(classes_list)

        for key in self.key_iterator(d):
            mask = d[key]
            # the first dimension of mask is channel
            mask = mask.squeeze()
            one_hot = np.zeros((num_classes, *mask.shape), dtype=np.uint8)

            for i, c in enumerate(classes_list):
                one_hot[i][mask == c] = 1

            d[key] = one_hot
        return d
