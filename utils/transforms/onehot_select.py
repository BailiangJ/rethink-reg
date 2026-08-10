from typing import Dict, Hashable, Mapping, Sequence

import numpy as np
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection


class SelectOneHotd(MapTransform):
    """Convert segmentation labels to one-hot encoding based on a provided list of labels.

    Args:
        keys: keys of the corresponding items to be transformed.
        label_list: list of labels to convert to one-hot encoding.
        allow_missing_keys: don't raise exception if key is missing.
    """
    def __init__(
        self,
        keys: KeysCollection,
        label_list: Sequence[int],
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.label_list = label_list
        self.num_classes = len(label_list)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key]
            # Remove channel dimension if present
            mask = mask.squeeze()
            one_hot = np.zeros((self.num_classes, *mask.shape), dtype=np.uint8)

            for i, label in enumerate(self.label_list):
                one_hot[i][mask == label] = 1

            d[key] = one_hot
        return d
