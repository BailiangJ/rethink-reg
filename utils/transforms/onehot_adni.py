from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union
import torch
import numpy as np
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform


class MindBoggleOneHotd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
            self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        # ADNI
        # classes_list = [
        #     0, 2, 3, 4, 7, 8, 10, 11, 12, 13,
        #     16, 17, 18, 26, 41, 42, 43,
        #     46, 47, 49, 50, 51, 52, 53, 54, 58,
        #     251, 252, 253, 254, 255
        # ]
        # Mindboggle manual:
        manual_classes_list = [0, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010,
                               1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019,
                               1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028,
                               1029, 1030, 1031, 1034, 1035, 2002, 2003, 2005, 2006,
                               2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                               2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,
                               2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035]
        # # Mindboggle exclude 30, 62, 72, 80, due to low volume
        aseg_classes_list = [
            0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 24, 26, 28, 31, 41, 43,
            44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
            63, 77, 85, 251, 252, 253, 254, 255,
        ]
        classes_list = manual_classes_list + aseg_classes_list
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


class ADNIOneHotd(MapTransform):
    def __init__(self,
                 keys: KeysCollection,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
            self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        # ADNI
        # classes_list = [
        #     0, 2, 3, 4, 7, 8, 10, 11, 12, 13,
        #     16, 17, 18, 26, 41, 42, 43,
        #     46, 47, 49, 50, 51, 52, 53, 54, 58,
        #     251, 252, 253, 254, 255
        # ]
        # corresponde to OASIS:
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
