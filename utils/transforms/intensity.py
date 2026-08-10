from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
from monai.config import DtypeLike, IndexSelection, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.intensity.array import ScaleIntensityRange
from monai.transforms.transform import MapTransform
from monai.transforms.utils_pytorch_numpy_unification import (clip, percentile,
                                                              where)


class ScaleIntensityRanged(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.ScaleIntensityRange`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        a_min: intensity original range min.
        upper: intensity upper percentile.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = ScaleIntensityRange.backend

    def __init__(
            self,
            keys: KeysCollection,
            a_min: float,
            upper: float,
            b_min: float,
            b_max: float,
            clip: bool = False,
            relative: bool = False,
            dtype: DtypeLike = np.float32,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.a_min = a_min
        self.b_min = b_min
        self.b_max = b_max
        self.upper = upper
        self.clip = clip
        self.relative = relative
        self.dtype = dtype

    def __call__(
            self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]

            a_max: float = percentile(img, self.upper)
            # a_max: float = percentile(img[img>0.0], self.upper)
            # print('a_max:', a_max, 'max:', img.max())

            if self.relative:
                b_max = ((self.b_max - self.b_min) *
                         (self.upper / 100.0)) + self.b_min
            else:
                b_max = self.b_max
            scaler = ScaleIntensityRange(a_min=self.a_min,
                                         a_max=a_max,
                                         b_min=self.b_min,
                                         b_max=b_max,
                                         clip=self.clip,
                                         dtype=self.dtype)
            img_sp = scaler(d[key])
            d[key] = clip(img_sp, self.b_min, None)
        return d
