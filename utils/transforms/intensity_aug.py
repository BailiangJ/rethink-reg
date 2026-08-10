from __future__ import annotations

from typing import (Callable, Hashable, Iterable, Mapping, Optional, Sequence,
                    Union)

import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor, NdarrayTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.intensity.array import (MedianSmooth, RandBiasField,
                                              RandGaussianNoise,
                                              RandGaussianSharpen,
                                              RandGaussianSmooth)
from monai.transforms.transform import (MapTransform, RandomizableTransform,
                                        Transform)
from monai.transforms.utils import is_positive
from monai.transforms.utils_pytorch_numpy_unification import (clip, percentile,
                                                              where)
from monai.utils import convert_to_tensor, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix, TransformBackends

__all__ = [
    'AdjustContrast', 'RandAdjustContrast', 'RandAdjustContrastd',
    'MedianSmoothd', 'RandGaussianNoised', 'RandGaussianSmoothd',
    'RandGaussianSharpend', 'RandBiasFieldd'
]


class AdjustContrast(Transform):
    """Changes image intensity with gamma transform. Each pixel/voxel intensity is
    updated as::

        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min

    Args:
        gamma: gamma value to adjust the contrast as function.
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self,
                 gamma: float,
                 upper: float = 99.9,
                 invert_image: bool = False,
                 retain_stats: bool = False) -> None:
        if not isinstance(gamma, (int, float)):
            raise ValueError(
                f'gamma must be a float or int number, got {type(gamma)} {gamma}.'
            )
        self.gamma = gamma
        self.upper = upper
        self.invert_image = invert_image
        self.retain_stats = retain_stats

    def __call__(self, img: NdarrayOrTensor, gamma=None) -> NdarrayOrTensor:
        """Apply the transform to `img`.

        gamma: gamma value to adjust the contrast as function.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        gamma = gamma if gamma is not None else self.gamma

        if self.invert_image:
            img = -img

        if self.retain_stats:
            mn = img.mean()
            sd = img.std()

        epsilon = 1e-7
        img_min = img.min()
        img_range = percentile(img, self.upper) - img_min
        ret: NdarrayOrTensor = (
            (img - img_min) /
            float(img_range + epsilon))**gamma * img_range + img_min

        if self.retain_stats:
            # zero mean and normalize
            ret = ret - ret.mean()
            ret = ret / (ret.std() + 1e-8)
            # restore old mean and standard deviation
            ret = sd * ret + mn

        if self.invert_image:
            ret = -ret

        return ret


class RandAdjustContrast(RandomizableTransform):
    """Randomly changes image intensity with gamma transform. Each pixel/voxel intensity
    is updated as:

        x = ((x - min) / intensity_range) ^ gamma * intensity_range + min

    Args:
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
    """

    backend = AdjustContrast.backend

    def __init__(self,
                 prob: float = 0.1,
                 gamma: Union[Sequence[float], float] = (0.5, 4.5),
                 upper: float = 99.9,
                 invert_image: bool = False,
                 retain_stats: bool = False) -> None:
        RandomizableTransform.__init__(self, prob)

        if isinstance(gamma, (int, float)):
            if gamma <= 0.5:
                raise ValueError(
                    f'if gamma is a number, must greater than 0.5 and value is picked from (0.5, gamma), got {gamma}'
                )
            self.gamma = (0.5, gamma)
        elif len(gamma) != 2:
            raise ValueError('gamma should be a number or pair of numbers.')
        else:
            self.gamma = (min(gamma), max(gamma))

        self.gamma_value: float = 1.0
        self.upper: float = upper
        self.invert_image: bool = invert_image
        self.retain_stats: bool = retain_stats

        self.adjust_contrast = AdjustContrast(self.gamma_value,
                                              self.upper,
                                              invert_image=self.invert_image,
                                              retain_stats=self.retain_stats)

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.gamma_value = self.R.uniform(low=self.gamma[0],
                                          high=self.gamma[1])

    def __call__(self,
                 img: NdarrayOrTensor,
                 randomize: bool = True) -> NdarrayOrTensor:
        """Apply the transform to `img`."""
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()

        if not self._do_transform:
            return img

        if self.gamma_value is None:
            raise RuntimeError(
                'gamma_value is not set, please call `randomize` function first.'
            )

        return self.adjust_contrast(img, self.gamma_value)


class RandAdjustContrastd(RandomizableTransform, MapTransform):
    """Dictionary-based version :py:class:`monai.transforms.RandAdjustContrast`.
    Randomly changes image intensity with gamma transform. Each pixel/voxel intensity is
    updated as:

        `x = ((x - min) / intensity_range) ^ gamma * intensity_range + min`

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        prob: Probability of adjustment.
        gamma: Range of gamma values.
            If single number, value is picked from (0.5, gamma), default is (0.5, 4.5).
        invert_image: whether to invert the image before applying gamma augmentation. If True, multiply all intensity
            values with -1 before the gamma transform and again after the gamma transform. This behaviour is mimicked
            from `nnU-Net <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        retain_stats: if True, applies a scaling factor and an offset to all intensity values after gamma transform to
            ensure that the output intensity distribution has the same mean and standard deviation as the intensity
            distribution of the input. This behaviour is mimicked from `nnU-Net
            <https://www.nature.com/articles/s41592-020-01008-z>`_, specifically `this
            <https://github.com/MIC-DKFZ/batchgenerators/blob/7fb802b28b045b21346b197735d64f12fbb070aa/batchgenerators/augmentations/color_augmentations.py#L107>`_
            function.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandAdjustContrast.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        gamma: Union[Sequence[float], float] = (0.5, 4.5),
        upper: float = 99.9,
        invert_image: bool = False,
        retain_stats: bool = False,
        new_key_postfix: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.adjuster = RandAdjustContrast(gamma=gamma,
                                           upper=upper,
                                           prob=1.0,
                                           invert_image=invert_image,
                                           retain_stats=retain_stats)
        self.invert_image = invert_image
        self.new_key_postfix = new_key_postfix

    def set_random_state(
            self,
            seed: int | None = None,
            state: np.random.RandomState | None = None) -> RandAdjustContrastd:
        super().set_random_state(seed, state)
        self.adjuster.set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random gamma value
        self.adjuster.randomize(None)
        for key in self.key_iterator(d):
            new_key = key if self.new_key_postfix is None else key + self.new_key_postfix
            d[new_key] = self.adjuster(d[key], randomize=False)
        return d


class MedianSmoothd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.MedianSmooth`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        radius: if a list of values, must match the count of spatial dimensions of input data,
            and apply every value in the list to 1 spatial dimension. if only 1 value provided,
            use it for all spatial dimensions.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = MedianSmooth.backend

    def __init__(self,
                 keys: KeysCollection,
                 radius: Optional[Sequence[int], int],
                 new_key_postfix: Optional[str] = None,
                 allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = MedianSmooth(radius)
        self.new_key_postfix = new_key_postfix

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            new_key = key if self.new_key_postfix is None else key + self.new_key_postfix
            d[new_key] = self.converter(d[key])
        return d


class RandGaussianNoised(RandomizableTransform, MapTransform):
    """Dictionary-based version :py:class:`monai.transforms.RandGaussianNoise`. Add
    Gaussian noise to image. This transform assumes all the expected fields have same
    shape, if want to add different noise for every field, please use this transform
    separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        prob: Probability to add Gaussian noise.
        mean: Mean or “centre” of the distribution.
        std: Standard deviation (spread) of distribution.
        dtype: output data type, if None, same as input image. defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandGaussianNoise.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        mean: float = 0.0,
        std: float = 0.1,
        dtype: DtypeLike = np.float32,
        new_key_postfix: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_gaussian_noise = RandGaussianNoise(mean=mean,
                                                     std=std,
                                                     prob=1.0,
                                                     dtype=dtype)
        self.new_key_postfix = new_key_postfix

    def set_random_state(
            self,
            seed: int | None = None,
            state: np.random.RandomState | None = None) -> RandGaussianNoised:
        super().set_random_state(seed, state)
        self.rand_gaussian_noise.set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random noise
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.rand_gaussian_noise.randomize(d[first_key])

        for key in self.key_iterator(d):
            new_key = key if self.new_key_postfix is None else key + self.new_key_postfix
            d[new_key] = self.rand_gaussian_noise(img=d[key], randomize=False)
        return d


class RandGaussianSmoothd(RandomizableTransform, MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.GaussianSmooth`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        sigma_x: randomly select sigma value for the first spatial dimension.
        sigma_y: randomly select sigma value for the second spatial dimension if have.
        sigma_z: randomly select sigma value for the third spatial dimension if have.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.
        prob: probability of Gaussian smooth.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandGaussianSmooth.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma_x: tuple[float, float] = (0.25, 1.5),
        sigma_y: tuple[float, float] = (0.25, 1.5),
        sigma_z: tuple[float, float] = (0.25, 1.5),
        approx: str = 'erf',
        prob: float = 0.1,
        new_key_postfix: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_smooth = RandGaussianSmooth(sigma_x=sigma_x,
                                              sigma_y=sigma_y,
                                              sigma_z=sigma_z,
                                              approx=approx,
                                              prob=1.0)
        self.new_key_postfix = new_key_postfix

    def set_random_state(
            self,
            seed: int | None = None,
            state: np.random.RandomState | None = None) -> RandGaussianSmoothd:
        super().set_random_state(seed, state)
        self.rand_smooth.set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma
        self.rand_smooth.randomize(None)
        for key in self.key_iterator(d):
            new_key = key if self.new_key_postfix is None else key + self.new_key_postfix
            d[new_key] = self.rand_smooth(d[key], randomize=False)
        return d


class RandGaussianSharpend(RandomizableTransform, MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.GaussianSharpen`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        sigma1_x: randomly select sigma value for the first spatial dimension of first gaussian kernel.
        sigma1_y: randomly select sigma value for the second spatial dimension(if have) of first gaussian kernel.
        sigma1_z: randomly select sigma value for the third spatial dimension(if have) of first gaussian kernel.
        sigma2_x: randomly select sigma value for the first spatial dimension of second gaussian kernel.
            if only 1 value `X` provided, it must be smaller than `sigma1_x` and randomly select from [X, sigma1_x].
        sigma2_y: randomly select sigma value for the second spatial dimension(if have) of second gaussian kernel.
            if only 1 value `Y` provided, it must be smaller than `sigma1_y` and randomly select from [Y, sigma1_y].
        sigma2_z: randomly select sigma value for the third spatial dimension(if have) of second gaussian kernel.
            if only 1 value `Z` provided, it must be smaller than `sigma1_z` and randomly select from [Z, sigma1_z].
        alpha: randomly select weight parameter to compute the final result.
        approx: discrete Gaussian kernel type, available options are "erf", "sampled", and "scalespace".
            see also :py:meth:`monai.networks.layers.GaussianFilter`.
        prob: probability of Gaussian sharpen.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandGaussianSharpen.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigma1_x: tuple[float, float] = (0.5, 1.0),
        sigma1_y: tuple[float, float] = (0.5, 1.0),
        sigma1_z: tuple[float, float] = (0.5, 1.0),
        sigma2_x: tuple[float, float] | float = 0.5,
        sigma2_y: tuple[float, float] | float = 0.5,
        sigma2_z: tuple[float, float] | float = 0.5,
        alpha: tuple[float, float] = (10.0, 30.0),
        approx: str = 'erf',
        prob: float = 0.1,
        new_key_postfix: Optional[str] = None,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_sharpen = RandGaussianSharpen(
            sigma1_x=sigma1_x,
            sigma1_y=sigma1_y,
            sigma1_z=sigma1_z,
            sigma2_x=sigma2_x,
            sigma2_y=sigma2_y,
            sigma2_z=sigma2_z,
            alpha=alpha,
            approx=approx,
            prob=1.0,
        )
        self.new_key_postfix = new_key_postfix

    def set_random_state(
            self,
            seed: int | None = None,
            state: np.random.RandomState | None = None
    ) -> RandGaussianSharpend:
        super().set_random_state(seed, state)
        self.rand_sharpen.set_random_state(seed, state)
        return self

    def __call__(
        self, data: dict[Hashable,
                         NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random sigma1, sigma2, etc.
        self.rand_sharpen.randomize(None)
        for key in self.key_iterator(d):
            new_key = key if self.new_key_postfix is None else key + self.new_key_postfix
            d[new_key] = self.rand_sharpen(d[key], randomize=False)
        return d


class RandBiasFieldd(RandomizableTransform, MapTransform):
    """Dictionary-based version :py:class:`monai.transforms.RandBiasField`."""

    backend = RandBiasField.backend

    def __init__(
        self,
        keys: KeysCollection,
        degree: int = 3,
        coeff_range: tuple[float, float] = (0.0, 0.1),
        dtype: DtypeLike = np.float32,
        prob: float = 0.1,
        new_key_postfix: Optional[str] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            degree: degree of freedom of the polynomials. The value should be no less than 1.
                Defaults to 3.
            coeff_range: range of the random coefficients. Defaults to (0.0, 0.1).
            dtype: output data type, if None, same as input image. defaults to float32.
            prob: probability to do random bias field.
            allow_missing_keys: don't raise exception if key is missing.

        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.rand_bias_field = RandBiasField(degree=degree,
                                             coeff_range=coeff_range,
                                             dtype=dtype,
                                             prob=1.0)
        self.new_key_postfix = new_key_postfix

    def set_random_state(
            self,
            seed: int | None = None,
            state: np.random.RandomState | None = None) -> RandBiasFieldd:
        super().set_random_state(seed, state)
        self.rand_bias_field.set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random bias factor
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.rand_bias_field.randomize(img_size=d[first_key].shape[1:])

        for key in self.key_iterator(d):
            new_key = key if self.new_key_postfix is None else key + self.new_key_postfix
            d[new_key] = self.rand_bias_field(d[key], randomize=False)
        return d
