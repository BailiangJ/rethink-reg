from typing import Dict, Hashable, Mapping

import numpy as np
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform

CT_CLASS_MAP = {1: "spleen",
                2: "kidney_right",
                3: "kidney_left",
                5: "liver",
                6: "stomach",
                7: "pancreas",
                8: "adrenal_gland_right",
                9: "adrenal_gland_left",
                10: "lung_upper_lobe_left",
                11: "lung_lower_lobe_left",
                12: "lung_upper_lobe_right",
                13: "lung_middle_lobe_right",
                14: "lung_lower_lobe_right",
                15: "esophagus",
                16: "trachea",
                17: "thyroid_gland",
                20: "colon",
                31: "vertebrae_L1",
                32: "vertebrae_T12",
                33: "vertebrae_T11",
                34: "vertebrae_T10",
                35: "vertebrae_T9",
                36: "vertebrae_T8",
                37: "vertebrae_T7",
                38: "vertebrae_T6",
                39: "vertebrae_T5",
                40: "vertebrae_T4",
                41: "vertebrae_T3",
                42: "vertebrae_T2",
                43: "vertebrae_T1",
                44: "vertebrae_C7",
                51: "heart",
                52: "aorta",
                53: "pulmonary_vein",
                54: "brachiocephalic_trunk",
                55: "subclavian_artery_right",
                56: "subclavian_artery_left",
                57: "common_carotid_artery_right",
                58: "common_carotid_artery_left",
                59: "brachiocephalic_vein_left",
                60: "brachiocephalic_vein_right",
                61: "atrial_appendage_left",
                62: "superior_vena_cava",
                63: "inferior_vena_cava",
                64: "portal_vein_and_splenic_vein",
                69: "humerus_left",
                70: "humerus_right",
                71: "scapula_left",
                72: "scapula_right",
                73: "clavicula_left",
                74: "clavicula_right",
                79: "spinal_cord",
                86: "autochthon_left",
                87: "autochthon_right",
                92: "rib_left_1",
                93: "rib_left_2",
                94: "rib_left_3",
                95: "rib_left_4",
                96: "rib_left_5",
                97: "rib_left_6",
                98: "rib_left_7",
                99: "rib_left_8",
                100: "rib_left_9",
                101: "rib_left_10",
                102: "rib_left_11",
                103: "rib_left_12",
                104: "rib_right_1",
                105: "rib_right_2",
                106: "rib_right_3",
                107: "rib_right_4",
                108: "rib_right_5",
                109: "rib_right_6",
                110: "rib_right_7",
                111: "rib_right_8",
                112: "rib_right_9",
                113: "rib_right_10",
                114: "rib_right_11",
                115: "rib_right_12",
                116: "sternum",
                117: "costal_cartilages"}

class LungCTOneHotd(MapTransform):
    def __init__(
            self,
            keys: list,
            allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed
            allow_missing_keys: don't raise exception if key is missing
        """
        super().__init__(keys, allow_missing_keys)
        self.class_map = CT_CLASS_MAP

    def _get_common_labels(self, exp_label: np.ndarray, inp_label: np.ndarray) -> list:
        """Get labels that are present in both exp_label and inp_label."""
        exp_unique = set(np.unique(exp_label).astype(int).tolist())
        inp_unique = set(np.unique(inp_label).astype(int).tolist())

        # Only consider labels in CT_CLASS_MAP
        exp_classes = {label for label in exp_unique if label in self.class_map}
        inp_classes = {label for label in inp_unique if label in self.class_map}

        # Common labels (present in both)
        common_classes = sorted(exp_classes.intersection(inp_classes))

        return common_classes

    def _convert_to_one_hot(self, label: np.ndarray, common_classes: list) -> np.ndarray:
        """Convert label to one-hot format using common classes and background."""
        # # Add background (0) to the beginning of the class list if not already there
        # if 0 not in common_classes:
        #     classes_with_bg = [0] + common_classes
        # else:
        #     classes_with_bg = common_classes
        classes_with_bg = common_classes

        # Create one-hot encoded array
        shape = (len(classes_with_bg),) + label.shape
        one_hot = np.zeros(shape, dtype=np.float32)

        # Fill one-hot channels for each class including background
        for i, class_id in enumerate(classes_with_bg):
            mask = (label == class_id)
            one_hot[i][mask] = 1.0

        return one_hot

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        # Get the common labels
        exp_label = d['exp_label'].squeeze().astype(int)
        inp_label = d['inp_label'].squeeze().astype(int)

        common_classes = self._get_common_labels(exp_label, inp_label)

        # # Always include background (0) in the class list
        # if 0 not in common_classes:
        #     common_classes = [0] + common_classes
        # else:
        #     common_classes = sorted(common_classes)  # Ensure background is first

        # Store the list of classes used for one-hot encoding
        d['one_hot_classes'] = common_classes

        # Convert labels to one-hot
        for key in self.keys:
            if key in d:
                label = d[key].squeeze().astype(int)
                d[key] = self._convert_to_one_hot(label, common_classes)

        return d