from typing import Dict, Hashable, Mapping

import numpy as np
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform
#from totalsegmentator.map_to_binary import class_map
CT_CLASS_MAP = {1: 'spleen',
                2: 'kidney_right',
                3: 'kidney_left',
                4: 'gallbladder',
                5: 'liver',
                6: 'stomach',
                7: 'pancreas',
                8: 'adrenal_gland_right',
                9: 'adrenal_gland_left',
                10: "lung_upper_lobe_left",
                11: "lung_lower_lobe_left",
                12: "lung_upper_lobe_right",
                13: "lung_middle_lobe_right",
                14: "lung_lower_lobe_right",
                15: 'esophagus',
                18: 'small_bowel',
                19: 'duodenum',
                20: 'colon',
                21: 'urinary_bladder',
                22: 'prostate',
                25: 'sacrum',
                51: 'heart',
                52: 'aorta',
                63: 'inferior_vena_cava',
                64: 'portal_vein_and_splenic_vein',
                65: 'iliac_artery_left',
                66: 'iliac_artery_right',
                67: 'iliac_vena_left',
                68: 'iliac_vena_right',
                69: 'humerus_left',
                70: 'humerus_right',
                71: 'scapula_left',
                72: 'scapula_right',
                73: 'clavicula_left',
                74: 'clavicula_right',
                75: 'femur_left',
                76: 'femur_right',
                77: 'hip_left',
                78: 'hip_right',
                79: 'spinal_cord',
                80: 'gluteus_maximus_left',
                81: 'gluteus_maximus_right',
                82: 'gluteus_medius_left',
                83: 'gluteus_medius_right',
                84: 'gluteus_minimus_left',
                85: 'gluteus_minimus_right',
                86: 'autochthon_left',
                87: 'autochthon_right',
                88: 'iliopsoas_left',
                89: 'iliopsoas_right',
                90: 'brain'}

MR_CLASS_MAP = {1: 'spleen',
                2: 'kidney_right',
                3: 'kidney_left',
                4: 'gallbladder',
                5: 'liver',
                6: 'stomach',
                7: 'pancreas',
                8: 'adrenal_gland_right',
                9: 'adrenal_gland_left',
                10: "lung_left",
                11: "lung_right",
                12: 'esophagus',
                13: 'small_bowel',
                14: 'duodenum',
                15: 'colon',
                16: 'urinary_bladder',
                17: 'prostate',
                18: 'sacrum',
                21: 'spinal_cord',
                22: 'heart',
                23: 'aorta',
                24: 'inferior_vena_cava',
                25: 'portal_vein_and_splenic_vein',
                26: 'iliac_artery_left',
                27: 'iliac_artery_right',
                28: 'iliac_vena_left',
                29: 'iliac_vena_right',
                30: 'humerus_left',
                31: 'humerus_right',
                32: 'scapula_left',
                33: 'scapula_right',
                34: 'clavicula_left',
                35: 'clavicula_right',
                36: 'femur_left',
                37: 'femur_right',
                38: 'hip_left',
                39: 'hip_right',
                40: 'gluteus_maximus_left',
                41: 'gluteus_maximus_right',
                42: 'gluteus_medius_left',
                43: 'gluteus_medius_right',
                44: 'gluteus_minimus_left',
                45: 'gluteus_minimus_right',
                46: 'autochthon_left',
                47: 'autochthon_right',
                48: 'iliopsoas_left',
                49: 'iliopsoas_right',
                50: 'brain'}

COMMON_ORGANS = ['spleen',
                 'kidney_right',
                 'kidney_left',
                 'gallbladder',
                 'liver',
                 'stomach',
                 'pancreas',
                 'adrenal_gland_right',
                 'adrenal_gland_left',
                 'esophagus',
                 'small_bowel',
                 'duodenum',
                 'colon',
                 'urinary_bladder',
                 'prostate',
                 'sacrum',
                 'spinal_cord',
                 'heart',
                 'aorta',
                 # 'inferior_vena_cava',
                 # 'portal_vein_and_splenic_vein',
                 # 'iliac_artery_left',
                 # 'iliac_artery_right',
                 # 'iliac_vena_left',
                 # 'iliac_vena_right',
                 # 'humerus_left',
                 # 'humerus_right',
                 # 'scapula_left',
                 # 'scapula_right',
                 # 'clavicula_left',
                 # 'clavicula_right',
                 # 'femur_left',
                 # 'femur_right',
                 # 'hip_left',
                 # 'hip_right',
                 # 'gluteus_maximus_left',
                 # 'gluteus_maximus_right',
                 # 'gluteus_medius_left',
                 # 'gluteus_medius_right',
                 # 'gluteus_minimus_left',
                 # 'gluteus_minimus_right',
                 # 'autochthon_left',
                 # 'autochthon_right',
                 # 'iliopsoas_left',
                 # 'iliopsoas_right',
                 # 'brain'
                 ]

LARGE_ORGANS = [
                'spleen',
                'kidney_right',
                'kidney_left',
                'liver',
]

CT_CLASS_MAP = {v:k for k,v in CT_CLASS_MAP.items()}
MR_CLASS_MAP = {v:k for k,v in MR_CLASS_MAP.items()}

class AbdomenOneHotd(MapTransform):
    def __init__(self, ct_mask_key, mr_mask_key, allow_missing_keys=False, min_voxels: int = 3000):
        super().__init__([ct_mask_key, mr_mask_key], allow_missing_keys)
        self.ct_mask_key = ct_mask_key
        self.mr_mask_key = mr_mask_key
        self.min_voxels = int(min_voxels)

    def __call__(
            self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        ct_mask = d[self.ct_mask_key].numpy()
        mr_mask = d[self.mr_mask_key].numpy()

        ###### COMMON ORGANS ######
        common_present_organs = []
        for organ in COMMON_ORGANS:
            ct_count = (ct_mask == CT_CLASS_MAP[organ]).sum()
            mr_count = (mr_mask == MR_CLASS_MAP[organ]).sum()
            ct_present = ct_count >= self.min_voxels if self.min_voxels > 0 else ct_count > 0
            mr_present = mr_count >= self.min_voxels if self.min_voxels > 0 else mr_count > 0

            if ct_present and mr_present:
                common_present_organs.append(organ)

        ct_oh = np.zeros((len(common_present_organs), *ct_mask.shape), dtype=np.uint8)
        mr_oh = np.zeros((len(common_present_organs), *mr_mask.shape), dtype=np.uint8)

        for i, organ in enumerate(common_present_organs):
            ct_oh[i] = (ct_mask == CT_CLASS_MAP[organ]).astype(np.uint8)
            mr_oh[i] = (mr_mask == MR_CLASS_MAP[organ]).astype(np.uint8)

        ###### LARGE ORGANS ######
        large_present_organs = []
        for organ in LARGE_ORGANS:
            ct_count = (ct_mask == CT_CLASS_MAP[organ]).sum()
            mr_count = (mr_mask == MR_CLASS_MAP[organ]).sum()
            ct_present = ct_count >= self.min_voxels if self.min_voxels > 0 else ct_count > 0
            mr_present = mr_count >= self.min_voxels if self.min_voxels > 0 else mr_count > 0

            if ct_present and mr_present:
                large_present_organs.append(organ)

        ct_oh_large = np.zeros((len(large_present_organs), *ct_mask.shape), dtype=np.uint8)
        mr_oh_large = np.zeros((len(large_present_organs), *mr_mask.shape), dtype=np.uint8)

        for i, organ in enumerate(large_present_organs):
            ct_oh_large[i] = (ct_mask == CT_CLASS_MAP[organ]).astype(np.uint8)
            mr_oh_large[i] = (mr_mask == MR_CLASS_MAP[organ]).astype(np.uint8)

        d['ct_oh'] = ct_oh
        d['mr_oh'] = mr_oh
        d['ct_oh_large'] = ct_oh_large
        d['mr_oh_large'] = mr_oh_large
        d['ct_mask_sub'] = np.argmax(ct_oh, axis=0)
        d['mr_mask_sub'] = np.argmax(mr_oh, axis=0)
        d['organs'] = common_present_organs
        d['organs_large'] = large_present_organs
        return d

# if __name__ == '__main__':
#     ct_class_map = class_map['total']
#     ct_class_map = {v: k for k, v in ct_class_map.items()}
#     mr_class_map = class_map['total_mr']
#     mr_class_map = {v: k for k, v in mr_class_map.items()}
#     common_organs = list(set(ct_class_map.keys()).intersection(set(mr_class_map.keys())))
#     common_ct_class_map = {k: ct_class_map[k] for k in common_organs}
#     common_mr_class_map = {k: mr_class_map[k] for k in common_organs}
#     common_ct_class_map = {v: k for k, v in common_ct_class_map.items()}
#     common_mr_class_map = {v: k for k, v in common_mr_class_map.items()}
#     common_ct_class_map = dict(sorted(common_ct_class_map.items()))
#     common_mr_class_map = dict(sorted(common_mr_class_map.items()))
#     print(common_mr_class_map.values())
