# from train.segment_anything_training.modeling.mask_decoder import MaskDecoder
from .sam_enhancev4_23_6 import MaskDecoderEnhance
from .sam_enhance_MGD import MaskDecoderMGD
from .sam_enhance_MGDv3 import MaskDecoderMGDv3
from .sam_enhancev4_23_6_light import MaskDecoderEnhanceLight
from .sam_enhancev4_23_6_fusion_results import MaskDecoderEnhancefusion
from .sam_enhance_dwd import MaskDecoderEnhanceDWD
from .sam_hq import MaskDecoderHQ
from .sam_en_s import MaskDecoderEnhanceS
from .sam import MaskDecoderSAM


model_registry = {
    "sam-en": MaskDecoderEnhance,
    "sam-mgd": MaskDecoderMGD,
    "sam-mgdv3": MaskDecoderMGDv3,
    "sam-hq": MaskDecoderHQ,
    "sam-en-s": MaskDecoderEnhanceS,
    "sam": MaskDecoderSAM,
    "sam-en-light": MaskDecoderEnhanceLight,
    "sam-en-fusion": MaskDecoderEnhancefusion,
    "sam-en-dwd": MaskDecoderEnhanceDWD,
}



